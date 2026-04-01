"""
Real-Time Processing Module for LLM Meta-Analysis

This module provides streaming/real-time processing capabilities for
handling clinical trial documents as they become available.
"""

import os
import json
import asyncio
import time
from typing import Dict, List, Optional, Callable, AsyncIterator, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
from pathlib import Path
import queue
import threading

from models.model import Model
from utils import load_json_file, save_json_file, format_example_with_prompt_template
from templates import DatasetTemplates


@dataclass
class ProcessingResult:
    """Result from processing a single document"""
    pmcid: str
    status: str  # 'pending', 'processing', 'completed', 'failed'
    task: str
    model: str
    timestamp: str
    output: Optional[Dict] = None
    error: Optional[str] = None
    processing_time: float = 0.0


@dataclass
class ProcessingStats:
    """Statistics for real-time processing"""
    total_submitted: int = 0
    total_completed: int = 0
    total_failed: int = 0
    total_pending: int = 0
    average_processing_time: float = 0.0
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())


class DocumentQueue:
    """Thread-safe queue for document processing"""

    def __init__(self, max_size: int = 1000):
        self.queue = queue.Queue(maxsize=max_size)
        self.pending: Dict[str, Dict] = {}
        self.completed: Dict[str, ProcessingResult] = {}
        self.lock = threading.Lock()

    def put(self, item: Dict) -> bool:
        """Add item to queue"""
        try:
            self.queue.put(item, block=False)
            with self.lock:
                self.pending[item['pmcid']] = item
            return True
        except queue.Full:
            return False

    def get(self, timeout: float = 1.0) -> Optional[Dict]:
        """Get item from queue"""
        try:
            item = self.queue.get(timeout=timeout)
            return item
        except queue.Empty:
            return None

    def mark_complete(self, pmcid: str, result: ProcessingResult) -> None:
        """Mark a document as complete"""
        with self.lock:
            if pmcid in self.pending:
                del self.pending[pmcid]
            self.completed[pmcid] = result

    def get_stats(self) -> Dict:
        """Get current queue statistics"""
        with self.lock:
            return {
                'pending': len(self.pending),
                'completed': len(self.completed),
                'queue_size': self.queue.qsize()
            }


class RealTimeProcessor:
    """
    Real-time processor for clinical trial documents.

    Features:
    - Streaming document processing
    - Parallel processing with configurable workers
    - Progress tracking and statistics
    - Callback hooks for events
    - Rate limiting for API-based models
    - Automatic retries on failures
    - Result streaming
    """

    def __init__(
        self,
        model: Model,
        model_name: str,
        task: str,
        max_workers: int = 4,
        rate_limit: float = 1.0,
        max_retries: int = 3
    ):
        """
        Initialize the real-time processor.

        :param model: Model instance for processing
        :param model_name: Name of the model
        :param task: Task type (outcome_type, binary_outcomes, continuous_outcomes)
        :param max_workers: Maximum number of parallel workers
        :param rate_limit: Minimum seconds between API calls (for API models)
        :param max_retries: Maximum number of retry attempts
        """
        self.model = model
        self.model_name = model_name
        self.task = task
        self.max_workers = max_workers
        self.rate_limit = rate_limit
        self.max_retries = max_retries

        # Processing state
        self.queue = DocumentQueue()
        self.stats = ProcessingStats()
        self.is_running = False
        self.workers: List[threading.Thread] = []

        # Callbacks
        self.on_document_complete: Optional[Callable] = None
        self.on_document_fail: Optional[Callable] = None
        self.on_progress: Optional[Callable] = None

        # Templates
        self.prompt_template = self._load_prompt_template()

        # Rate limiting
        self.last_api_call = 0

    def _load_prompt_template(self) -> DatasetTemplates:
        """Load appropriate prompt template for the task and model"""
        template_path = self.task
        if "gpt" in self.model_name or "claude" in self.model_name or "gemini" in self.model_name:
            template_path += "/gpt"
        elif "llama" in self.model_name.lower():
            template_path += "/llama3"
        else:
            template_path += "/" + self.model_name.lower()

        return DatasetTemplates(template_path)

    def submit_document(self, document: Dict, md_content: str) -> str:
        """
        Submit a document for processing.

        :param document: Document metadata (pmcid, intervention, comparator, outcome)
        :param md_content: Markdown content of the document
        :return: Document ID (pmcid)
        """
        item = {
            'pmcid': document.get('pmcid', ''),
            'document': document,
            'md_content': md_content,
            'submitted_at': time.time()
        }

        if self.queue.put(item):
            self.stats.total_submitted += 1
            self.stats.total_pending += 1
            return item['pmcid']
        else:
            raise RuntimeError("Queue is full")

    def submit_batch(self, documents: List[Dict], md_contents: Dict[str, str]) -> List[str]:
        """
        Submit multiple documents for processing.

        :param documents: List of document metadata
        :param md_contents: Dictionary mapping pmcid to markdown content
        :return: List of submitted document IDs
        """
        submitted_ids = []
        for doc in documents:
            pmcid = doc.get('pmcid', '')
            md_content = md_contents.get(pmcid, '')
            if md_content:
                try:
                    doc_id = self.submit_document(doc, md_content)
                    submitted_ids.append(doc_id)
                except RuntimeError:
                    break  # Queue is full
        return submitted_ids

    def _rate_limit_wait(self) -> None:
        """Wait to respect rate limit"""
        if self.model_name in ['gpt35', 'gpt4', 'claude', 'claude-opus', 'claude-sonnet', 'gemini', 'gemini-pro']:
            elapsed = time.time() - self.last_api_call
            if elapsed < self.rate_limit:
                time.sleep(self.rate_limit - elapsed)
        self.last_api_call = time.time()

    def _process_document(self, item: Dict) -> ProcessingResult:
        """Process a single document"""
        pmcid = item['pmcid']
        document = item['document']
        md_content = item['md_content']

        start_time = time.time()

        result = ProcessingResult(
            pmcid=pmcid,
            status='processing',
            task=self.task,
            model=self.model_name,
            timestamp=datetime.now().isoformat()
        )

        try:
            # Format with prompt template
            example = document.copy()
            example['abstract_and_results'] = md_content

            template_name = self.prompt_template.all_template_names[0]
            prompt = self.prompt_template[template_name]
            formatted = format_example_with_prompt_template(example, prompt)

            # Apply rate limiting
            self._rate_limit_wait()

            # Generate output
            max_tokens = self._get_max_tokens()
            output = self.model.generate_output(formatted['input'], max_new_tokens=max_tokens)

            result.output = {
                'pmcid': pmcid,
                'output': output,
                'intervention': document.get('intervention', ''),
                'comparator': document.get('comparator', ''),
                'outcome': document.get('outcome', ''),
                'task': self.task
            }
            result.status = 'completed'

        except Exception as e:
            result.status = 'failed'
            result.error = str(e)

        result.processing_time = time.time() - start_time
        return result

    def _worker_loop(self, worker_id: int) -> None:
        """Worker thread loop"""
        while self.is_running:
            item = self.queue.get(timeout=1.0)
            if item is None:
                continue

            retries = 0
            result = None

            while retries <= self.max_retries:
                try:
                    result = self._process_document(item)
                    break
                except Exception as e:
                    retries += 1
                    if retries > self.max_retries:
                        result = ProcessingResult(
                            pmcid=item['pmcid'],
                            status='failed',
                            task=self.task,
                            model=self.model_name,
                            timestamp=datetime.now().isoformat(),
                            error=str(e)
                        )

            # Update stats
            if result.status == 'completed':
                self.stats.total_completed += 1
            else:
                self.stats.total_failed += 1
            self.stats.total_pending -= 1

            # Mark complete
            self.queue.mark_complete(item['pmcid'], result)

            # Trigger callbacks
            if result.status == 'completed' and self.on_document_complete:
                self.on_document_complete(result)
            elif result.status == 'failed' and self.on_document_fail:
                self.on_document_fail(result)

            if self.on_progress:
                self.on_progress(self.get_progress())

    def _get_max_tokens(self) -> int:
        """Get max tokens for the current task"""
        token_limits = {
            'outcome_type': 5,
            'binary_outcomes': 50,
            'continuous_outcomes': 70
        }
        return token_limits.get(self.task, 50)

    def start(self) -> None:
        """Start the real-time processor"""
        if self.is_running:
            return

        self.is_running = True
        self.workers = []

        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker_loop, args=(i,))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

    def stop(self) -> None:
        """Stop the real-time processor"""
        self.is_running = False
        for worker in self.workers:
            worker.join(timeout=5.0)
        self.workers = []

    def get_progress(self) -> Dict:
        """Get current processing progress"""
        queue_stats = self.queue.get_stats()
        return {
            'total_submitted': self.stats.total_submitted,
            'total_completed': self.stats.total_completed,
            'total_failed': self.stats.total_failed,
            'total_pending': queue_stats['pending'],
            'queue_size': queue_stats['queue_size'],
            'progress_percent': (self.stats.total_completed / self.stats.total_submitted * 100)
                                if self.stats.total_submitted > 0 else 0
        }

    def get_results(self) -> List[ProcessingResult]:
        """Get all completed results"""
        return list(self.queue.completed.values())

    def get_result(self, pmcid: str) -> Optional[ProcessingResult]:
        """Get result for a specific document"""
        return self.queue.completed.get(pmcid)

    async def stream_results(self) -> AsyncIterator[ProcessingResult]:
        """
        Stream results as they complete.

        :yields: ProcessingResult as documents complete
        """
        completed_count = 0
        while self.is_running or self.queue.get_stats()['pending'] > 0:
            # Check for new results
            current_completed = self.stats.total_completed
            if current_completed > completed_count:
                new_results = [r for r in self.get_results()
                              if len([x for x in self.get_results() if x.pmcid == r.pmcid and
                                     list(self.queue.completed.values()).index(r) >= completed_count])]
                # Actually, let's just yield all results
                results = self.get_results()
                for result in results:
                    if result not in [r for r in []]:  # Track yielded
                        yield result
                completed_count = current_completed

            await asyncio.sleep(0.5)

    def save_results(self, output_path: str) -> None:
        """
        Save all completed results to a file.

        :param output_path: Path to save the results
        """
        results = self.get_results()
        output_data = [asdict(r) for r in results]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_json_file(output_path, output_data)


class RealTimeBatchProcessor:
    """
    Batch processor with real-time updates.

    Processes a batch of documents with streaming progress updates.
    """

    def __init__(self, model: Model, model_name: str, task: str):
        """
        Initialize the batch processor.

        :param model: Model instance
        :param model_name: Name of the model
        :param task: Task type
        """
        self.model = model
        self.model_name = model_name
        self.task = task
        self.processor = RealTimeProcessor(model, model_name, task)

    def process_batch(
        self,
        documents: List[Dict],
        md_contents: Dict[str, str],
        progress_callback: Optional[Callable] = None,
        result_callback: Optional[Callable] = None
    ) -> List[ProcessingResult]:
        """
        Process a batch of documents with progress updates.

        :param documents: List of document metadata
        :param md_contents: Dictionary mapping pmcid to markdown content
        :param progress_callback: Optional callback for progress updates
        :param result_callback: Optional callback for individual results
        :return: List of all processing results
        """
        # Set callbacks
        if progress_callback:
            self.processor.on_progress = progress_callback
        if result_callback:
            self.processor.on_document_complete = result_callback

        # Start processor
        self.processor.start()

        # Submit all documents
        submitted = self.processor.submit_batch(documents, md_contents)

        # Wait for completion
        while self.processor.get_progress()['total_pending'] > 0:
            time.sleep(0.5)

        # Stop processor
        self.processor.stop()

        return self.processor.get_results()


def process_streaming(
    model: Model,
    model_name: str,
    task: str,
    documents: List[Dict],
    md_contents: Dict[str, str],
    output_path: Optional[str] = None
) -> List[ProcessingResult]:
    """
    Convenience function for streaming processing.

    :param model: Model instance
    :param model_name: Name of model
    :param task: Task type
    :param documents: List of document metadata
    :param md_contents: Dictionary mapping pmcid to markdown content
    :param output_path: Optional path to save results
    :return: List of processing results
    """
    processor = RealTimeBatchProcessor(model, model_name, task)

    def print_progress(progress: Dict):
        percent = progress.get('progress_percent', 0)
        print(f"\rProgress: {percent:.1f}% ({progress['total_completed']}/{progress['total_submitted']})", end='')

    results = processor.process_batch(
        documents=documents,
        md_contents=md_contents,
        progress_callback=print_progress
    )

    print()  # New line after progress

    if output_path:
        processor.processor.save_results(output_path)
        print(f"Results saved to {output_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Real-time processing of clinical trial documents")
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument("--model", default="gpt35", help="Model to use")
    parser.add_argument("--task", default="binary_outcomes", choices=["outcome_type", "binary_outcomes", "continuous_outcomes"])
    parser.add_argument("--output", required=True, help="Output path for results")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")

    args = parser.parse_args()

    # This would require additional setup to load model and documents
    print("Real-time processing module loaded")
    print(f"Configuration: model={args.model}, task={args.task}, workers={args.workers}")
