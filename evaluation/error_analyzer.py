"""
Error Analysis Framework for LLM Meta-Analysis

This module provides comprehensive error analysis capabilities for evaluating
LLM performance on clinical trial data extraction tasks.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ErrorRecord:
    """Record of a single error instance"""
    pmcid: str
    intervention: str
    comparator: str
    outcome: str
    task: str
    error_type: str
    expected_value: Any
    predicted_value: Any
    context: str
    is_chunked: bool = False


@dataclass
class ErrorAnalysisReport:
    """Complete error analysis report"""
    timestamp: str
    model_name: str
    task: str
    total_instances: int
    correct_instances: int
    error_instances: int
    accuracy: float
    error_breakdown: Dict[str, int]
    fieldwise_errors: Dict[str, Dict[str, int]]
    common_error_patterns: List[Dict[str, Any]]
    chunking_impact: Dict[str, Any]
    error_records: List[Dict[str, Any]]


class ErrorAnalyzer:
    """
    Comprehensive error analyzer for meta-analysis tasks.

    Provides detailed analysis of:
    - Type of errors (extraction, parsing, calculation)
    - Field-wise error distribution
    - Common error patterns
    - Impact of input chunking on errors
    - Unknown value handling
    """

    ERROR_TYPES = {
        "extraction_error": "Failed to extract correct value from text",
        "parsing_error": "Failed to parse model output correctly",
        "calculation_error": "Error in derived calculation (e.g., log odds ratio)",
        "unknown_prediction": "Model predicted 'unknown' when value was available",
        "unknown_available": "True unknown value in reference",
        "type_mismatch": "Predicted value type doesn't match expected type",
        "out_of_range": "Predicted value outside valid range",
        "hallucination": "Predicted value not found in source text",
    }

    def __init__(self, task: str, output_path: str, pmc_files_path: Optional[str] = None):
        """
        Initialize the error analyzer.

        :param task: Task type (outcome_type, binary_outcomes, continuous_outcomes)
        :param output_path: Path to model output JSON file
        :param pmc_files_path: Path to PMC markdown files (for context analysis)
        """
        self.task = task
        self.output_path = output_path
        self.pmc_files_path = pmc_files_path
        self.data = self._load_data()
        self.error_records: List[ErrorRecord] = []

    def _load_data(self) -> List[Dict]:
        """Load the output data from JSON file"""
        with open(self.output_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _get_md_content(self, pmcid: str) -> str:
        """Get markdown content for a given PMC ID"""
        if self.pmc_files_path is None:
            return ""
        md_path = os.path.join(self.pmc_files_path, f"PMC{pmcid}.md")
        if os.path.exists(md_path):
            with open(md_path, 'r', encoding='utf-8') as f:
                return f.read()[:500]  # First 500 chars for context
        return ""

    def _classify_error(self, expected: Any, predicted: Any, field: str) -> str:
        """
        Classify the type of error based on expected vs predicted values.

        :param expected: The ground truth value
        :param predicted: The model's predicted value
        :param field: The field being evaluated
        :return: Error type string
        """
        # Check for unknown prediction
        if predicted == "x" or predicted == "unknown":
            if expected not in ["x", "unknown", None, ""]:
                return "unknown_prediction"
            return "unknown_available"

        # Check for type mismatch
        expected_type = type(expected)
        predicted_type = type(predicted)
        if expected_type != predicted_type and not (
            isinstance(expected, str) and isinstance(predicted, str)
        ):
            return "type_mismatch"

        # For numeric values, check ranges
        if isinstance(expected, (int, float)) and isinstance(predicted, (int, float)):
            if field in ["intervention_events", "comparator_events", "intervention_group_size", "comparator_group_size"]:
                if predicted < 0 or predicted > 10000:  # Reasonable bounds
                    return "out_of_range"

        # Default to extraction error
        return "extraction_error"

    def _check_value_in_source(self, value: Any, pmcid: str) -> bool:
        """
        Check if a predicted value exists in the source text.
        Useful for detecting hallucinations.

        :param value: The predicted value
        :param pmcid: The PMC ID for looking up source text
        :return: True if value is found in source
        """
        if self.pmc_files_path is None:
            return True  # Can't check without source files

        content = self._get_md_content(pmcid)
        if not content:
            return True

        value_str = str(value)
        # Check if the numeric value appears in the content
        return value_str in content

    def analyze_outcome_type_errors(self) -> None:
        """Analyze errors for outcome type classification task"""
        for example in self.data:
            expected = example.get("outcome_type")
            predicted = example.get("outcome_type_output")

            if expected != predicted:
                error_type = self._classify_error(expected, predicted, "outcome_type")
                context = self._get_md_content(example.get("pmcid", ""))

                self.error_records.append(ErrorRecord(
                    pmcid=example.get("pmcid", ""),
                    intervention=example.get("intervention", ""),
                    comparator=example.get("comparator", ""),
                    outcome=example.get("outcome", ""),
                    task=self.task,
                    error_type=error_type,
                    expected_value=expected,
                    predicted_value=predicted,
                    context=context,
                    is_chunked=example.get("is_chunked", False)
                ))

    def analyze_binary_outcome_errors(self) -> None:
        """Analyze errors for binary outcomes extraction task"""
        fields = ["intervention_events", "intervention_group_size",
                  "comparator_events", "comparator_group_size"]

        for example in self.data:
            is_chunked = example.get("is_chunked", False)
            context = self._get_md_content(example.get("pmcid", ""))

            for field in fields:
                expected = example.get(field)
                predicted = example.get(f"{field}_output")

                if expected != predicted:
                    error_type = self._classify_error(expected, predicted, field)

                    # Check for hallucination
                    if error_type == "extraction_error":
                        if not self._check_value_in_source(predicted, example.get("pmcid", "")):
                            error_type = "hallucination"

                    self.error_records.append(ErrorRecord(
                        pmcid=example.get("pmcid", ""),
                        intervention=example.get("intervention", ""),
                        comparator=example.get("comparator", ""),
                        outcome=example.get("outcome", ""),
                        task=self.task,
                        error_type=error_type,
                        expected_value=expected,
                        predicted_value=predicted,
                        context=context,
                        is_chunked=is_chunked
                    ))

    def analyze_continuous_outcome_errors(self) -> None:
        """Analyze errors for continuous outcomes extraction task"""
        fields = ["intervention_mean", "intervention_standard_deviation", "intervention_group_size",
                  "comparator_mean", "comparator_standard_deviation", "comparator_group_size"]

        for example in self.data:
            is_chunked = example.get("is_chunked", False)
            context = self._get_md_content(example.get("pmcid", ""))

            for field in fields:
                expected = example.get(field)
                predicted = example.get(f"{field}_output")

                if expected != predicted:
                    error_type = self._classify_error(expected, predicted, field)

                    # Check for hallucination
                    if error_type == "extraction_error":
                        if not self._check_value_in_source(predicted, example.get("pmcid", "")):
                            error_type = "hallucination"

                    self.error_records.append(ErrorRecord(
                        pmcid=example.get("pmcid", ""),
                        intervention=example.get("intervention", ""),
                        comparator=example.get("comparator", ""),
                        outcome=example.get("outcome", ""),
                        task=self.task,
                        error_type=error_type,
                        expected_value=expected,
                        predicted_value=predicted,
                        context=context,
                        is_chunked=is_chunked
                    ))

    def analyze_point_estimate_errors(self) -> None:
        """Analyze errors in derived point estimates (LOR, SMD)"""
        if self.task == "binary_outcomes":
            metric_field = "log_odds_ratio"
        elif self.task == "continuous_outcomes":
            metric_field = "standardized_mean_difference"
        else:
            return

        for example in self.data:
            expected = example.get(metric_field)
            predicted = example.get(f"{metric_field}_output")

            # Skip if either is None
            if expected is None or predicted is None:
                continue

            # Calculate difference
            try:
                expected_val = float(expected) if expected != "x" else None
                predicted_val = float(predicted) if predicted != "x" else None

                if expected_val is not None and predicted_val is not None:
                    diff = abs(expected_val - predicted_val)
                    # Consider it an error if difference is significant (>0.5)
                    if diff > 0.5:
                        context = self._get_md_content(example.get("pmcid", ""))

                        self.error_records.append(ErrorRecord(
                            pmcid=example.get("pmcid", ""),
                            intervention=example.get("intervention", ""),
                            comparator=example.get("comparator", ""),
                            outcome=example.get("outcome", ""),
                            task=self.task,
                            error_type="calculation_error",
                            expected_value=f"{expected_val:.4f}",
                            predicted_value=f"{predicted_val:.4f}",
                            context=f"Delta: {diff:.4f}",
                            is_chunked=example.get("is_chunked", False)
                        ))
            except (ValueError, TypeError):
                pass

    def run_analysis(self, model_name: str = "unknown") -> ErrorAnalysisReport:
        """
        Run complete error analysis.

        :param model_name: Name of the model being analyzed
        :return: ErrorAnalysisReport with all findings
        """
        # Run appropriate analysis based on task
        if self.task == "outcome_type":
            self.analyze_outcome_type_errors()
        elif self.task == "binary_outcomes":
            self.analyze_binary_outcome_errors()
            self.analyze_point_estimate_errors()
        elif self.task == "continuous_outcomes":
            self.analyze_continuous_outcome_errors()
            self.analyze_point_estimate_errors()

        # Compile statistics
        total_instances = len(self.data)
        error_instances = len(self.error_records)

        # Calculate accuracy based on exact matches
        if self.task == "outcome_type":
            correct = sum(1 for e in self.data if e.get("outcome_type") == e.get("outcome_type_output"))
        elif self.task == "binary_outcomes":
            correct = sum(1 for e in self.data if all(
                e.get(f) == e.get(f"{f}_output") for f in
                ["intervention_events", "intervention_group_size", "comparator_events", "comparator_group_size"]
            ))
        else:  # continuous_outcomes
            correct = sum(1 for e in self.data if all(
                e.get(f) == e.get(f"{f}_output") for f in
                ["intervention_mean", "intervention_standard_deviation", "intervention_group_size",
                 "comparator_mean", "comparator_standard_deviation", "comparator_group_size"]
            ))

        # Error breakdown by type
        error_breakdown = Counter(e.error_type for e in self.error_records)

        # Field-wise error breakdown
        if self.task in ["binary_outcomes", "continuous_outcomes"]:
            fieldwise_errors = defaultdict(lambda: defaultdict(int))
            for record in self.error_records:
                for field in ["intervention", "comparator"]:
                    if field in record.context.lower() or any(f in str(record.expected_value) for f in [field]):
                        fieldwise_errors[field][record.error_type] += 1
            fieldwise_errors = dict(fieldwise_errors)
        else:
            fieldwise_errors = {}

        # Common error patterns (top 10)
        pattern_counts = defaultdict(int)
        for record in self.error_records:
            pattern = f"{record.error_type}|{record.error_type}"
            pattern_counts[pattern] += 1

        common_error_patterns = [
            {"pattern": k, "count": v}
            for k, v in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]

        # Chunking impact
        chunked_errors = sum(1 for e in self.error_records if e.is_chunked)
        non_chunked_errors = sum(1 for e in self.error_records if not e.is_chunked)
        total_chunked = sum(1 for e in self.data if e.get("is_chunked", False))
        total_non_chunked = len(self.data) - total_chunked

        chunking_impact = {
            "total_chunked_instances": total_chunked,
            "chunked_with_errors": chunked_errors,
            "chunked_error_rate": chunked_errors / total_chunked if total_chunked > 0 else 0,
            "total_non_chunked_instances": total_non_chunked,
            "non_chunked_with_errors": non_chunked_errors,
            "non_chunked_error_rate": non_chunked_errors / total_non_chunked if total_non_chunked > 0 else 0,
        }

        return ErrorAnalysisReport(
            timestamp=datetime.now().isoformat(),
            model_name=model_name,
            task=self.task,
            total_instances=total_instances,
            correct_instances=correct,
            error_instances=error_instances,
            accuracy=correct / total_instances if total_instances > 0 else 0,
            error_breakdown=dict(error_breakdown),
            fieldwise_errors=fieldwise_errors,
            common_error_patterns=common_error_patterns,
            chunking_impact=chunking_impact,
            error_records=[asdict(r) for r in self.error_records[:100]]  # Limit to 100 records
        )

    def save_report(self, report: ErrorAnalysisReport, output_path: str) -> None:
        """
        Save error analysis report to JSON file.

        :param report: The error analysis report
        :param output_path: Path to save the report
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(report), f, indent=2, default=str)

    def print_summary(self, report: ErrorAnalysisReport) -> None:
        """Print a summary of the error analysis report"""
        print(f"\n{'='*60}")
        print(f"Error Analysis Report for {report.model_name}")
        print(f"Task: {report.task}")
        print(f"{'='*60}")
        print(f"Total Instances: {report.total_instances}")
        print(f"Correct: {report.correct_instances}")
        print(f"Errors: {report.error_instances}")
        print(f"Accuracy: {report.accuracy:.2%}")
        print(f"\nError Breakdown:")
        for error_type, count in sorted(report.error_breakdown.items(), key=lambda x: x[1], reverse=True):
            description = self.ERROR_TYPES.get(error_type, error_type)
            print(f"  - {description}: {count}")
        print(f"\nChunking Impact:")
        print(f"  - Chunked instances: {report.chunking_impact['total_chunked_instances']} "
              f"(error rate: {report.chunking_impact['chunked_error_rate']:.2%})")
        print(f"  - Non-chunked instances: {report.chunking_impact['total_non_chunked_instances']} "
              f"(error rate: {report.chunking_impact['non_chunked_error_rate']:.2%})")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze errors from model outputs")
    parser.add_argument("--task", required=True, choices=["outcome_type", "binary_outcomes", "continuous_outcomes"])
    parser.add_argument("--output_path", required=True, help="Path to model output JSON file")
    parser.add_argument("--metrics_path", default="./error_analysis", help="Path to save error analysis")
    parser.add_argument("--pmc_files_path", default=None, help="Path to PMC markdown files")
    parser.add_argument("--model_name", default="unknown", help="Name of the model being analyzed")

    args = parser.parse_args()

    analyzer = ErrorAnalyzer(args.task, args.output_path, args.pmc_files_path)
    report = analyzer.run_analysis(args.model_name)

    # Save report
    output_file = os.path.join(args.metrics_path, f"{args.model_name}_{args.task}_error_analysis.json")
    analyzer.save_report(report, output_file)

    # Print summary
    analyzer.print_summary(report)
