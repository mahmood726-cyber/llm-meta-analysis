"""
Web Interface for LLM Meta-Analysis

This module provides a FastAPI-based web interface for interacting with
the LLM meta-analysis system.
"""

import os
import json
import asyncio
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from uvicorn import run

# Import local modules
from run_task import MetaAnalysisTaskRunner, run_end_to_end_task
from evaluate_output import MetaAnalysisTaskEvaluator
from error_analyzer import ErrorAnalyzer
from visualizer import ResultsVisualizer, load_metrics_from_files


# Pydantic models for request/response
class TaskRequest(BaseModel):
    model: str = Field(..., description="Model to use")
    task: str = Field(..., description="Task to run")
    split: Optional[str] = Field(None, description="Dataset split (dev/test)")
    input_path: Optional[str] = Field(None, description="Custom input path")
    pmc_files_path: Optional[str] = Field(None, description="Path to PMC files")
    is_test: bool = Field(False, description="Run with 10 samples for testing")

    class Config:
        json_schema_extra = {
            "example": {
                "model": "gpt35",
                "task": "binary_outcomes",
                "split": "test",
                "is_test": False
            }
        }


class EvaluationRequest(BaseModel):
    task: str = Field(..., description="Task type")
    output_path: str = Field(..., description="Path to model output")
    pmc_files_path: Optional[str] = Field(None, description="Path to PMC files")

    class Config:
        json_schema_extra = {
            "example": {
                "task": "binary_outcomes",
                "output_path": "./outputs/binary_outcomes/gpt35_binary_outcomes_test_output.json"
            }
        }


class ErrorResponse(BaseModel):
    error: str
    detail: str


# Available models
AVAILABLE_MODELS = [
    "gpt35", "gpt4", "mistral7B", "biomistral", "pmc-llama",
    "gemma7B", "olmo7B", "alpaca13B", "claude", "claude-opus",
    "claude-sonnet", "claude-haiku", "gemini", "gemini-pro",
    "gemini-flash", "llama3", "llama3-8b", "llama3-70b"
]

AVAILABLE_TASKS = ["outcome_type", "binary_outcomes", "continuous_outcomes", "end_to_end"]


# Initialize FastAPI app
app = FastAPI(
    title="LLM Meta-Analysis API",
    description="API for automated meta-analysis of clinical trials using Large Language Models",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for running jobs
running_jobs: Dict[str, Dict] = {}


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "LLM Meta-Analysis API",
        "version": "2.0.0",
        "endpoints": {
            "/models": "List available models",
            "/tasks": "List available tasks",
            "/run": "Run a task",
            "/evaluate": "Evaluate model outputs",
            "/error-analysis": "Perform error analysis",
            "/visualize": "Generate visualizations"
        }
    }


@app.get("/models")
async def get_models():
    """Get list of available models"""
    return {
        "models": AVAILABLE_MODELS,
        "proprietary": ["gpt35", "gpt4", "claude", "claude-opus", "claude-sonnet", "claude-haiku", "gemini", "gemini-pro", "gemini-flash"],
        "open_source": ["mistral7B", "biomistral", "pmc-llama", "gemma7B", "olmo7B", "alpaca13B", "llama3", "llama3-8b", "llama3-70b"]
    }


@app.get("/tasks")
async def get_tasks():
    """Get list of available tasks"""
    return {
        "tasks": AVAILABLE_TASKS,
        "descriptions": {
            "outcome_type": "Classify outcome type (binary/continuous)",
            "binary_outcomes": "Extract 2x2 contingency tables",
            "continuous_outcomes": "Extract mean, SD, and group sizes",
            "end_to_end": "Run complete pipeline (all tasks)"
        }
    }


@app.post("/run")
async def run_task(request: TaskRequest, background_tasks: BackgroundTasks):
    """
    Run a meta-analysis task.

    Returns a job ID that can be used to check status.
    """
    # Validate inputs
    if request.model not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model. Choose from: {AVAILABLE_MODELS}")

    if request.task not in AVAILABLE_TASKS:
        raise HTTPException(status_code=400, detail=f"Invalid task. Choose from: {AVAILABLE_TASKS}")

    if request.split is None and request.input_path is None:
        raise HTTPException(status_code=400, detail="Either split or input_path must be specified")

    # Generate job ID
    import uuid
    job_id = str(uuid.uuid4())

    # Create output directory
    output_dir = f"./outputs/{request.task}"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize job status
    running_jobs[job_id] = {
        "status": "queued",
        "model": request.model,
        "task": request.task,
        "created_at": str(asyncio.get_event_loop().time())
    }

    # Add background task
    background_tasks.add_task(
        _run_task_job,
        job_id,
        request.model,
        request.task,
        request.split,
        output_dir,
        request.is_test,
        None,
        request.input_path,
        request.pmc_files_path
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"Task {request.task} with model {request.model} has been queued"
    }


async def _run_task_job(job_id: str, model: str, task: str, split: str, output_path: str,
                        is_test: bool, prompt_name: Optional[str], input_path: Optional[str],
                        pmc_files_path: Optional[str]):
    """Background task to run the meta-analysis task"""
    try:
        running_jobs[job_id]["status"] = "running"

        if task == "end_to_end":
            result = run_end_to_end_task(model, split, input_path, output_path, pmc_files_path, is_test)
        else:
            task_runner = MetaAnalysisTaskRunner(model, task, split, output_path, is_test, prompt_name, input_path, pmc_files_path)
            result = task_runner.run_task()

        running_jobs[job_id]["status"] = "completed"
        running_jobs[job_id]["result"] = {
            "output_files": result,
            "completed_at": str(asyncio.get_event_loop().time())
        }
    except Exception as e:
        running_jobs[job_id]["status"] = "failed"
        running_jobs[job_id]["error"] = str(e)


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a running job"""
    if job_id not in running_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return running_jobs[job_id]


@app.post("/evaluate")
async def evaluate_output(request: EvaluationRequest):
    """
    Evaluate model outputs and calculate metrics.
    """
    # Validate inputs
    if request.task not in ["outcome_type", "binary_outcomes", "continuous_outcomes"]:
        raise HTTPException(status_code=400, detail="Invalid task")

    if not os.path.exists(request.output_path):
        raise HTTPException(status_code=404, detail=f"Output file not found: {request.output_path}")

    # Create metrics directory
    metrics_dir = "./metrics/evaluated"
    os.makedirs(metrics_dir, exist_ok=True)

    # Run evaluation
    try:
        evaluator = MetaAnalysisTaskEvaluator(
            task=request.task,
            output_path=request.output_path,
            metrics_path=metrics_dir,
            pmc_files_path=request.pmc_files_path
        )
        evaluator.run_evaluation()

        # Load and return metrics
        output_name = Path(request.output_path).stem
        metrics_file = os.path.join(metrics_dir, f"{output_name}_metrics.json")

        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            return {
                "status": "success",
                "metrics": metrics,
                "metrics_file": metrics_file
            }
        else:
            raise HTTPException(status_code=500, detail="Metrics calculation failed")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/error-analysis")
async def error_analysis(
    task: str = Form(...),
    output_path: str = Form(...),
    model_name: str = Form(...),
    pmc_files_path: Optional[str] = Form(None)
):
    """
    Perform comprehensive error analysis on model outputs.
    """
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail=f"Output file not found: {output_path}")

    # Create output directory
    error_dir = "./error_analysis/evaluated"
    os.makedirs(error_dir, exist_ok=True)

    try:
        analyzer = ErrorAnalyzer(task, output_path, pmc_files_path)
        report = analyzer.run_analysis(model_name)

        # Save report
        output_file = os.path.join(error_dir, f"{model_name}_{task}_error_analysis.json")
        analyzer.save_report(report, output_file)

        return {
            "status": "success",
            "report": report,
            "report_file": output_file
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/visualize")
async def create_visualizations(
    metrics_path: str = Form(...),
    model_name: str = Form(...),
    task: str = Form(...)
):
    """
    Generate visualizations for metrics.
    """
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail=f"Metrics file not found: {metrics_path}")

    # Create output directory
    viz_dir = "./visualizations/generated"
    os.makedirs(viz_dir, exist_ok=True)

    try:
        # Load metrics
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)

        visualizer = ResultsVisualizer()

        # Generate visualizations
        output_files = {}

        # Field-wise accuracy
        fieldwise_path = os.path.join(viz_dir, f"{model_name}_{task}_fieldwise.html")
        visualizer.plot_fieldwise_performance(metrics_data, output_path=fieldwise_path)
        output_files["fieldwise"] = fieldwise_path

        # Partial match progression
        partial_path = os.path.join(viz_dir, f"{model_name}_{task}_partial_match.html")
        visualizer.plot_partial_match_progression(metrics_data, output_path=partial_path)
        output_files["partial_match"] = partial_path

        # Dashboard
        dashboard_path = os.path.join(viz_dir, f"{model_name}_{task}_dashboard.html")
        visualizer.create_dashboard(metrics_data, model_name, dashboard_path)
        output_files["dashboard"] = dashboard_path

        return {
            "status": "success",
            "visualizations": output_files
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard")
async def get_dashboard():
    """
    Serve the main dashboard HTML page.
    """
    dashboard_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>LLM Meta-Analysis Dashboard</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                padding: 40px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            }
            h1 {
                color: #333;
                margin-bottom: 10px;
                font-size: 2.5em;
            }
            .subtitle { color: #666; margin-bottom: 40px; }
            .form-group { margin-bottom: 25px; }
            label {
                display: block;
                margin-bottom: 8px;
                font-weight: 600;
                color: #444;
            }
            select, input[type="text"] {
                width: 100%;
                padding: 12px 15px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-size: 14px;
                transition: border-color 0.3s;
            }
            select:focus, input:focus {
                outline: none;
                border-color: #667eea;
            }
            .button-group {
                display: flex;
                gap: 15px;
                margin-top: 30px;
            }
            button {
                flex: 1;
                padding: 15px 30px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            button.primary {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            button.secondary {
                background: #f0f0f0;
                color: #333;
            }
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }
            .status {
                margin-top: 30px;
                padding: 20px;
                border-radius: 8px;
                display: none;
            }
            .status.success { background: #d4edda; color: #155724; }
            .status.error { background: #f8d7da; color: #721c24; }
            .status.info { background: #d1ecf1; color: #0c5460; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>LLM Meta-Analysis Dashboard</h1>
            <p class="subtitle">Automated Clinical Trial Data Extraction</p>

            <form id="runForm">
                <div class="form-group">
                    <label for="model">Model</label>
                    <select id="model" name="model" required>
                        <option value="">Select a model...</option>
                        <optgroup label="Proprietary Models">
                            <option value="gpt35">GPT-3.5 Turbo</option>
                            <option value="gpt4">GPT-4</option>
                            <option value="claude">Claude 3.5 Sonnet</option>
                            <option value="claude-opus">Claude 3 Opus</option>
                            <option value="claude-haiku">Claude 3 Haiku</option>
                            <option value="gemini-pro">Gemini Pro</option>
                        </optgroup>
                        <optgroup label="Open Source Models">
                            <option value="mistral7B">Mistral 7B</option>
                            <option value="biomistral">BioMistral 7B</option>
                            <option value="llama3">Llama 3 8B</option>
                            <option value="gemma7B">Gemma 7B</option>
                        </optgroup>
                    </select>
                </div>

                <div class="form-group">
                    <label for="task">Task</label>
                    <select id="task" name="task" required>
                        <option value="">Select a task...</option>
                        <option value="outcome_type">Outcome Type Classification</option>
                        <option value="binary_outcomes">Binary Outcomes Extraction</option>
                        <option value="continuous_outcomes">Continuous Outcomes Extraction</option>
                        <option value="end_to_end">End-to-End Pipeline</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="split">Dataset Split</label>
                    <select id="split" name="split">
                        <option value="test">Test Set</option>
                        <option value="dev">Development Set</option>
                    </select>
                </div>

                <div class="button-group">
                    <button type="submit" class="primary">Run Task</button>
                    <button type="button" class="secondary" onclick="loadModels()">Refresh Models</button>
                </div>
            </form>

            <div id="status" class="status"></div>
        </div>

        <script>
            const API_BASE = window.location.origin;

            document.getElementById('runForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const formData = new FormData(e.target);
                const data = Object.fromEntries(formData);

                showStatus('info', 'Starting task...');

                try {
                    const response = await fetch(API_BASE + '/run', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(data)
                    });

                    const result = await response.json();

                    if (response.ok) {
                        showStatus('success', `Task started! Job ID: ${result.job_id}`);
                        pollJobStatus(result.job_id);
                    } else {
                        showStatus('error', result.detail || 'Error starting task');
                    }
                } catch (error) {
                    showStatus('error', 'Network error: ' + error.message);
                }
            });

            async function pollJobStatus(jobId) {
                const interval = setInterval(async () => {
                    try {
                        const response = await fetch(API_BASE + '/jobs/' + jobId);
                        const job = await response.json();

                        if (job.status === 'completed') {
                            clearInterval(interval);
                            showStatus('success', 'Task completed! Outputs: ' + JSON.stringify(job.result));
                        } else if (job.status === 'failed') {
                            clearInterval(interval);
                            showStatus('error', 'Task failed: ' + job.error);
                        } else {
                            showStatus('info', 'Task running...');
                        }
                    } catch (error) {
                        clearInterval(interval);
                        showStatus('error', 'Error checking status');
                    }
                }, 2000);
            }

            function showStatus(type, message) {
                const status = document.getElementById('status');
                status.className = 'status ' + type;
                status.textContent = message;
                status.style.display = 'block';
            }

            async function loadModels() {
                const response = await fetch(API_BASE + '/models');
                const data = await response.json();
                console.log('Available models:', data.models);
            }
        </script>
    </body>
    </html>
    """

    return HTMLResponse(content=dashboard_html)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "jobs_running": len(running_jobs)}


def HTMLResponse(content: str):
    """Helper to return HTML content"""
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=content)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the LLM Meta-Analysis web server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    print(f"Starting LLM Meta-Analysis API server on {args.host}:{args.port}")
    print(f"Dashboard available at http://{args.host}:{args.port}/dashboard")

    run(app, host=args.host, port=args.port, reload=args.reload)
