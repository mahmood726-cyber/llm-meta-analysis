"""
Visualization Module for LLM Meta-Analysis

This module provides comprehensive visualization capabilities for analyzing
and presenting results from LLM-based meta-analysis tasks.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


@dataclass
class VisualizationConfig:
    """Configuration for visualizations"""
    colors: List[str] = None
    figsize: Tuple[int, int] = (12, 6)
    dpi: int = 100
    style: str = "whitegrid"
    color_palette: str = "husl"

    def __post_init__(self):
        if self.colors is None:
            self.colors = sns.color_palette(self.color_palette, 10).as_hex()


class ResultsVisualizer:
    """
    Comprehensive visualization for meta-analysis results.

    Supports visualization of:
    - Model comparison metrics
    - Error distributions
    - Point estimate accuracy
    - Field-wise performance
    - Chunking impact
    - Task-specific visualizations
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize the visualizer.

        :param config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        sns.set_style(self.config.style)

    def plot_model_comparison(self, metrics_data: Dict[str, Dict], output_path: Optional[str] = None) -> go.Figure:
        """
        Create a comparison plot of multiple models.

        :param metrics_data: Dictionary mapping model names to their metrics
        :param output_path: Optional path to save the figure
        :return: Plotly figure object
        """
        models = list(metrics_data.keys())
        metrics = list(metrics_data[models[0]].keys())

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Accuracy Comparison", "F1 Scores", "Error Rates", "Coverage"),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )

        # Accuracy comparison
        if "exact_match_accuracy" in metrics_data[models[0]]:
            accuracies = [metrics_data[m].get("exact_match_accuracy", {}).get("total", 0) for m in models]
            fig.add_trace(
                go.Bar(x=models, y=accuracies, name="Accuracy", marker_color=self.config.colors[0]),
                row=1, col=1
            )

        # F1 scores
        if "outcome_type_f_score" in metrics_data[models[0]]:
            f1_scores = [metrics_data[m].get("outcome_type_f_score", {}).get("outcome_type", {}).get("f1_score_binary", 0) for m in models]
            fig.add_trace(
                go.Bar(x=models, y=f1_scores, name="F1 Score", marker_color=self.config.colors[1]),
                row=1, col=2
            )

        # Error rates
        error_rates = []
        for m in models:
            total_errors = metrics_data[m].get("number_of_model_unknowns", {}).get("total", 0)
            error_rates.append(total_errors)
        fig.add_trace(
            go.Bar(x=models, y=error_rates, name="Unknown Count", marker_color=self.config.colors[2]),
            row=2, col=1
        )

        # Coverage (computable instances)
        coverages = []
        for m in models:
            cov = metrics_data[m].get("percentage_of_computable_instances", 0)
            coverages.append(cov * 100 if cov < 1 else cov)
        fig.add_trace(
            go.Bar(x=models, y=coverages, name="Coverage %", marker_color=self.config.colors[3]),
            row=2, col=2
        )

        fig.update_layout(
            title_text="Model Performance Comparison",
            showlegend=False,
            height=600
        )

        if output_path:
            fig.write_html(output_path)

        return fig

    def plot_error_distribution(self, error_data: List[Dict], output_path: Optional[str] = None) -> go.Figure:
        """
        Visualize error distribution by type.

        :param error_data: List of error records
        :param output_path: Optional path to save the figure
        :return: Plotly figure object
        """
        error_types = {}
        for record in error_data:
            error_type = record.get("error_type", "unknown")
            error_types[error_type] = error_types.get(error_type, 0) + 1

        # Sort by count
        sorted_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)
        types, counts = zip(*sorted_errors) if sorted_errors else ([], [])

        fig = go.Figure(data=[
            go.Bar(x=list(types), y=list(counts), marker_color=self.config.colors[0])
        ])

        fig.update_layout(
            title="Error Distribution by Type",
            xaxis_title="Error Type",
            yaxis_title="Count",
            showlegend=False
        )

        if output_path:
            fig.write_html(output_path)

        return fig

    def plot_point_estimate_accuracy(self, data: List[Dict], task: str, output_path: Optional[str] = None) -> plt.Figure:
        """
        Plot actual vs predicted point estimates (LOR or SMD).

        :param data: Data with actual and predicted values
        :param task: Task type (binary_outcomes or continuous_outcomes)
        :param output_path: Optional path to save the figure
        :return: Matplotlib figure
        """
        if task == "binary_outcomes":
            actual_field = "log_odds_ratio"
            pred_field = "log_odds_ratio_output"
            title = "Log Odds Ratio: Actual vs Predicted"
        else:
            actual_field = "standardized_mean_difference"
            pred_field = "standardized_mean_difference_output"
            title = "Standardized Mean Difference: Actual vs Predicted"

        # Extract valid pairs
        actual_vals = []
        pred_vals = []

        for item in data:
            actual = item.get(actual_field)
            pred = item.get(pred_field)

            if actual is not None and pred is not None:
                try:
                    actual_vals.append(float(actual))
                    pred_vals.append(float(pred))
                except (ValueError, TypeError):
                    continue

        fig, ax = plt.subplots(figsize=self.config.figsize)

        # Scatter plot
        ax.scatter(actual_vals, pred_vals, alpha=0.6, s=50, color=self.config.colors[0])

        # Perfect prediction line
        min_val = min(min(actual_vals), min(pred_vals))
        max_val = max(max(actual_vals), max(pred_vals))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        # Calculate correlation and MAE
        correlation = np.corrcoef(actual_vals, pred_vals)[0, 1]
        mae = np.mean([abs(a - p) for a, p in zip(actual_vals, pred_vals)])

        ax.set_xlabel('Actual Value', fontsize=12)
        ax.set_ylabel('Predicted Value', fontsize=12)
        ax.set_title(f'{title}\nCorrelation: {correlation:.3f}, MAE: {mae:.3f}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')

        return fig

    def plot_fieldwise_performance(self, metrics_data: Dict, output_path: Optional[str] = None) -> go.Figure:
        """
        Plot performance breakdown by field.

        :param metrics_data: Metrics data with fieldwise breakdown
        :param output_path: Optional path to save the figure
        :return: Plotly figure object
        """
        exact_match = metrics_data.get("exact_match_accuracy", {})

        fields = [k for k in exact_match.keys() if k != "total"]
        accuracies = [exact_match.get(f, 0) for f in fields]

        fig = go.Figure(data=[
            go.Bar(x=fields, y=accuracies, marker_color=self.config.colors[:len(fields)])
        ])

        fig.update_layout(
            title="Field-wise Accuracy",
            xaxis_title="Field",
            yaxis_title="Accuracy",
            yaxis=dict(range=[0, 1]),
            showlegend=False
        )

        if output_path:
            fig.write_html(output_path)

        return fig

    def plot_chunking_impact(self, metrics_data: Dict, output_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize the impact of input chunking on performance.

        :param metrics_data: Metrics data including chunking information
        :param output_path: Optional path to save the figure
        :return: Matplotlib figure
        """
        chunked_count = metrics_data.get("num_of_chunked_instances", 0)
        total_count = metrics_data.get("total_instances", len(metrics_data.get("data", [])))

        non_chunked = total_count - chunked_count

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.config.figsize)

        # Pie chart of chunked vs non-chunked
        sizes = [chunked_count, non_chunked]
        labels = [f'Chunked\n({chunked_count})', f'Not Chunked\n({non_chunked})']
        colors = [self.config.colors[0], self.config.colors[1]]

        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Input Chunking Distribution')

        # Bar chart comparison (if available)
        exact_match = metrics_data.get("exact_match_accuracy", {})
        total_acc = exact_match.get("total", 0)

        categories = ['Overall']
        accuracies = [total_acc]

        ax2.bar(categories, accuracies, color=[self.config.colors[2]])
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Overall Accuracy')
        ax2.set_ylim([0, 1])

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')

        return fig

    def plot_confusion_matrix(self, data: List[Dict], output_path: Optional[str] = None) -> plt.Figure:
        """
        Plot confusion matrix for outcome type classification.

        :param data: Data with actual and predicted outcome types
        :param output_path: Optional path to save the figure
        :return: Matplotlib figure
        """
        # Extract actual and predicted values
        actual = []
        predicted = []

        for item in data:
            a = item.get("outcome_type")
            p = item.get("outcome_type_output")
            if a and p:
                actual.append(a)
                predicted.append(p)

        # Get unique labels
        labels = sorted(set(actual + predicted))

        # Create confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(actual, predicted, labels=labels)

        fig, ax = plt.subplots(figsize=self.config.figsize)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=ax)

        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title('Confusion Matrix - Outcome Type Classification', fontsize=14)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')

        return fig

    def plot_partial_match_progression(self, metrics_data: Dict, output_path: Optional[str] = None) -> go.Figure:
        """
        Plot partial match accuracy progression.

        :param metrics_data: Metrics data with partial match information
        :param output_path: Optional path to save the figure
        :return: Plotly figure object
        """
        partial_match = metrics_data.get("partial_match_accuracy", {})

        # Extract partial match data
        matches = []
        accuracies = []

        for key, value in partial_match.items():
            if "partial_match_accuracy" in key:
                num_match = key.split("_")[-1]
                matches.append(f"{num_match}+ Correct")
                accuracies.append(value)

        if matches:
            matches.append("All Correct")
            total_acc = metrics_data.get("exact_match_accuracy", {}).get("total", 0)
            accuracies.append(total_acc)

        fig = go.Figure(data=[
            go.Scatter(x=matches, y=accuracies, mode='lines+markers',
                      line=dict(color=self.config.colors[0], width=3),
                      marker=dict(size=10))
        ])

        fig.update_layout(
            title="Partial Match Accuracy Progression",
            xaxis_title="Number of Correct Fields",
            yaxis_title="Accuracy",
            yaxis=dict(range=[0, 1])
        )

        if output_path:
            fig.write_html(output_path)

        return fig

    def create_dashboard(self, metrics_data: Dict, model_name: str, output_path: str) -> None:
        """
        Create a comprehensive HTML dashboard with all visualizations.

        :param metrics_data: Complete metrics data
        :param model_name: Name of the model
        :param output_path: Path to save the dashboard
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LLM Meta-Analysis Dashboard - {model_name}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1400px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
                h2 {{ color: #555; margin-top: 30px; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
                .metric-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }}
                .metric-value {{ font-size: 2em; font-weight: bold; }}
                .metric-label {{ font-size: 0.9em; opacity: 0.9; }}
                .chart {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>LLM Meta-Analysis Dashboard</h1>
                <h2>Model: {model_name}</h2>

                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{metrics_data.get('exact_match_accuracy', {}).get('total', 0):.2%}</div>
                        <div class="metric-label">Overall Accuracy</div>
                    </div>
                    <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                        <div class="metric-value">{metrics_data.get('percentage_of_computable_instances', 0):.2%}</div>
                        <div class="metric-label">Coverage</div>
                    </div>
                    <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                        <div class="metric-value">{metrics_data.get('number_of_model_unknowns', {}).get('total', 0)}</div>
                        <div class="metric-label">Unknown Predictions</div>
                    </div>
                </div>

                <div class="chart" id="fieldwise-chart"></div>
                <div class="chart" id="partial-match-chart"></div>
            </div>

            <script>
                // Field-wise accuracy chart
                const fieldwiseData = {json.dumps(metrics_data.get('exact_match_accuracy', {}))};
                const fieldwiseTrace = {{
                    x: Object.keys(fieldwiseData).filter(k => k !== 'total'),
                    y: Object.values(fieldwiseData).filter((_, i) => Object.keys(fieldwiseData)[i] !== 'total'),
                    type: 'bar',
                    marker: {{ color: '#667eea' }}
                }};
                Plotly.newPlot('fieldwise-chart', [fieldwiseTrace], {{
                    title: 'Field-wise Accuracy',
                    xaxis: {{ title: 'Field' }},
                    yaxis: {{ title: 'Accuracy', range: [0, 1] }}
                }});

                // Partial match progression
                const partialMatchData = {json.dumps(metrics_data.get('partial_match_accuracy', {}))};
                const partialTrace = {{
                    x: Object.keys(partialMatchData).map(k => k.replace('partial_match_accuracy_', '') + '+'),
                    y: Object.values(partialMatchData),
                    type: 'scatter',
                    mode: 'lines+markers',
                    line: {{ color: '#f5576c', width: 3 }}
                }};
                Plotly.newPlot('partial-match-chart', [partialTrace], {{
                    title: 'Partial Match Accuracy Progression',
                    xaxis: {{ title: 'Number of Correct Fields' }},
                    yaxis: {{ title: 'Accuracy', range: [0, 1] }}
                }});
            </script>
        </body>
        </html>
        """

        with open(output_path, 'w') as f:
            f.write(html_content)


def load_metrics_from_files(metrics_files: List[str]) -> Dict[str, Dict]:
    """
    Load metrics from multiple JSON files for comparison.

    :param metrics_files: List of paths to metrics JSON files
    :return: Dictionary mapping model names to their metrics
    """
    metrics_data = {}

    for file_path in metrics_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            # Extract model name from filename
            model_name = Path(file_path).stem.replace('_metrics', '').replace('_output', '')
            metrics_data[model_name] = data

    return metrics_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize meta-analysis results")
    parser.add_argument("--metrics_path", required=True, help="Path to metrics JSON file or directory")
    parser.add_argument("--output_dir", default="./visualizations", help="Directory to save visualizations")
    parser.add_argument("--task", required=True, choices=["outcome_type", "binary_outcomes", "continuous_outcomes"])

    args = parser.parse_args()

    visualizer = ResultsVisualizer()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load metrics
    if os.path.isfile(args.metrics_path):
        with open(args.metrics_path, 'r') as f:
            metrics_data = json.load(f)
        model_name = Path(args.metrics_path).stem.replace('_metrics', '')
    else:
        # Load all metrics from directory
        metrics_files = list(Path(args.metrics_path).glob("*.json"))
        metrics_data = load_metrics_from_files([str(f) for f in metrics_files])
        model_name = "comparison"

    # Create visualizations
    if isinstance(metrics_data, dict) and "exact_match_accuracy" in metrics_data:
        # Single model visualization
        visualizer.plot_fieldwise_performance(
            metrics_data,
            output_path=os.path.join(args.output_dir, f"{model_name}_fieldwise.html")
        )

        visualizer.plot_partial_match_progression(
            metrics_data,
            output_path=os.path.join(args.output_dir, f"{model_name}_partial_match.html")
        )

        # Create dashboard
        visualizer.create_dashboard(
            metrics_data,
            model_name,
            output_path=os.path.join(args.output_dir, f"{model_name}_dashboard.html")
        )
    else:
        # Multi-model comparison
        visualizer.plot_model_comparison(
            metrics_data,
            output_path=os.path.join(args.output_dir, "model_comparison.html")
        )

    print(f"Visualizations saved to {args.output_dir}")
