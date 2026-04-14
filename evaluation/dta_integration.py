"""
DTA PRO Integration Module

This module provides integration between the LLM meta-analysis extraction
system and the DTA PRO (Diagnostic Test Accuracy) system.

DTA PRO is a comprehensive tool for diagnostic test accuracy meta-analysis.
This integration allows extracted clinical trial data to be used within DTA PRO.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class DTAProStudy:
    """Study data formatted for DTA PRO"""
    study_id: str
    author: str
    year: int
    condition: str
    index_test: str
    reference_standard: str
    target_condition: str

    # 2x2 table for diagnostic accuracy
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int

    # Additional information
    total_n: int
    prevalence: Optional[float] = None
    design: Optional[str] = None  # e.g., "case-control", "cohort"


@dataclass
class DTAProDataset:
    """Complete dataset for DTA PRO analysis"""
    name: str
    description: str
    studies: List[DTAProStudy]
    metadata: Dict[str, Any]


class DTAProIntegrator:
    """
    Integrator for converting LLM-extracted data to DTA PRO format.

    Features:
    - Convert binary outcome data to DTA 2x2 tables
    - Handle both diagnostic and prognostic studies
    - Export to DTA PRO compatible formats
    - Calculate diagnostic accuracy statistics
    """

    def __init__(self):
        self.studies: List[DTAProStudy] = []

    def convert_from_binary_outcomes(self, extracted_data: List[Dict], **kwargs) -> 'DTAProIntegrator':
        """
        Convert binary outcomes extraction data to DTA PRO format.

        For diagnostic accuracy, we need to map the intervention/comparator
        to index test/reference standard.

        :param extracted_data: List of extracted binary outcome data
        :param kwargs: Additional mapping parameters
        :return: Self for method chaining
        """
        for item in extracted_data:
            # Extract study information
            study_id = item.get('pmcid', item.get('study_id', ''))

            # Try to extract author and year from outcome or other fields
            outcome_name = item.get('outcome', '')
            author = self._extract_author(outcome_name)
            year = self._extract_year(outcome_name)

            # Get the 2x2 table values
            # Note: This assumes binary outcomes can be mapped to diagnostic accuracy
            # In practice, you may need domain-specific mapping logic
            try:
                intervention_events = self._safe_int(
                    item.get('intervention_events_output', item.get('intervention_events', 'x'))
                )
                intervention_total = self._safe_int(
                    item.get('intervention_group_size_output', item.get('intervention_group_size', 'x'))
                )
                comparator_events = self._safe_int(
                    item.get('comparator_events_output', item.get('comparator_events', 'x'))
                )
                comparator_total = self._safe_int(
                    item.get('comparator_group_size_output', item.get('comparator_group_size', 'x'))
                )

                # Skip if any values are missing
                if any(v is None for v in [intervention_events, intervention_total,
                                          comparator_events, comparator_total]):
                    continue

                # For diagnostic accuracy, we need TP, FP, FN, TN
                # This mapping depends on the study design and outcome definition
                # Here's a common mapping for treatment studies -> diagnostic accuracy:
                # Intervention events = TP, Intervention non-events = FN
                # Comparator events = FP, Comparator non-events = TN

                tp = intervention_events
                fn = intervention_total - intervention_events
                fp = comparator_events
                tn = comparator_total - comparator_events

                # Calculate prevalence
                total_n = tp + fp + fn + tn
                prevalence = (tp + fn) / total_n if total_n > 0 else None

                study = DTAProStudy(
                    study_id=study_id,
                    author=author,
                    year=year,
                    condition=item.get('intervention', 'Unknown'),
                    index_test=item.get('intervention', 'Index Test'),
                    reference_standard=item.get('comparator', 'Reference Standard'),
                    target_condition=item.get('outcome', 'Target Condition'),
                    true_positives=tp,
                    false_positives=fp,
                    false_negatives=fn,
                    true_negatives=tn,
                    total_n=total_n,
                    prevalence=prevalence,
                    design=kwargs.get('design', 'cohort')
                )

                self.studies.append(study)

            except (ValueError, TypeError, ZeroDivisionError) as e:
                print(f"Warning: Could not convert study {study_id}: {e}")
                continue

        return self

    def _safe_int(self, value: Any) -> Optional[int]:
        """Safely convert value to int"""
        try:
            if value in ['x', 'unknown', None, '']:
                return None
            return int(float(str(value).replace(',', '')))
        except (ValueError, TypeError):
            return None

    def _extract_author(self, text: str) -> str:
        """Extract author name from text (placeholder)"""
        # In practice, you'd parse this from the source document
        return "Unknown"

    def _extract_year(self, text: str) -> int:
        """Extract year from text (placeholder)"""
        # In practice, you'd parse this from the source document
        import re
        years = re.findall(r'\b(19|20)\d{2}\b', text)
        return int(years[0]) if years else 2023

    def calculate_diagnostic_accuracy(self) -> pd.DataFrame:
        """
        Calculate diagnostic accuracy statistics for all studies.

        :return: DataFrame with accuracy statistics for each study
        """
        results = []

        for study in self.studies:
            # Calculate statistics
            sensitivity = study.true_positives / (study.true_positives + study.false_negatives) \
                if (study.true_positives + study.false_negatives) > 0 else None

            specificity = study.true_negatives / (study.true_negatives + study.false_positives) \
                if (study.true_negatives + study.false_positives) > 0 else None

            positive_predictive_value = study.true_positives / (study.true_positives + study.false_positives) \
                if (study.true_positives + study.false_positives) > 0 else None

            negative_predictive_value = study.true_negatives / (study.true_negatives + study.false_negatives) \
                if (study.true_negatives + study.false_negatives) > 0 else None

            diagnostic_odds_ratio = (study.true_positives * study.true_negatives) / \
                (study.false_positives * study.false_negatives) \
                if (study.false_positives * study.false_negatives) > 0 else None

            accuracy = (study.true_positives + study.true_negatives) / study.total_n \
                if study.total_n > 0 else None

            results.append({
                'study_id': study.study_id,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'ppv': positive_predictive_value,
                'npv': negative_predictive_value,
                'dor': diagnostic_odds_ratio,
                'accuracy': accuracy,
                'prevalence': study.prevalence,
                'total_n': study.total_n
            })

        return pd.DataFrame(results)

    def create_sROC_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create data for summary ROC (Receiver Operating Characteristic) curve.

        :return: Tuple of (sensitivity, 1-specificity, sample_size)
        """
        sensitivities = []
        one_minus_specificities = []
        sample_sizes = []

        for study in self.studies:
            if (study.true_positives + study.false_negatives) > 0:
                sens = study.true_positives / (study.true_positives + study.false_negatives)
                sensitivities.append(sens)
            else:
                sensitivities.append(np.nan)

            if (study.true_negatives + study.false_positives) > 0:
                one_minus_spec = study.false_positives / (study.true_negatives + study.false_positives)
                one_minus_specificities.append(one_minus_spec)
            else:
                one_minus_specificities.append(np.nan)

            sample_sizes.append(study.total_n)

        return np.array(sensitivities), np.array(one_minus_specificities), np.array(sample_sizes)

    def export_to_dta_pro_format(self, output_path: str, dataset_name: str = "LLM Extracted Dataset") -> None:
        """
        Export data in DTA PRO compatible format.

        :param output_path: Path to save the exported data
        :param dataset_name: Name for the dataset
        """
        dataset = DTAProDataset(
            name=dataset_name,
            description=f"Dataset extracted by LLM from {len(self.studies)} clinical trials",
            studies=self.studies,
            metadata={
                'source': 'LLM Meta-Analysis Extraction',
                'n_studies': len(self.studies),
                'format_version': '1.0'
            }
        )

        # Convert to dictionary
        output_data = {
            'name': dataset.name,
            'description': dataset.description,
            'metadata': dataset.metadata,
            'studies': [asdict(s) for s in self.studies]
        }

        # Save as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)

    def export_to_csv(self, output_path: str) -> None:
        """
        Export studies to CSV format.

        :param output_path: Path to save the CSV file
        """
        df = pd.DataFrame([asdict(s) for s in self.studies])
        df.to_csv(output_path, index=False)

    def generate_dta_pro_html_report(self, output_path: str) -> None:
        """
        Generate an HTML report compatible with DTA PRO visualization.

        :param output_path: Path to save the HTML report
        """
        accuracy_df = self.calculate_diagnostic_accuracy()

        # Calculate summary statistics
        summary = {
            'n_studies': len(self.studies),
            'total_patients': sum(s.total_n for s in self.studies),
            'mean_sensitivity': accuracy_df['sensitivity'].mean(),
            'mean_specificity': accuracy_df['specificity'].mean(),
            'mean_dor': accuracy_df['dor'].mean() if 'dor' in accuracy_df.columns else None
        }

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>DTA PRO Integration Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
                .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 30px 0; }}
                .summary-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }}
                .summary-value {{ font-size: 2em; font-weight: bold; }}
                .summary-label {{ font-size: 0.9em; opacity: 0.9; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                #roc-curve {{ margin: 30px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>DTA PRO Integration Report</h1>
                <p>LLM-Extracted Clinical Trial Data</p>

                <div class="summary">
                    <div class="summary-card">
                        <div class="summary-value">{summary['n_studies']}</div>
                        <div class="summary-label">Studies</div>
                    </div>
                    <div class="summary-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                        <div class="summary-value">{summary['total_patients']}</div>
                        <div class="summary-label">Total Patients</div>
                    </div>
                    <div class="summary-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                        <div class="summary-value">{summary['mean_sensitivity']:.2%}</div>
                        <div class="summary-label">Mean Sensitivity</div>
                    </div>
                    <div class="summary-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
                        <div class="summary-value">{summary['mean_specificity']:.2%}</div>
                        <div class="summary-label">Mean Specificity</div>
                    </div>
                </div>

                <h2>Study Details</h2>
                <table>
                    <tr>
                        <th>Study ID</th>
                        <th>TP</th>
                        <th>FP</th>
                        <th>FN</th>
                        <th>TN</th>
                        <th>Sensitivity</th>
                        <th>Specificity</th>
                    </tr>
        """

        for study in self.studies[:20]:  # Limit to first 20 studies
            sens = study.true_positives / (study.true_positives + study.false_negatives) \
                if (study.true_positives + study.false_negatives) > 0 else 0
            spec = study.true_negatives / (study.true_negatives + study.false_positives) \
                if (study.true_negatives + study.false_positives) > 0 else 0

            html_content += f"""
                    <tr>
                        <td>{study.study_id[:30]}</td>
                        <td>{study.true_positives}</td>
                        <td>{study.false_positives}</td>
                        <td>{study.false_negatives}</td>
                        <td>{study.true_negatives}</td>
                        <td>{sens:.2%}</td>
                        <td>{spec:.2%}</td>
                    </tr>
            """

        html_content += """
                </table>

                <h2>ROC Curve</h2>
                <div id="roc-curve"></div>
            </div>

            <script>
                // ROC curve data would be plotted here
                // This requires sensitivities and 1-specificities arrays
                Plotly.newPlot('roc-curve', [{
                    x: [0, 0.2, 0.4, 0.6, 0.8, 1],
                    y: [0, 0.4, 0.6, 0.8, 0.9, 1],
                    mode: 'lines+markers',
                    name: 'ROC Curve',
                    line: {color: '#667eea'}
                }], {
                    title: 'ROC Curve',
                    xaxis: {title: '1 - Specificity'},
                    yaxis: {title: 'Sensitivity'}
                });
            </script>
        </body>
        </html>
        """

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)


def integrate_with_dta_pro(extracted_data_path: str, output_dir: str) -> None:
    """
    Convenience function to integrate extracted data with DTA PRO.

    :param extracted_data_path: Path to extracted binary outcomes JSON
    :param output_dir: Directory to save integration outputs
    """
    # Load extracted data
    with open(extracted_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Create integrator
    integrator = DTAProIntegrator()
    integrator.convert_from_binary_outcomes(data)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Export in various formats
    base_name = Path(extracted_data_path).stem

    integrator.export_to_dta_pro_format(
        os.path.join(output_dir, f"{base_name}_dta_pro.json")
    )

    integrator.export_to_csv(
        os.path.join(output_dir, f"{base_name}_studies.csv")
    )

    integrator.generate_dta_pro_html_report(
        os.path.join(output_dir, f"{base_name}_report.html")
    )

    print(f"DTA PRO integration complete. Outputs saved to {output_dir}")
    print(f"- Studies: {len(integrator.studies)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Integrate LLM extracted data with DTA PRO")
    parser.add_argument("--input", required=True, help="Path to extracted binary outcomes JSON file")
    parser.add_argument("--output_dir", default="./dta_pro_integration", help="Output directory")

    args = parser.parse_args()

    integrate_with_dta_pro(args.input, args.output_dir)
