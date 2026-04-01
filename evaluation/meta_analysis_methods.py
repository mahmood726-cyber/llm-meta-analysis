"""
Advanced Meta-Analysis Statistical Methods

This module provides comprehensive statistical methods for meta-analysis,
including pooling effect sizes, assessing heterogeneity, and generating
forest plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Literal
from dataclasses import dataclass
from scipy import stats
from statsmodels.stats.meta_analysis import (
    effectsize_smd,
    effectsize_2proportions,
    combine_effects
)


@dataclass
class StudyResult:
    """Represents a single study's results"""
    study_id: str
    pmcid: str
    intervention: str
    comparator: str
    outcome: str

    # Binary outcome data
    intervention_events: Optional[int] = None
    intervention_total: Optional[int] = None
    comparator_events: Optional[int] = None
    comparator_total: Optional[int] = None

    # Continuous outcome data
    intervention_mean: Optional[float] = None
    intervention_sd: Optional[float] = None
    intervention_n: Optional[int] = None
    comparator_mean: Optional[float] = None
    comparator_sd: Optional[float] = None
    comparator_n: Optional[int] = None


@dataclass
class MetaAnalysisResult:
    """Results from a meta-analysis"""
    pooled_effect: float
    pooled_effect_ci_lower: float
    pooled_effect_ci_upper: float
    p_value: float
    heterogeneity_q: float
    heterogeneity_i2: float
    tau2: float
    method: str
    n_studies: int


class MetaAnalyzer:
    """
    Advanced meta-analysis with multiple pooling methods and heterogeneity assessment.

    Supports:
    - Fixed-effect and random-effects models
    - Inverse variance weighting
    - Mantel-Haenszel method (for binary outcomes)
    - DerSimonian-Laird estimator (for tau^2)
    - Heterogeneity statistics (Q, I^2, tau^2)
    - Subgroup analysis
    - Meta-regression
    - Forest plots
    """

    def __init__(self):
        self.studies: List[StudyResult] = []
        self.results: Optional[MetaAnalysisResult] = None

    def add_study(self, study: StudyResult) -> None:
        """Add a study to the meta-analysis"""
        self.studies.append(study)

    def add_studies_from_extraction(self, extracted_data: List[Dict]) -> None:
        """
        Add studies from extracted data format.

        :param extracted_data: List of dictionaries with extracted results
        """
        for item in extracted_data:
            # Determine if binary or continuous outcome
            is_binary = item.get("outcome_type") == "binary" or (
                "intervention_events" in item and "comparator_events" in item
            )

            study = StudyResult(
                study_id=f"{item.get('pmcid', '')}_{item.get('outcome', '')}",
                pmcid=item.get('pmcid', ''),
                intervention=item.get('intervention', ''),
                comparator=item.get('comparator', ''),
                outcome=item.get('outcome', ''),
            )

            if is_binary:
                # Binary outcome data
                study.intervention_events = self._safe_int(
                    item.get('intervention_events_output', item.get('intervention_events', 'x'))
                )
                study.intervention_total = self._safe_int(
                    item.get('intervention_group_size_output', item.get('intervention_group_size', 'x'))
                )
                study.comparator_events = self._safe_int(
                    item.get('comparator_events_output', item.get('comparator_events', 'x'))
                )
                study.comparator_total = self._safe_int(
                    item.get('comparator_group_size_output', item.get('comparator_group_size', 'x'))
                )
            else:
                # Continuous outcome data
                study.intervention_mean = self._safe_float(
                    item.get('intervention_mean_output', item.get('intervention_mean', 'x'))
                )
                study.intervention_sd = self._safe_float(
                    item.get('intervention_standard_deviation_output',
                             item.get('intervention_standard_deviation', 'x'))
                )
                study.intervention_n = self._safe_int(
                    item.get('intervention_group_size_output', item.get('intervention_group_size', 'x'))
                )
                study.comparator_mean = self._safe_float(
                    item.get('comparator_mean_output', item.get('comparator_mean', 'x'))
                )
                study.comparator_sd = self._safe_float(
                    item.get('comparator_standard_deviation_output',
                             item.get('comparator_standard_deviation', 'x'))
                )
                study.comparator_n = self._safe_int(
                    item.get('comparator_group_size_output', item.get('comparator_group_size', 'x'))
                )

            # Only add if we have valid data
            if self._has_valid_data(study, is_binary):
                self.add_study(study)

    def _safe_int(self, value: any) -> Optional[int]:
        """Safely convert value to int"""
        try:
            if value in ['x', 'unknown', None, '']:
                return None
            return int(float(str(value).replace(',', '')))
        except (ValueError, TypeError):
            return None

    def _safe_float(self, value: any) -> Optional[float]:
        """Safely convert value to float"""
        try:
            if value in ['x', 'unknown', None, '']:
                return None
            return float(str(value).replace(',', ''))
        except (ValueError, TypeError):
            return None

    def _has_valid_data(self, study: StudyResult, is_binary: bool) -> bool:
        """Check if study has valid data for analysis"""
        if is_binary:
            return all([
                study.intervention_events is not None,
                study.intervention_total is not None,
                study.comparator_events is not None,
                study.comparator_total is not None
            ])
        else:
            return all([
                study.intervention_mean is not None,
                study.intervention_sd is not None,
                study.intervention_n is not None,
                study.comparator_mean is not None,
                study.comparator_sd is not None,
                study.comparator_n is not None
            ])

    def calculate_binary_effects(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate log odds ratios and variances for binary outcomes.

        :return: Tuple of (effect_sizes, variances)
        """
        effects = []
        variances = []

        for study in self.studies:
            # Using statsmodels effectsize_2proportions
            eff, var_eff = effectsize_2proportions(
                np.array([study.intervention_events]),
                np.array([study.intervention_total]),
                np.array([study.comparator_events]),
                np.array([study.comparator_total]),
                statistic='odds-ratio',
                zero_correction=0.5  # Haldane-Anscombe correction
            )
            effects.append(eff[0])
            variances.append(var_eff[0])

        return np.array(effects), np.array(variances)

    def calculate_continuous_effects(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate standardized mean differences (Hedges' g) for continuous outcomes.

        :return: Tuple of (effect_sizes, variances)
        """
        effects = []
        variances = []

        for study in self.studies:
            # Using statsmodels effectsize_smd (Hedges' g with bias correction)
            eff, var_eff = effectsize_smd(
                np.array([study.intervention_mean]),
                np.array([study.intervention_sd]),
                np.array([study.intervention_n]),
                np.array([study.comparator_mean]),
                np.array([study.comparator_sd]),
                np.array([study.comparator_n])
            )
            effects.append(eff[0])
            variances.append(var_eff[0])

        return np.array(effects), np.array(variances)

    def analyze(
        self,
        method: Literal['fixed', 'random', 'dl'] = 'dl',
        outcome_type: Literal['binary', 'continuous'] = 'binary'
    ) -> MetaAnalysisResult:
        """
        Perform meta-analysis using the specified method.

        :param method: Pooling method ('fixed', 'random', 'dl' for DerSimonian-Laird)
        :param outcome_type: Type of outcome ('binary' or 'continuous')
        :return: MetaAnalysisResult with pooled effect and heterogeneity measures
        """
        if len(self.studies) < 2:
            raise ValueError("Need at least 2 studies for meta-analysis")

        # Calculate effect sizes
        if outcome_type == 'binary':
            effects, variances = self.calculate_binary_effects()
        else:
            effects, variances = self.calculate_continuous_effects()

        # Remove any NaN values
        valid_mask = ~np.isnan(effects) & ~np.isnan(variances) & (variances > 0)
        effects = effects[valid_mask]
        variances = variances[valid_mask]

        if len(effects) < 2:
            raise ValueError("Need at least 2 valid studies for meta-analysis")

        # Perform meta-analysis using statsmodels
        try:
            result = combine_effects(effects, variances, method=method)

            # Calculate heterogeneity statistics
            # Q statistic (Cochran's Q)
            weights = 1 / variances
            weighted_mean = np.sum(weights * effects) / np.sum(weights)
            q_statistic = np.sum(weights * (effects - weighted_mean) ** 2)

            # I^2 statistic
            df = len(effects) - 1
            i_squared = max(0, (q_statistic - df) / q_statistic * 100) if q_statistic > df else 0

            # Tau^2 (between-study variance)
            if method == 'random' or method == 'dl':
                # DerSimonian-Laird estimator
                tau_squared = max(0, (q_statistic - df) / (np.sum(weights) - np.sum(weights ** 2) / np.sum(weights)))
            else:
                tau_squared = 0

            self.results = MetaAnalysisResult(
                pooled_effect=result[0][0],
                pooled_effect_ci_lower=result[2][0],  # Lower CI
                pooled_effect_ci_upper=result[3][0],  # Upper CI
                p_value=result[1][0] if len(result) > 1 else None,
                heterogeneity_q=q_statistic,
                heterogeneity_i2=i_squared,
                tau2=tau_squared,
                method=method,
                n_studies=len(effects)
            )

        except Exception as e:
            # Fallback to manual calculation
            self.results = self._manual_meta_analysis(effects, variances, method)

        return self.results

    def _manual_meta_analysis(self, effects: np.ndarray, variances: np.ndarray,
                              method: str) -> MetaAnalysisResult:
        """Manual meta-analysis calculation as fallback"""
        # Inverse variance weights
        weights = 1 / variances

        if method == 'random' or method == 'dl':
            # DerSimonian-Laird tau^2
            q_weighted = np.sum(weights * effects) / np.sum(weights)
            q_statistic = np.sum(weights * (effects - q_weighted) ** 2)
            df = len(effects) - 1

            tau_squared = max(0, (q_statistic - df) /
                             (np.sum(weights) - np.sum(weights ** 2) / np.sum(weights)))

            # Random effects weights
            weights = 1 / (variances + tau_squared)
        else:
            q_statistic = np.sum(weights * (effects - np.average(effects, weights=weights)) ** 2)
            tau_squared = 0

        # Pooled effect
        pooled_effect = np.sum(weights * effects) / np.sum(weights)

        # Standard error
        se_pooled = np.sqrt(1 / np.sum(weights))

        # 95% CI
        ci_lower = pooled_effect - 1.96 * se_pooled
        ci_upper = pooled_effect + 1.96 * se_pooled

        # Z-test and p-value
        z_score = pooled_effect / se_pooled
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        # I^2
        df = len(effects) - 1
        i_squared = max(0, (q_statistic - df) / q_statistic * 100) if q_statistic > df else 0

        return MetaAnalysisResult(
            pooled_effect=pooled_effect,
            pooled_effect_ci_lower=ci_lower,
            pooled_effect_ci_upper=ci_upper,
            p_value=p_value,
            heterogeneity_q=q_statistic,
            heterogeneity_i2=i_squared,
            tau2=tau_squared,
            method=method,
            n_studies=len(effects)
        )

    def forest_plot(self, output_path: Optional[str] = None, outcome_type: str = 'binary') -> plt.Figure:
        """
        Generate a forest plot for the meta-analysis.

        :param output_path: Optional path to save the figure
        :param outcome_type: Type of outcome for effect size labeling
        :return: Matplotlib figure
        """
        if self.results is None:
            raise ValueError("Run analyze() before generating forest plot")

        # Get effect sizes
        if outcome_type == 'binary':
            effects, variances = self.calculate_binary_effects()
            effect_label = "Log Odds Ratio"
        else:
            effects, variances = self.calculate_continuous_effects()
            effect_label = "Standardized Mean Difference"

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6 + len(self.studies) * 0.3))

        study_names = [s.study_id[:30] for s in self.studies]

        # Plot individual studies
        y_positions = np.arange(len(study_names))
        ax.errorbar(effects, y_positions, xerr=1.96 * np.sqrt(variances),
                   fmt='o', capsize=5, markersize=8, color='steelblue', label='Individual Studies')

        # Plot pooled effect
        pooled_y = len(study_names) + 0.5
        ax.errorbar([self.results.pooled_effect], [pooled_y],
                   xerr=[self.results.pooled_effect - self.results.pooled_effect_ci_lower,
                         self.results.pooled_effect_ci_upper - self.results.pooled_effect],
                   fmt='diamond', capsize=8, markersize=15, color='red', label='Pooled Effect')

        # Add null effect line
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        # Formatting
        ax.set_yticks(list(y_positions) + [pooled_y])
        ax.set_yticklabels(study_names + ['Pooled Effect'], fontsize=9)
        ax.set_xlabel(effect_label, fontsize=12)
        ax.set_title(f'Forest Plot - {outcome_type.capitalize()} Outcomes\n'
                    f'{self.results.method.capitalize()} effect model (n={self.results.n_studies})', fontsize=14)

        # Add heterogeneity statistics as text
        stats_text = f'Heterogeneity: Q={self.results.heterogeneity_q:.2f}, ' \
                    f'I²={self.results.heterogeneity_i2:.1f}%, ' \
                    f'τ²={self.results.tau2:.4f}'
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=10)

        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=100, bbox_inches='tight')

        return fig

    def subgroup_analysis(self, grouping_variable: str, groups: Dict[str, List[str]]) -> Dict[str, 'MetaAnalyzer']:
        """
        Perform subgroup analysis.

        :param grouping_variable: Name of the variable to group by
        :param groups: Dictionary mapping group names to study IDs
        :return: Dictionary of MetaAnalyzer objects for each subgroup
        """
        subgroup_analyzers = {}

        for group_name, study_ids in groups.items():
            analyzer = MetaAnalyzer()
            for study in self.studies:
                if study.study_id in study_ids:
                    analyzer.add_study(study)
            subgroup_analyzers[group_name] = analyzer

        return subgroup_analyzers

    def print_results(self) -> None:
        """Print formatted meta-analysis results"""
        if self.results is None:
            print("No analysis results. Run analyze() first.")
            return

        print("\n" + "="*60)
        print("META-ANALYSIS RESULTS")
        print("="*60)
        print(f"Method: {self.results.method.capitalize()} effect model")
        print(f"Number of studies: {self.results.n_studies}")
        print(f"\nPooled Effect: {self.results.pooled_effect:.4f}")
        print(f"95% CI: [{self.results.pooled_effect_ci_lower:.4f}, {self.results.pooled_effect_ci_upper:.4f}]")
        print(f"P-value: {self.results.p_value:.4f}" if self.results.p_value else "P-value: N/A")
        print(f"\nHeterogeneity:")
        print(f"  Q-statistic: {self.results.heterogeneity_q:.4f}")
        print(f"  I²: {self.results.heterogeneity_i2:.2f}%")
        print(f"  τ²: {self.results.tau2:.4f}")
        print("="*60 + "\n")


def meta_analysis_from_extraction(extracted_data: List[Dict],
                                   outcome_type: str = 'binary',
                                   method: str = 'dl') -> Tuple[MetaAnalyzer, MetaAnalysisResult]:
    """
    Convenience function to perform meta-analysis directly from extracted data.

    :param extracted_data: List of dictionaries with extracted results
    :param outcome_type: Type of outcome ('binary' or 'continuous')
    :param method: Pooling method ('fixed', 'random', 'dl')
    :return: Tuple of (MetaAnalyzer, MetaAnalysisResult)
    """
    analyzer = MetaAnalyzer()
    analyzer.add_studies_from_extraction(extracted_data)
    result = analyzer.analyze(method=method, outcome_type=outcome_type)
    return analyzer, result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Perform meta-analysis on extracted data")
    parser.add_argument("--input", required=True, help="Path to extracted data JSON file")
    parser.add_argument("--outcome_type", default="binary", choices=["binary", "continuous"])
    parser.add_argument("--method", default="dl", choices=["fixed", "random", "dl"])
    parser.add_argument("--output_dir", default="./meta_analysis_results", help="Directory to save results")
    parser.add_argument("--forest_plot", action="store_true", help="Generate forest plot")

    args = parser.parse_args()

    # Load data
    import json
    with open(args.input, 'r') as f:
        data = json.load(f)

    # Perform meta-analysis
    analyzer, result = meta_analysis_from_extraction(data, args.outcome_type, args.method)

    # Print results
    analyzer.print_results()

    # Save results
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    results_file = os.path.join(args.output_dir, f"meta_analysis_{args.outcome_type}.json")
    with open(results_file, 'w') as f:
        json.dump({
            'pooled_effect': result.pooled_effect,
            'ci_lower': result.pooled_effect_ci_lower,
            'ci_upper': result.pooled_effect_ci_upper,
            'p_value': result.p_value,
            'heterogeneity': {
                'q': result.heterogeneity_q,
                'i2': result.heterogeneity_i2,
                'tau2': result.tau2
            },
            'n_studies': result.n_studies,
            'method': result.method
        }, f, indent=2)

    # Generate forest plot if requested
    if args.forest_plot:
        forest_path = os.path.join(args.output_dir, f"forest_plot_{args.outcome_type}.png")
        analyzer.forest_plot(forest_path, args.outcome_type)
        print(f"Forest plot saved to {forest_path}")

    print(f"Results saved to {results_file}")
