"""
Cumulative Meta-Analysis and Scan Statistics (DEPRECATED)

⚠️ DEPRECATION WARNING (v2.0): This module is deprecated.

Please use cumulative_analysis_v2.py instead, which includes:
- Explicit temporal data validation with clear error messages
- Lan-DeMets alpha-spending function (not just O'Brien-Fleming)
- Multiple spending functions (O'Brien-Fleming, Pocock, power family)
- Proper handling of edge cases

Migration: Replace `from cumulative_analysis import` with `from cumulative_analysis_v2 import`

This module will be removed in version 3.0.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from scipy import stats
from scipy.stats import norm, chi2
import warnings

# Issue deprecation warning
warnings.warn(
    "cumulative_analysis.py is deprecated. Use cumulative_analysis_v2.py instead. "
    "This module will be removed in version 3.0.",
    DeprecationWarning,
    stacklevel=2
)


@dataclass
class CumulativeResult:
    """Result from cumulative meta-analysis at a specific time point"""
    n_studies_included: int
    pooled_effect: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    z_score: float
    p_value: float
    heterogeneity_q: float
    heterogeneity_i2: float
    tau_squared: float
    publication_date: Optional[str] = None
    study_id: Optional[str] = None
    is_significant: bool = False
    reached_conclusion: bool = False


@dataclass
class ScanStatisticResult:
    """Result from scan statistics analysis"""
    cluster_start: int
    cluster_end: int
    cluster_size: int
    observed_statistic: float
    expected_statistic: float
    relative_risk: float
    p_value: float
    cluster_studies: List[str]
    cluster_effects: np.ndarray
    cluster_mean: float
    overall_mean: float


@dataclass
class TrialSequentialBoundary:
    """Trial sequential analysis boundaries"""
    n_studies: int
    boundary_upper: float
    boundary_lower: float
    information_size: float
    alpha: float
    beta: float
    type_i_error: float
    type_ii_error: float
    futility_boundaries: np.ndarray
    monitoring_boundaries: np.ndarray


class CumulativeMetaAnalyzer:
    """
    Cumulative meta-analysis for monitoring evidence accumulation.

    Tracks how pooled estimates evolve as new studies are added,
    helping identify when conclusions become stable.
    """

    def __init__(self):
        """Initialize cumulative analyzer"""
        self.study_order: List[str] = []
        self.cumulative_results: List[CumulativeResult] = []
        self.final_result: Optional[CumulativeResult] = None

    def analyze(
        self,
        effects: np.ndarray,
        variances: np.ndarray,
        study_ids: List[str],
        publication_dates: Optional[List[str]] = None,
        order_by: str = "chronological",
        effect_measure: str = "MD",
        pooled_estimate_target: Optional[float] = None,
        alpha: float = 0.05
    ) -> List[CumulativeResult]:
        """
        Perform cumulative meta-analysis.

        :param effects: Study effect estimates
        :param variances: Study variances
        :param study_ids: Study identifiers
        :param publication_dates: Publication dates for chronological ordering
        :param order_by: Ordering method ('chronological', 'precision', 'custom')
        :param effect_measure: Type of effect measure ('MD', 'SMD', 'OR', 'RR')
        :param pooled_estimate_target: Target effect size for TSA
        :param alpha: Significance level
        :return: List of cumulative results
        """
        n_studies = len(effects)

        if n_studies == 0:
            raise ValueError("No studies provided")

        # Determine order
        if order_by == "chronological" and publication_dates is not None:
            order = np.argsort(publication_dates)
        elif order_by == "precision":
            order = np.argsort(variances)[::-1]  # Most precise first
        else:
            order = np.arange(n_studies)

        self.study_order = [study_ids[i] for i in order]

        # Cumulative analysis
        cumulative_results = []

        for i in range(1, n_studies + 1):
            # Studies 1 through i
            included_effects = effects[order[:i]]
            included_variances = variances[order[:i]]

            # Compute pooled estimate
            result = self._compute_cumulative_estimate(
                included_effects,
                included_variances,
                study_ids[order[i-1]],
                publication_dates[order[i-1]] if publication_dates else None
            )

            # Check if significant
            result.is_significant = result.p_value < alpha

            cumulative_results.append(result)

        self.cumulative_results = cumulative_results
        self.final_result = cumulative_results[-1]

        return cumulative_results

    def _compute_cumulative_estimate(
        self,
        effects: np.ndarray,
        variances: np.ndarray,
        study_id: str,
        publication_date: Optional[str]
    ) -> CumulativeResult:
        """Compute pooled estimate for cumulative studies"""
        n = len(effects)

        # Inverse variance weights
        weights = 1 / variances
        sum_weights = np.sum(weights)

        # Pooled effect (fixed effect)
        pooled_effect = np.sum(weights * effects) / sum_weights
        se = np.sqrt(1 / sum_weights)

        # Confidence interval
        z = 1.96
        ci_lower = pooled_effect - z * se
        ci_upper = pooled_effect + z * se

        # Z-score and p-value
        z_score = pooled_effect / se
        p_value = 2 * (1 - norm.cdf(abs(z_score)))

        # Heterogeneity
        q_statistic = np.sum(weights * (effects - pooled_effect)**2)
        df = n - 1

        if df > 0 and q_statistic > df:
            i2 = max(0, 100 * (q_statistic - df) / q_statistic)
        else:
            i2 = 0

        # Tau squared (DerSimonian-Laird)
        if df > 0:
            tau_squared = max(0, (q_statistic - df) / (sum_weights - np.sum(weights**2) / sum_weights))
        else:
            tau_squared = 0

        return CumulativeResult(
            n_studies_included=n,
            pooled_effect=pooled_effect,
            standard_error=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            z_score=z_score,
            p_value=p_value,
            heterogeneity_q=q_statistic,
            heterogeneity_i2=i2,
            tau_squared=tau_squared,
            publication_date=publication_date,
            study_id=study_id,
            is_significant=False,
            reached_conclusion=False
        )

    def identify_stability_point(
        self,
        stability_criteria: str = "ci_width",
        tolerance: float = 0.1,
        min_studies: int = 3
    ) -> Optional[int]:
        """
        Identify when pooled estimate becomes stable.

        :param stability_criteria: Method to assess stability
        :param tolerance: Tolerance for stability
        :param min_studies: Minimum studies before assessing stability
        :return: Study index where stability is reached
        """
        if not self.cumulative_results:
            return None

        if len(self.cumulative_results) < min_studies:
            return None

        if stability_criteria == "ci_width":
            # CI width stabilizes
            for i in range(min_studies, len(self.cumulative_results)):
                ci_width = (
                    self.cumulative_results[i].ci_upper -
                    self.cumulative_results[i].ci_lower
                )
                prev_width = (
                    self.cumulative_results[i-1].ci_upper -
                    self.cumulative_results[i-1].ci_lower
                )

                if abs(ci_width - prev_width) / prev_width < tolerance:
                    return i

        elif stability_criteria == "effect_change":
            # Pooled effect stabilizes
            for i in range(min_studies, len(self.cumulative_results)):
                effect_change = abs(
                    self.cumulative_results[i].pooled_effect -
                    self.cumulative_results[i-1].pooled_effect
                )

                if effect_change < tolerance:
                    return i

        elif stability_criteria == "significance":
            # Significance status stabilizes
            for i in range(min_studies, len(self.cumulative_results)):
                if (self.cumulative_results[i].is_significant and
                    self.cumulative_results[i-1].is_significant):
                    # Check next few studies also significant
                    if (i + 2 < len(self.cumulative_results) and
                        self.cumulative_results[i+1].is_significant and
                        self.cumulative_results[i+2].is_significant):
                        return i

        return None

    def detect_reversal(
        self,
        reversal_threshold: float = 0.5
    ) -> List[Tuple[int, int, str]]:
        """
        Detect effect reversals over time.

        :param reversal_threshold: Effect size change to qualify as reversal
        :return: List of (from_study, to_study, direction) reversals
        """
        if not self.cumulative_results:
            return []

        reversals = []

        for i in range(1, len(self.cumulative_results)):
            prev_effect = self.cumulative_results[i-1].pooled_effect
            curr_effect = self.cumulative_results[i].pooled_effect

            # Sign change with substantial magnitude
            if (prev_effect * curr_effect < 0 and
                abs(curr_effect - prev_effect) > reversal_threshold):

                direction = "positive_to_negative" if prev_effect > 0 else "negative_to_positive"
                reversals.append((i-1, i, direction))

        return reversals

    def plot_cumulative_forest(
        self,
        output_path: Optional[str] = None
    ) -> None:
        """
        Generate cumulative forest plot.

        :param output_path: Path to save plot
        """
        try:
            import matplotlib.pyplot as plt

            if not self.cumulative_results:
                print("No cumulative results to plot")
                return

            fig, ax = plt.subplots(figsize=(10, 6 + len(self.cumulative_results) * 0.3))

            y_positions = np.arange(len(self.cumulative_results))

            # Plot each cumulative result
            effects = [r.pooled_effect for r in self.cumulative_results]
            ci_lowers = [r.ci_lower for r in self.cumulative_results]
            ci_uppers = [r.ci_upper for r in self.cumulative_results]

            # Confidence intervals
            ax.errorbar(
                effects, y_positions,
                xerr=[np.array(effects) - np.array(ci_lowers),
                      np.array(ci_uppers) - np.array(effects)],
                fmt='o', capsize=5, capthick=2, markersize=8,
                color='steelblue', ecolor='steelblue'
            )

            # Reference line at null
            ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

            # Study labels
            study_labels = [
                f"Cumulative at study {i+1} ({r.study_id})"
                for i, r in enumerate(self.cumulative_results)
            ]
            ax.set_yticks(y_positions)
            ax.set_yticklabels(study_labels)

            ax.set_xlabel('Cumulative Effect Size', fontsize=12)
            ax.set_ylabel('Cumulative Analysis', fontsize=12)
            ax.set_title('Cumulative Meta-Analysis Forest Plot', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)

            plt.tight_layout()

            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()

        except ImportError:
            print("Matplotlib not available for plotting")


class ScanStatisticsAnalyzer:
    """
    Scan statistics for detecting temporal/spatial clusters.

    Identifies unusual clusters of studies with similar effect sizes
    or outcome patterns.
    """

    def __init__(self):
        """Initialize scan statistics analyzer"""
        self.results: List[ScanStatisticResult] = []

    def temporal_scan(
        self,
        effects: np.ndarray,
        study_ids: List[str],
        publication_dates: List[str],
        window_size: int = 3,
        n_permutations: int = 999
    ) -> List[ScanStatisticResult]:
        """
        Perform temporal scan statistics.

        :param effects: Study effect estimates
        :param study_ids: Study identifiers
        :param publication_dates: Publication dates (sorted)
        :param window_size: Size of sliding window
        :param n_permutations: Number of permutations for p-value
        :return: List of detected clusters
        """
        n_studies = len(effects)

        if window_size >= n_studies:
            window_size = n_studies - 1

        # Sliding window scan
        clusters = []

        for i in range(n_studies - window_size + 1):
            cluster_effects = effects[i:i + window_size]

            # Test if cluster mean differs from overall mean
            cluster_mean = np.mean(cluster_effects)
            overall_mean = np.mean(effects)

            # Standardized statistic
            cluster_se = np.std(cluster_effects) / np.sqrt(window_size)
            if cluster_se > 0:
                z_score = abs(cluster_mean - overall_mean) / cluster_se
            else:
                z_score = 0

            # Permutation test for p-value
            p_value = self._permutation_test(
                effects, window_size, abs(cluster_mean - overall_mean), n_permutations
            )

            # Relative risk (or effect ratio)
            if overall_mean != 0:
                relative_risk = cluster_mean / overall_mean
            else:
                relative_risk = 1.0

            clusters.append(ScanStatisticResult(
                cluster_start=i,
                cluster_end=i + window_size - 1,
                cluster_size=window_size,
                observed_statistic=z_score,
                expected_statistic=0,
                relative_risk=relative_risk,
                p_value=p_value,
                cluster_studies=[study_ids[j] for j in range(i, i + window_size)],
                cluster_effects=cluster_effects,
                cluster_mean=cluster_mean,
                overall_mean=overall_mean
            ))

        # Filter significant clusters
        significant_clusters = [c for c in clusters if c.p_value < 0.05]

        # Remove overlapping clusters (keep most significant)
        self.results = self._remove_overlaps(significant_clusters)

        return self.results

    def _permutation_test(
        self,
        effects: np.ndarray,
        window_size: int,
        observed_diff: float,
        n_permutations: int
    ) -> float:
        """Permutation test for cluster significance"""
        count_more_extreme = 0

        for _ in range(n_permutations):
            permuted = np.random.permutation(effects)

            # Check all windows
            max_diff = 0
            for i in range(len(effects) - window_size + 1):
                window_mean = np.mean(permuted[i:i + window_size])
                overall_mean = np.mean(permuted)
                diff = abs(window_mean - overall_mean)
                max_diff = max(max_diff, diff)

            if max_diff >= observed_diff:
                count_more_extreme += 1

        return (count_more_extreme + 1) / (n_permutations + 1)

    def _remove_overlaps(
        self,
        clusters: List[ScanStatisticResult]
    ) -> List[ScanStatisticResult]:
        """Remove overlapping clusters, keeping most significant"""
        if not clusters:
            return []

        # Sort by p-value
        sorted_clusters = sorted(clusters, key=lambda x: x.p_value)

        non_overlapping = []
        used_indices = set()

        for cluster in sorted_clusters:
            cluster_indices = set(range(cluster.cluster_start, cluster.cluster_end + 1))

            if not cluster_indices & used_indices:
                non_overlapping.append(cluster)
                used_indices.update(cluster_indices)

        return non_overlapping


class TrialSequentialAnalyzer:
    """
    Trial Sequential Analysis (TSA).

    Monitors cumulative evidence with adjusted boundaries to control
    type I and II errors in sequential testing.
    """

    def __init__(self):
        """Initialize TSA analyzer"""
        self.boundaries: Optional[TrialSequentialBoundary] = None

    def calculate_information_size(
        self,
        effect_size: float,
        variance: float,
        alpha: float = 0.05,
        beta: float = 0.20,
        delta: float = None
    ) -> float:
        """
        Calculate required information size (optimal sample size).

        :param effect_size: True effect size
        :param variance: Variance of effect size
        :param alpha: Type I error rate
        :param beta: Type II error rate
        :param delta: Minimal clinically important difference
        :return: Required information size
        """
        if delta is None:
            delta = abs(effect_size)

        # Z values
        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(1 - beta)

        # Required information size
        # Based on Wetterslev et al. formula
        n_is = 4 * (z_alpha + z_beta)**2 * variance / delta**2

        return n_is

    def construct_boundaries(
        self,
        n_studies: int,
        information_size: float,
        alpha: float = 0.05,
        beta: float = 0.20,
        side: str = "two-sided"
    ) -> TrialSequentialBoundary:
        """
        Construct trial sequential boundaries.

        :param n_studies: Number of interim analyses
        :param information_size: Total required information
        :param alpha: Type I error
        :param beta: Type II error
        :param side: 'two-sided' or 'one-sided'
        :return: Boundary object
        """
        # Information fractions
        information_fractions = np.linspace(0, 1, n_studies + 1)[1:]

        # O'Brien-Fleming type boundaries
        if side == "two-sided":
            z_alpha = norm.ppf(1 - alpha / 2)
        else:
            z_alpha = norm.ppf(1 - alpha)

        # Spending function approach (Lan-DeMets)
        # Alpha spending
        alpha_spent = np.zeros(n_studies)
        for i, t in enumerate(information_fractions):
            alpha_spent[i] = 2 - 2 * norm.ppf(z_alpha / np.sqrt(t))

        # Boundary values
        boundary_values = z_alpha / np.sqrt(information_fractions)

        # Futility boundaries (non-binding)
        z_beta = norm.ppf(1 - beta)
        futility_values = -z_beta / np.sqrt(information_fractions)

        self.boundaries = TrialSequentialBoundary(
            n_studies=n_studies,
            boundary_upper=boundary_values[-1],
            boundary_lower=-boundary_values[-1],
            information_size=information_size,
            alpha=alpha,
            beta=beta,
            type_i_error=alpha,
            type_ii_error=beta,
            futility_boundaries=futility_values,
            monitoring_boundaries=boundary_values
        )

        return self.boundaries

    def assess_trial_status(
        self,
        cumulative_z_score: float,
        information_fraction: float
    ) -> str:
        """
        Assess trial status against boundaries.

        :param cumulative_z_score: Cumulative Z statistic
        :param information_fraction: Proportion of information accrued
        :return: Status ('continue', 'efficacy', 'futility')
        """
        if self.boundaries is None:
            return "continue"

        # Find boundary at this information fraction
        idx = int(information_fraction * self.boundaries.n_studies) - 1
        idx = max(0, min(idx, self.boundaries.n_studies - 1))

        upper_bound = self.boundaries.monitoring_boundaries[idx]
        lower_bound = -upper_bound

        if cumulative_z_score >= upper_bound:
            return "efficacy"
        elif cumulative_z_score <= lower_bound:
            return "futility"  # For beneficial effect
        else:
            return "continue"

    def plot_tsa_boundaries(
        self,
        cumulative_z_scores: List[float],
        output_path: Optional[str] = None
    ) -> None:
        """
        Plot TSA boundaries with cumulative Z scores.

        :param cumulative_z_scores: List of cumulative Z scores
        :param output_path: Path to save plot
        """
        try:
            import matplotlib.pyplot as plt

            if self.boundaries is None:
                print("No boundaries to plot")
                return

            n = len(cumulative_z_scores)
            information_fractions = np.linspace(0, 1, n + 1)[1:]

            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot boundaries
            ax.plot(
                information_fractions,
                self.boundaries.monitoring_boundaries[:n],
                'r-', linewidth=2, label='Efficacy boundary'
            )
            ax.plot(
                information_fractions,
                -self.boundaries.monitoring_boundaries[:n],
                'r-', linewidth=2
            )

            # Plot futility boundaries
            ax.plot(
                information_fractions,
                self.boundaries.futility_boundaries[:n],
                'b--', linewidth=1.5, label='Futility boundary'
            )

            # Plot cumulative Z scores
            ax.plot(
                information_fractions,
                cumulative_z_scores,
                'ko-', markersize=6, label='Cumulative Z-score'
            )

            # Reference lines
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            ax.axhline(y=1.96, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.axhline(y=-1.96, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

            ax.set_xlabel('Information Fraction', fontsize=12)
            ax.set_ylabel('Cumulative Z-score', fontsize=12)
            ax.set_title('Trial Sequential Analysis', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)

            plt.tight_layout()

            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()

        except ImportError:
            print("Matplotlib not available for plotting")


class EvolutionaryAnalyzer:
    """
    Analyze evolution of evidence over time.

    Tracks how effect sizes, heterogeneity, and conclusions change
    as new evidence accumulates.
    """

    def __init__(self):
        """Initialize evolutionary analyzer"""
        self.cumulative_analyzer = CumulativeMetaAnalyzer()

    def analyze_evolution(
        self,
        effects: np.ndarray,
        variances: np.ndarray,
        study_ids: List[str],
        publication_dates: List[str]
    ) -> Dict:
        """
        Analyze evidence evolution.

        :param effects: Study effects
        :param variances: Study variances
        :param study_ids: Study identifiers
        :param publication_dates: Publication dates
        :return: Dictionary with evolution metrics
        """
        # Cumulative analysis
        cumulative = self.cumulative_analyzer.analyze(
            effects, variances, study_ids, publication_dates, order_by="chronological"
        )

        if not cumulative:
            return {}

        # Evolution metrics
        effects_over_time = [r.pooled_effect for r in cumulative]
        ci_widths_over_time = [
            r.ci_upper - r.ci_lower for r in cumulative
        ]
        p_values_over_time = [r.p_value for r in cumulative]
        i2_over_time = [r.heterogeneity_i2 for r in cumulative]

        # Evolution trends
        effect_trend = np.polyfit(
            range(len(effects_over_time)),
            effects_over_time,
            1
        )[0]

        precision_trend = np.polyfit(
            range(len(ci_widths_over_time)),
            [-w for w in ci_widths_over_time],  # Negative because decreasing width = increasing precision
            1
        )[0]

        # Identify key events
        first_significant = next(
            (i for i, r in enumerate(cumulative) if r.is_significant),
            None
        )

        stability_point = self.cumulative_analyzer.identify_stability_point()

        reversals = self.cumulative_analyzer.detect_reversal()

        return {
            "n_studies": len(effects),
            "effects_over_time": effects_over_time,
            "ci_widths_over_time": ci_widths_over_time,
            "p_values_over_time": p_values_over_time,
            "i2_over_time": i2_over_time,
            "effect_trend": effect_trend,
            "precision_trend": precision_trend,
            "first_significant_study": first_significant,
            "stability_study": stability_point,
            "reversals": reversals,
            "conclusion_stable": stability_point is not None and stability_point < len(cumulative) - 2
        }


def perform_cumulative_meta_analysis(
    data: pd.DataFrame,
    effect_col: str = "effect",
    variance_col: str = "variance",
    study_col: str = "study_id",
    date_col: str = "publication_date"
) -> List[CumulativeResult]:
    """
    Convenience function for cumulative meta-analysis.

    :param data: DataFrame with study data
    :param effect_col: Column with effect sizes
    :param variance_col: Column with variances
    :param study_col: Column with study IDs
    :param date_col: Column with publication dates
    :return: List of cumulative results
    """
    analyzer = CumulativeMetaAnalyzer()

    return analyzer.analyze(
        effects=data[effect_col].values,
        variances=data[variance_col].values,
        study_ids=data[study_col].tolist(),
        publication_dates=data[date_col].tolist() if date_col in data.columns else None
    )


if __name__ == "__main__":
    print("Cumulative Meta-Analysis and Scan Statistics module loaded")
    print("Features:")
    print("  - Cumulative meta-analysis over time")
    print("  - Stability detection")
    print("  - Effect reversal detection")
    print("  - Temporal scan statistics")
    print("  - Trial Sequential Analysis (TSA)")
    print("  - Evidence evolution tracking")
