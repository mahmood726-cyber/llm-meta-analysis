"""
Power Analysis for Meta-Analysis

Implements methods for:
1. Power analysis for detecting effects in meta-analysis
2. Sample size calculations for new studies
3. Detecting funnel plot asymmetry (power of tests)
4. Optimal information size calculation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize_scalar


@dataclass
class MetaAnalysisPower:
    """Results from power analysis"""
    power: float
    alpha: float
    effect_size: float
    n_studies: int
    tau_squared: float
    within_study_variance: float
    optimal_n_studies: Optional[int] = None
    optimal_sample_size: Optional[int] = None


@dataclass
class SampleSizeResult:
    """Results from sample size calculation"""
    required_n_per_group: int
    required_total_n: int
    detectable_effect: float
    power: float
    alpha: float
    assumptions: Dict[str, float]


class MetaAnalysisPowerCalculator:
    """
    Power analysis for meta-analysis.

    Backward-compatible aliases (referenced by
    evaluation/integrated_meta_analysis.py): PowerAnalysis and
    SampleSizeCalculator. Both point at this class; splitting them is
    future work if the APIs need to diverge.

    References:
    - Jackson et al. (2018) Power and sample size for meta-analysis
    - Hedges & Pigott (2001) The power of statistical tests in meta-analysis
    - Borenstein et al. (2011) Introduction to Meta-Analysis
    """

    def __init__(self):
        self.calculated_power = []

    def power_for_meta_analysis(
        self,
        effect_size: float,
        n_studies: int,
        tau_squared: float,
        within_study_variance: float,
        alpha: float = 0.05,
        alternative: str = "two-sided"
    ) -> MetaAnalysisPower:
        """
        Calculate the power of a meta-analysis to detect a given effect.

        :param effect_size: True effect size (e.g., log odds ratio, SMD)
        :param n_studies: Number of studies
        :param tau_squared: Between-study variance
        :param within_study_variance: Average within-study variance
        :param alpha: Significance level
        :param alternative: 'two-sided' or 'one-sided'
        :return: MetaAnalysisPower object
        """
        # Total variance of effect estimate
        total_variance = tau_squared + within_study_variance / n_studies

        # Standard error
        se = np.sqrt(total_variance)

        # Critical value
        if alternative == "two-sided":
            z_crit = stats.norm.ppf(1 - alpha/2)
        else:
            z_crit = stats.norm.ppf(1 - alpha)

        # Z-score for the effect
        z_score = abs(effect_size) / se

        # Power = P(|Z| > z_crit | alternative true)
        # = P(Z > z_crit - z_score) + P(Z < -z_crit - z_score)
        power = 1 - stats.norm.cdf(z_crit - z_score) + stats.norm.cdf(-z_crit - z_score)
        power = max(0, min(1, power))

        # Calculate optimal number of studies
        optimal_n = self._calculate_optimal_n_studies(
            effect_size, tau_squared, within_study_variance, alpha, target_power=0.8
        )

        return MetaAnalysisPower(
            power=power,
            alpha=alpha,
            effect_size=effect_size,
            n_studies=n_studies,
            tau_squared=tau_squared,
            within_study_variance=within_study_variance,
            optimal_n_studies=optimal_n
        )

    def power_for_detecting_heterogeneity(
        self,
        tau_squared: float,
        n_studies: int,
        within_study_variance: float,
        alpha: float = 0.05
    ) -> float:
        """
        Calculate power to detect heterogeneity (tau² > 0).

        :param tau_squared: True between-study variance
        :param n_studies: Number of studies
        :param within_study_variance: Average within-study variance
        :param alpha: Significance level
        :return: Power to detect heterogeneity
        """
        # Q statistic follows chi-square with df = n-1 under H0: tau² = 0
        # Under H1: tau² > 0, Q follows non-central chi-square

        df = n_studies - 1

        # Non-centrality parameter
        # This is an approximation
        ncp = (n_studies - 1) * (1 + tau_squared / within_study_variance)

        # Power = P(Q > chi2_crit | H1)
        chi2_crit = stats.chi2.ppf(1 - alpha, df)
        power = 1 - stats.ncx2.cdf(chi2_crit, df, ncp)

        return power

    def calculate_optimal_information_size(
        self,
        effect_size: float,
        within_study_variance: float,
        tau_squared: float = 0,
        alpha: float = 0.05,
        power: float = 0.8,
        ICC: Optional[float] = None
    ) -> int:
        """
        Calculate Optimal Information Size (OIS) for meta-analysis.

        The OIS is the total number of participants needed to reliably detect
        an effect in a meta-analysis.

        Reference: Pogue & Yusuf (1998)

        :param effect_size: Minimal clinically important effect
        :param within_study_variance: Average within-study variance
        :param tau_squared: Between-study variance
        :param alpha: Significance level
        :param power: Desired power
        :param ICC: Intra-class correlation (for cluster designs)
        :return: Required total sample size
        """
        # For cluster designs, adjust for design effect
        if ICC is not None:
            # Design effect = 1 + (m - 1) * ICC where m is cluster size
            # This is a simplification
            design_effect = 1 + ICC  # Approximate
            within_study_variance *= design_effect

        # Standard two-sample size calculation
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)

        # For random-effects, need to account for tau²
        # Total variance per comparison = 2 * within_study_variance + tau²
        total_variance = 2 * within_study_variance + tau_squared

        # Required sample per group
        n_per_group = 2 * total_variance * (z_alpha + z_beta)**2 / effect_size**2
        n_per_group = np.ceil(n_per_group)

        return int(2 * n_per_group)

    def sample_size_for_new_study(
        self,
        target_effect: float,
        control_rate: Optional[float] = None,
        control_mean: Optional[float] = None,
        control_sd: Optional[float] = None,
        alpha: float = 0.05,
        power: float = 0.8,
        outcome_type: str = "binary",
        allocation_ratio: float = 1.0
    ) -> SampleSizeResult:
        """
        Calculate sample size for a new study.

        :param target_effect: Target effect size (log OR for binary, SMD for continuous)
        :param control_rate: Control event rate (for binary outcomes)
        :param control_mean: Control mean (for continuous outcomes)
        :param control_sd: Control SD (for continuous outcomes)
        :param alpha: Significance level
        :param power: Desired power
        :param outcome_type: 'binary' or 'continuous'
        :param allocation_ratio: Ratio of intervention to control participants
        :return: SampleSizeResult
        """
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)

        if outcome_type == "binary":
            if control_rate is None:
                raise ValueError("control_rate required for binary outcomes")

            # Convert log OR to probabilities
            # OR = exp(target_effect)
            # Intervention rate can be derived from OR and control rate
            or_val = np.exp(target_effect)

            # Assume equal allocation
            p1 = control_rate  # Control
            p2 = p1 * or_val / (1 + p1 * (or_val - 1))  # Intervention

            # Two-sample proportion test
            p_pooled = (p1 + p2) / 2

            n_per_group = (
                (z_alpha * np.sqrt(2 * p_pooled * (1 - p_pooled)) +
                 z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))**2
                / (p1 - p2)**2
            )

        else:  # continuous
            if control_mean is None or control_sd is None:
                raise ValueError("control_mean and control_sd required for continuous outcomes")

            # Convert SMD to actual effect size
            raw_effect = target_effect * control_sd

            # Two-sample t-test (using normal approximation)
            sd_pooled = control_sd  # Assuming equal SDs

            n_per_group = 2 * sd_pooled**2 * (z_alpha + z_beta)**2 / raw_effect**2

        n_per_group = np.ceil(n_per_group)

        # Adjust for allocation ratio
        if allocation_ratio != 1.0:
            n_intervention = n_per_group
            n_control = n_per_group / allocation_ratio
            n_per_group = max(n_intervention, n_control)

        return SampleSizeResult(
            required_n_per_group=int(n_per_group),
            required_total_n=int(2 * n_per_group),
            detectable_effect=target_effect,
            power=power,
            alpha=alpha,
            assumptions={
                "control_rate": control_rate,
                "control_mean": control_mean,
                "control_sd": control_sd,
                "allocation_ratio": allocation_ratio
            }
        )

    def power_of_publication_bias_tests(
        self,
        n_studies: int,
        true_effect: float,
        se_typical: float = 0.2
    ) -> Dict[str, float]:
        """
        Calculate power of publication bias tests.

        Reference: Sterne et al. (2011) Tests for funnel plot asymmetry

        :param n_studies: Number of studies
        :param true_effect: True effect size
        :param se_typical: Typical standard error of studies
        :return: Dictionary with power of different tests
        """
        # Power of Egger's test
        # This is an approximation - full power calculation is complex

        # Standard error of intercept
        se_intercept = se_typical * np.sqrt(1/n_studies)

        # Effect on intercept from publication bias
        # Assuming moderate bias: intercept shift of 1 SE
        bias_effect = se_intercept

        # Power to detect this bias
        z_score = bias_effect / se_intercept
        power_egger = 1 - stats.norm.cdf(1.96 - z_score)

        # Power of Begg's test (rank correlation)
        # Approximation using normal approximation to Kendall's tau
        power_begg = self._power_rank_correlation(n_studies, true_effect, se_typical)

        return {
            "egger_test": max(0, min(1, power_egger)),
            "beggs_test": max(0, min(1, power_begg)),
            "n_studies": n_studies
        }

    def _power_rank_correlation(
        self,
        n: int,
        true_effect: float,
        se: float,
        alpha: float = 0.05
    ) -> float:
        """
        Approximate power of rank correlation test (Begg's test).

        :param n: Number of studies
        :param true_effect: True effect size
        :param se: Typical standard error
        :param alpha: Significance level
        :return: Power
        """
        # This is a simplified approximation
        # Full derivation is complex

        # Expected correlation under bias
        # Assuming bias creates correlation of 0.3 between effect and SE
        expected_correlation = 0.3 if true_effect != 0 else 0

        # Standard error of Kendall's tau under H0
        se_tau = np.sqrt(2 * (2*n + 5) / (9*n * (n - 1)))

        # Critical value
        tau_crit = stats.norm.ppf(1 - alpha/2) * se_tau

        # Power
        z = (expected_correlation - tau_crit) / se_tau
        power = 1 - stats.norm.cdf(z)

        return power

    def _calculate_optimal_n_studies(
        self,
        effect_size: float,
        tau_squared: float,
        within_study_variance: float,
        alpha: float = 0.05,
        target_power: float = 0.8
    ) -> int:
        """
        Calculate optimal number of studies for desired power.

        :param effect_size: True effect size
        :param tau_squared: Between-study variance
        :param within_study_variance: Within-study variance
        :param alpha: Significance level
        :param target_power: Target power
        :return: Optimal number of studies
        """
        # Binary search for n_studies that achieves target power
        def objective(n):
            result = self.power_for_meta_analysis(
                effect_size, int(n), tau_squared, within_study_variance, alpha
            )
            return abs(result.power - target_power)

        # Search range: 2 to 100 studies
        result = minimize_scalar(
            objective,
            bounds=(2, 100),
            method='bounded'
        )

        optimal_n = int(np.ceil(result.x))

        return optimal_n

    def sensitivity_analysis_power(
        self,
        effect_sizes: np.ndarray,
        n_studies_range: np.ndarray,
        tau_squared: float,
        within_study_variance: float
    ) -> pd.DataFrame:
        """
        Create power analysis sensitivity table.

        :param effect_sizes: Array of effect sizes to test
        :param n_studies_range: Array of n_studies values to test
        :param tau_squared: Between-study variance
        :param within_study_variance: Within-study variance
        :return: DataFrame with power calculations
        """
        results = []

        for effect in effect_sizes:
            for n in n_studies_range:
                power_result = self.power_for_meta_analysis(
                    effect, n, tau_squared, within_study_variance
                )
                results.append({
                    'effect_size': effect,
                    'n_studies': n,
                    'power': power_result.power
                })

        return pd.DataFrame(results)

    def plot_power_curves(
        self,
        effect_sizes: np.ndarray,
        tau_squared: float,
        within_study_variance: float,
        n_studies_max: int = 20
    ) -> None:
        """
        Create power curves for different effect sizes.

        :param effect_sizes: Array of effect sizes
        :param tau_squared: Between-study variance
        :param within_study_variance: Within-study variance
        :param n_studies_max: Maximum number of studies to plot
        """
        import matplotlib.pyplot as plt

        n_range = np.arange(2, n_studies_max + 1)

        plt.figure(figsize=(10, 6))

        for effect in effect_sizes:
            powers = []
            for n in n_range:
                result = self.power_for_meta_analysis(
                    effect, n, tau_squared, within_study_variance
                )
                powers.append(result.power)

            plt.plot(n_range, powers, marker='o', label=f'Effect = {effect:.2f}')

        plt.axhline(y=0.8, color='r', linestyle='--', label='80% Power')
        plt.xlabel('Number of Studies')
        plt.ylabel('Power')
        plt.title('Power Curves for Meta-Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


def calculate_required_n_studies(
    min_detectable_effect: float,
    within_study_variance: float,
    tau_squared: float = 0,
    alpha: float = 0.05,
    target_power: float = 0.8
) -> int:
    """
    Convenience function to calculate required number of studies.

    :param min_detectable_effect: Minimum effect to detect
    :param within_study_variance: Average within-study variance
    :param tau_squared: Between-study variance
    :param alpha: Significance level
    :param target_power: Target power
    :return: Required number of studies
    """
    calculator = MetaAnalysisPowerCalculator()

    return calculator._calculate_optimal_n_studies(
        min_detectable_effect, tau_squared, within_study_variance, alpha, target_power
    )


if __name__ == "__main__":
    print("Power Analysis module loaded")
    print("Features:")
    print("  - Power for meta-analysis")
    print("  - Power for heterogeneity detection")
    print("  - Optimal information size")
    print("  - Sample size for new studies")
    print("  - Power of publication bias tests")


# ── Backward-compat aliases for integrated_meta_analysis.py ────────
# These are maintained by the Overmind smoke-repair pass (2026-04-14).
# If the API needs to diverge, split them into real classes.
PowerAnalysis = MetaAnalysisPowerCalculator
SampleSizeCalculator = MetaAnalysisPowerCalculator
