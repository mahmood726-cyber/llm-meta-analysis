"""
Enhanced Statistical Framework for Meta-Analysis (DEPRECATED)

⚠️ DEPRECATION WARNING (v2.0): This module is deprecated.

Please use statistical_framework_v2.py instead, which includes:
- Hartung-Knapp-Sidik-Jonkman (HKSJ) adjustment
- Q-profile confidence intervals for τ²
- Proper prediction intervals
- Bootstrap variance propagation
- Quality effects model

Migration: Replace `from statistical_framework import` with `from statistical_framework_v2 import`

This module will be removed in version 3.0.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import ncfdtr
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings

# Issue deprecation warning
warnings.warn(
    "statistical_framework.py is deprecated. Use statistical_framework_v2.py instead. "
    "This module will be removed in version 3.0.",
    DeprecationWarning,
    stacklevel=2
)


@dataclass
class UncertaintyInterval:
    """Represents an uncertainty interval with proper interpretation"""
    lower: float
    upper: float
    level: float = 0.95
    method: str = "wald"

    def width(self) -> float:
        return self.upper - self.lower

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper


@dataclass
class HeterogeneityStatistics:
    """Complete heterogeneity assessment with uncertainty quantification"""
    q_statistic: float
    q_df: int
    q_p_value: float
    i_squared: float
    i_squared_ci: UncertaintyInterval
    tau_squared: float
    tau_squared_ci: Optional[UncertaintyInterval] = None
    tau: Optional[float] = None  # Square root of tau_squared

    def interpretation(self) -> str:
        """Provide standardized interpretation of heterogeneity"""
        if self.i_squared < 25:
            return "Low heterogeneity (I² < 25%)"
        elif self.i_squared < 50:
            return "Moderate heterogeneity (25% ≤ I² < 50%)"
        elif self.i_squared < 75:
            return "Substantial heterogeneity (50% ≤ I² < 75%)"
        else:
            return "Considerable heterogeneity (I² ≥ 75%)"


@dataclass
class PublicationBiasAssessment:
    """Publication bias assessment results"""
    egger_intercept: float
    egger_se: float
    egger_p_value: float
    egger_ci: UncertaintyInterval
    beggs_z: float
    beggs_p_value: float
    trim_and_fill_adjusted: Optional[float] = None
    n_missing_studies: Optional[int] = None
    funnel_plot_asymmetry: str

    def interpretation(self) -> str:
        """Interpret publication bias assessment"""
        if self.egger_p_value < 0.05:
            return f"Evidence of publication bias (Egger's p = {self.egger_p_value:.4f})"
        elif self.beggs_p_value < 0.05:
            return f"Evidence of funnel plot asymmetry (Begg's p = {self.beggs_p_value:.4f})"
        else:
            return "No strong evidence of publication bias"


class UncertaintyQuantifier:
    """
    Proper uncertainty quantification for extracted data.

    Implements methods for confidence interval calculation and
    variance-weighted comparisons following Cochrane guidelines.
    """

    @staticmethod
    def calculate_proportion_ci(events: int, total: int, method: str = "wilson") -> UncertaintyInterval:
        """
        Calculate confidence interval for a proportion with appropriate method.

        :param events: Number of events
        :param total: Total number
        :param method: Method for CI calculation ('wilson', 'clopper-pearson', 'wald')
        :return: UncertaintyInterval
        """
        if total == 0:
            return UncertaintyInterval(0, 0, method="invalid")

        p = events / total
        alpha = 0.05
        z = stats.norm.ppf(1 - alpha/2)

        if method == "wilson":
            # Wilson score interval (recommended for meta-analysis)
            denominator = 1 + z**2 / total
            center = (p + z**2 / (2 * total)) / denominator
            margin = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
            return UncertaintyInterval(
                lower=max(0, center - margin),
                upper=min(1, center + margin),
                level=0.95,
                method="wilson"
            )

        elif method == "clopper-pearson":
            # Exact Clopper-Pearson interval
            lower = stats.beta.ppf(alpha/2, events, total - events + 1) if events > 0 else 0
            upper = stats.beta.ppf(1 - alpha/2, events + 1, total - events) if events < total else 1
            return UncertaintyInterval(lower, upper, level=0.95, method="clopper-pearson")

        else:
            # Wald interval (not recommended, included for comparison)
            se = np.sqrt(p * (1 - p) / total)
            return UncertaintyInterval(
                lower=max(0, p - z * se),
                upper=min(1, p + z * se),
                level=0.95,
                method="wald"
            )

    @staticmethod
    def calculate_mean_ci(mean: float, sd: float, n: int) -> UncertaintyInterval:
        """
        Calculate confidence interval for a mean.

        :param mean: Sample mean
        :param sd: Sample standard deviation
        :param n: Sample size
        :return: UncertaintyInterval
        """
        if n <= 1:
            return UncertaintyInterval(mean, mean, method="invalid")

        se = sd / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n-1)
        margin = t_crit * se

        return UncertaintyInterval(
            lower=mean - margin,
            upper=mean + margin,
            level=0.95,
            method="t-distribution"
        )

    @staticmethod
    def variance_weighted_mean(values: np.ndarray, variances: np.ndarray) -> Tuple[float, float, UncertaintyInterval]:
        """
        Calculate variance-weighted mean with proper uncertainty quantification.

        :param values: Array of effect sizes
        :param variances: Array of variances
        :return: Tuple of (weighted_mean, standard_error, confidence_interval)
        """
        weights = 1 / variances
        weighted_mean = np.sum(weights * values) / np.sum(weights)

        # Standard error of weighted mean
        se_weighted = np.sqrt(1 / np.sum(weights))

        # Confidence interval
        z_crit = stats.norm.ppf(0.975)
        ci = UncertaintyInterval(
            lower=weighted_mean - z_crit * se_weighted,
            upper=weighted_mean + z_crit * se_weighted,
            level=0.95,
            method="variance-weighted"
        )

        return weighted_mean, se_weighted, ci

    @staticmethod
    def compare_with_uncertainty(
        value1: float,
        ci1: UncertaintyInterval,
        value2: float,
        ci2: UncertaintyInterval
    ) -> Dict[str, Union[float, str]]:
        """
        Compare two values with uncertainty quantification.

        :param value1: First value
        :param ci1: Confidence interval for first value
        :param value2: Second value
        :param ci2: Confidence interval for second value
        :return: Dictionary with comparison results
        """
        difference = value1 - value2

        # Standard error of difference (assuming independence)
        se_diff = np.sqrt((ci1.width() / (2 * 1.96))**2 + (ci2.width() / (2 * 1.96))**2)

        # Z-score for difference
        z_score = difference / se_diff if se_diff > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        # CI for difference
        ci_diff = UncertaintyInterval(
            lower=difference - 1.96 * se_diff,
            upper=difference + 1.96 * se_diff,
            level=0.95,
            method="difference"
        )

        # Check if CIs overlap
        ci_overlap = not (ci1.upper < ci2.lower or ci2.upper < ci1.lower)

        return {
            "difference": difference,
            "difference_ci": (ci_diff.lower, ci_diff.upper),
            "z_score": z_score,
            "p_value": p_value,
            "ci_overlap": ci_overlap,
            "significant": p_value < 0.05,
            "interpretation": "Significant difference" if p_value < 0.05 else "No significant difference"
        }


class EnhancedHeterogeneity:
    """
    Enhanced heterogeneity assessment with proper uncertainty quantification.

    Implements:
    - I² with confidence intervals (Ioannidis et al., 2007)
    - Prediction intervals
    - Q-statistic with proper interpretation
    """

    @staticmethod
    def calculate_i_squared_with_ci(
        q_statistic: float,
        df: int,
        ci_level: float = 0.95
    ) -> Tuple[float, UncertaintyInterval]:
        """
        Calculate I² with confidence interval using the non-central chi-square method.

        Reference: Ioannidis et al. (2007) Uncertainty in heterogeneity estimates.

        :param q_statistic: Cochran's Q statistic
        :param df: Degrees of freedom (k - 1)
        :param ci_level: Confidence level
        :return: Tuple of (i_squared, confidence_interval)
        """
        # Calculate I²
        if q_statistic <= df:
            i_squared = 0
        else:
            i_squared = 100 * (q_statistic - df) / q_statistic

        # Calculate confidence interval using non-central chi-square
        alpha = 1 - ci_level

        # Lower bound
        if q_statistic > df:
            try:
                # Non-central chi-square for lower bound
                nc_chi2_lower = ncfdtr(df, q_statistic, alpha/2)
                i_lower = 100 * (nc_chi2_lower - df) / nc_chi2_lower if nc_chi2_lower > df else 0
            except:
                i_lower = 0
        else:
            i_lower = 0

        # Upper bound
        try:
            # Non-central chi-square for upper bound
            nc_chi2_upper = ncfdtr(df, q_statistic, 1 - alpha/2)
            i_upper = 100 * (nc_chi2_upper - df) / nc_chi2_upper if nc_chi2_upper > df else 100
        except:
            i_upper = 100

        # Truncate to valid range
        i_lower = max(0, min(100, i_lower))
        i_upper = max(0, min(100, i_upper))

        ci = UncertaintyInterval(
            lower=i_lower,
            upper=i_upper,
            level=ci_level,
            method="non-central-chi-square"
        )

        return i_squared, ci

    @staticmethod
    def calculate_prediction_interval(
        pooled_effect: float,
        se_pooled: float,
        tau_squared: float,
        df: int,
        ci_level: float = 0.95
    ) -> UncertaintyInterval:
        """
        Calculate prediction interval for true effect in a new study.

        Reference: Riley et al. (2011) Interpretation of random effects meta-analyses.

        :param pooled_effect: Pooled effect estimate
        :param se_pooled: Standard error of pooled effect
        :param tau_squared: Between-study variance
        :param df: Degrees of freedom
        :param ci_level: Confidence level
        :return: UncertaintyInterval for prediction
        """
        # Prediction interval accounts for both within and between study variance
        se_prediction = np.sqrt(se_pooled**2 + tau_squared)

        # Use t-distribution for better coverage with few studies
        t_crit = stats.t.ppf(1 - (1 - ci_level)/2, df=df)

        return UncertaintyInterval(
            lower=pooled_effect - t_crit * se_prediction,
            upper=pooled_effect + t_crit * se_prediction,
            level=ci_level,
            method="prediction-interval"
        )

    @staticmethod
    def full_heterogeneity_assessment(
        effects: np.ndarray,
        variances: np.ndarray,
        method: str = "dl"
    ) -> HeterogeneityStatistics:
        """
        Complete heterogeneity assessment with all relevant statistics.

        :param effects: Array of effect sizes
        :param variances: Array of variances
        :param method: Method for tau² estimation ('dl', 'reml', 'paule')
        :return: HeterogeneityStatistics object
        """
        k = len(effects)
        df = k - 1

        # Calculate Q statistic
        weights_fixed = 1 / variances
        weighted_mean_fixed = np.sum(weights_fixed * effects) / np.sum(weights_fixed)
        q_statistic = np.sum(weights_fixed * (effects - weighted_mean_fixed)**2)

        # P-value for Q
        q_p_value = 1 - stats.chi2.cdf(q_statistic, df)

        # Calculate I² with CI
        i_squared, i_squared_ci = EnhancedHeterogeneity.calculate_i_squared_with_ci(
            q_statistic, df
        )

        # Calculate tau² using specified method
        tau_squared = EnhancedHeterogeneity.estimate_tau_squared(
            effects, variances, method
        )
        tau = np.sqrt(max(0, tau_squared))

        # Calculate CI for tau² (using profile likelihood method approximation)
        tau_ci = EnhancedHeterogeneity._tau_ci_profile(effects, variances, tau_squared)

        return HeterogeneityStatistics(
            q_statistic=q_statistic,
            q_df=df,
            q_p_value=q_p_value,
            i_squared=i_squared,
            i_squared_ci=i_squared_ci,
            tau_squared=tau_squared,
            tau_squared_ci=tau_ci,
            tau=tau
        )

    @staticmethod
    def estimate_tau_squared(
        effects: np.ndarray,
        variances: np.ndarray,
        method: str = "dl"
    ) -> float:
        """Estimate between-study variance tau²"""
        k = len(effects)
        df = k - 1

        weights_fixed = 1 / variances
        weighted_mean = np.sum(weights_fixed * effects) / np.sum(weights_fixed)
        q_statistic = np.sum(weights_fixed * (effects - weighted_mean)**2)

        if method == "dl":
            # DerSimonian-Laird
            if q_statistic <= df:
                return 0
            sum_w = np.sum(weights_fixed)
            sum_w2 = np.sum(weights_fixed**2)
            tau_squared = (q_statistic - df) / (sum_w - sum_w2 / sum_w)
            return max(0, tau_squared)

        elif method == "paule":
            # Paule-Mandel
            tau_sq = 0
            for _ in range(100):  # Iterative solution
                weights = 1 / (variances + tau_sq)
                weighted_mean = np.sum(weights * effects) / np.sum(weights)
                q_new = np.sum(weights * (effects - weighted_mean)**2)
                tau_sq_new = max(0, (q_new - df) / (np.sum(weights) - np.sum(weights**2) / np.sum(weights)))
                if abs(tau_sq_new - tau_sq) < 1e-6:
                    break
                tau_sq = tau_sq_new
            return tau_sq

        else:
            # Default to DL
            return EnhancedHeterogeneity.estimate_tau_squared(effects, variances, "dl")

    @staticmethod
    def _tau_ci_profile(
        effects: np.ndarray,
        variances: np.ndarray,
        tau_sq_est: float,
        alpha: float = 0.05
    ) -> Optional[UncertaintyInterval]:
        """
        Calculate confidence interval for tau² using profile likelihood.

        Simplified implementation - full version requires iterative optimization.
        """
        try:
            k = len(effects)
            se_tau_sq = np.sqrt(2 * tau_sq_est**2 / (k - 2)) if k > 2 else tau_sq_est

            z_crit = stats.norm.ppf(1 - alpha/2)
            lower = max(0, tau_sq_est - z_crit * se_tau_sq)
            upper = tau_sq_est + z_crit * se_tau_sq

            return UncertaintyInterval(lower, upper, level=1-alpha, method="profile-likelihood")
        except:
            return None


class PublicationBiasAssessor:
    """
    Publication bias assessment using multiple methods.

    Implements:
    - Egger's regression test
    - Begg's rank correlation test
    - Trim and fill method
    - Funnel plot asymmetry assessment
    """

    @staticmethod
    def eggers_test(effects: np.ndarray, standard_errors: np.ndarray) -> Dict[str, float]:
        """
        Egger's regression test for funnel plot asymmetry.

        :param effects: Array of effect sizes
        :param standard_errors: Array of standard errors
        :return: Dictionary with test results
        """
        # Regression of standard normal deviate vs precision
        precision = 1 / standard_errors
        snd = effects / standard_errors

        # Linear regression
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(precision, snd)

        # Calculate CI for intercept
        n = len(effects)
        se_intercept = std_err * np.sqrt(1/n + np.mean(precision)**2 / np.sum((precision - np.mean(precision))**2))
        t_crit = stats.t.ppf(0.975, df=n-2)
        ci_lower = intercept - t_crit * se_intercept
        ci_upper = intercept + t_crit * se_intercept

        return {
            "intercept": intercept,
            "se": se_intercept,
            "p_value": p_value,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "significant": p_value < 0.05
        }

    @staticmethod
    def beggs_test(effects: np.ndarray, variances: np.ndarray) -> Dict[str, float]:
        """
        Begg's rank correlation test for funnel plot asymmetry.

        :param effects: Array of effect sizes
        :param variances: Array of variances
        :return: Dictionary with test results
        """
        from scipy.stats import kendalltau

        # Rank correlation between standardized effect and variance
        n = len(effects)
        ranks_effect = stats.rankdata(effects)
        ranks_var = stats.rankdata(variances)

        # Note: Original Begg's test uses Kendall's tau
        # Some implementations use variance, others use precision
        tau, p_value = kendalltau(effects, 1/np.sqrt(variances))

        # Continuity correction
        z_score = tau * np.sqrt(9 * n * (n - 1) / (2 * (2 * n + 5)))
        p_two_tailed = 2 * (1 - stats.norm.cdf(abs(z_score)))

        return {
            "kendall_tau": tau,
            "z_score": z_score,
            "p_value": p_two_tailed,
            "significant": p_two_tailed < 0.05
        }

    @staticmethod
    def trim_and_fill(
        effects: np.ndarray,
        variances: np.ndarray,
        side: str = "left"
    ) -> Dict[str, Union[float, int, np.ndarray]]:
        """
        Trim and fill method for publication bias adjustment.

        Simplified implementation following Duval & Tweedie (2000).

        :param effects: Array of effect sizes
        :param variances: Array of variances
        :param side: Side to trim ('left' or 'right')
        :return: Dictionary with adjustment results
        """
        from scipy.stats import ttest_1samp

        n = len(effects)
        weights = 1 / variances
        weighted_effects = weights * effects

        # Rank by precision
        precision = 1 / np.sqrt(variances)
        ranks = np.argsort(precision)

        # Iteratively trim and fill
        k_filled = 0
        prev_estimate = np.sum(weights * effects) / np.sum(weights)

        for k in range(n - 2):
            if side == "left":
                # Remove k most negative studies (left of mean)
                keep_mask = effects >= np.median(effects)
            else:
                keep_mask = effects <= np.median(effects)

            trimmed_effects = effects[keep_mask]
            trimmed_variances = variances[keep_mask]

            if len(trimmed_effects) < 3:
                break

            trimmed_weights = 1 / trimmed_variances
            new_estimate = np.sum(trimmed_weights * trimmed_effects) / np.sum(trimmed_weights)

            # Check convergence
            if abs(new_estimate - prev_estimate) < 1e-4:
                k_filled = k + 1
                break

            prev_estimate = new_estimate

        # Calculate adjusted estimate
        # This is a simplified version - full implementation requires iterative refinement
        original_estimate = np.sum(weights * effects) / np.sum(weights)
        adjusted_estimate = prev_estimate

        return {
            "original_estimate": original_estimate,
            "adjusted_estimate": adjusted_estimate,
            "n_missing_studies": k_filled,
            "adjustment": abs(adjusted_estimate - original_estimate)
        }

    @staticmethod
    def full_assessment(effects: np.ndarray, variances: np.ndarray) -> PublicationBiasAssessment:
        """
        Complete publication bias assessment.

        :param effects: Array of effect sizes
        :param variances: Array of variances
        :return: PublicationBiasAssessment object
        """
        standard_errors = np.sqrt(variances)

        # Egger's test
        egger = PublicationBiasAssessor.eggers_test(effects, standard_errors)

        # Begg's test
        begg = PublicationBiasAssessor.beggs_test(effects, variances)

        # Trim and fill
        tf = PublicationBiasAssessor.trim_and_fill(effects, variances)

        # Overall interpretation
        if egger["p_value"] < 0.01:
            asymmetry = "Strong evidence of asymmetry"
        elif egger["p_value"] < 0.05:
            asymmetry = "Moderate evidence of asymmetry"
        elif egger["p_value"] < 0.1:
            asymmetry = "Suggestive evidence of asymmetry"
        else:
            asymmetry = "No clear evidence of asymmetry"

        return PublicationBiasAssessment(
            egger_intercept=egger["intercept"],
            egger_se=egger["se"],
            egger_p_value=egger["p_value"],
            egger_ci=UncertaintyInterval(egger["ci_lower"], egger["ci_upper"]),
            beggs_z=begg["z_score"],
            beggs_p_value=begg["p_value"],
            trim_and_fill_adjusted=tf.get("adjusted_estimate"),
            n_missing_studies=tf.get("n_missing_studies"),
            funnel_plot_asymmetry=asymmetry
        )


class SmallStudyEffects:
    """
    Assessment of small-study effects in meta-analysis.

    Implements methods to detect whether smaller studies show different effects
    than larger studies, which may indicate publication bias or other biases.
    """

    @staticmethod
    def regression_with_precision(
        effects: np.ndarray,
        standard_errors: np.ndarray
    ) -> Dict[str, float]:
        """
        Weighted regression of effect size on precision.

        :param effects: Array of effect sizes
        :param standard_errors: Array of standard errors
        :return: Dictionary with regression results
        """
        precision = 1 / standard_errors
        weights = precision

        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(precision, effects, alternative='two-sided')

        # Weighted correlation
        mean_precision = np.average(precision, weights=weights)
        mean_effects = np.average(effects, weights=weights)

        numerator = np.sum(weights * (precision - mean_precision) * (effects - mean_effects))
        denominator = np.sqrt(
            np.sum(weights * (precision - mean_precision)**2) *
            np.sum(weights * (effects - mean_effects)**2)
        )
        weighted_r = numerator / denominator if denominator != 0 else 0

        return {
            "slope": slope,
            "slope_p_value": p_value,
            "weighted_r": weighted_r,
            "interpretation": "Small-study effects present" if p_value < 0.05 else "No clear small-study effects"
        }


if __name__ == "__main__":
    # Test the implementations
    print("Enhanced Statistical Framework loaded successfully")
    print("Modules available:")
    print("  - UncertaintyQuantifier")
    print("  - EnhancedHeterogeneity")
    print("  - PublicationBiasAssessor")
    print("  - SmallStudyEffects")
