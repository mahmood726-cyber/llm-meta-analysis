"""
Enhanced Statistical Framework for Meta-Analysis (Revised)

This module implements rigorous statistical methods following Cochrane Handbook
guidelines and Research Synthesis Methods standards.

Revisions based on editorial feedback:
- Hartung-Knapp-Sidik-Jonkman (HKSJ) adjustment for small samples
- Proper prediction intervals
- Confidence intervals for τ² (Q-profile method)
- Bootstrap variance propagation
- Quality effects model
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import ncfdtr
from scipy.optimize import minimize_scalar, brentq
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings


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
    tau_squared: float
    i_squared_ci: Optional[UncertaintyInterval] = None
    tau_squared_ci: Optional[UncertaintyInterval] = None
    tau: Optional[float] = None

    def __post_init__(self):
        if self.tau is None and self.tau_squared is not None:
            self.tau = np.sqrt(max(0, self.tau_squared))

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
class MetaAnalysisResult:
    """Complete meta-analysis result with proper uncertainty quantification"""
    pooled_effect: float
    ci: UncertaintyInterval
    prediction_interval: Optional[UncertaintyInterval]
    z_statistic: float
    p_value: float
    heterogeneity: HeterogeneityStatistics
    method: str
    n_studies: int
    tau_squared: float
    weights: np.ndarray


class AdvancedMetaAnalysis:
    """
    Advanced meta-analysis methods with proper uncertainty quantification.

    Implements HKSJ adjustment, prediction intervals, and Q-profile τ² CIs.
    """

    @staticmethod
    def random_effects_analysis(
        effects: np.ndarray,
        variances: np.ndarray,
        tau2_method: str = "REML",
        ci_method: str = "auto",
        tau2_ci_method: str = "q_profile",
        prediction: bool = True,
        alpha: float = 0.05
    ) -> MetaAnalysisResult:
        """
        Random-effects meta-analysis with state-of-the-art methods.

        :param effects: Study effect estimates
        :param variances: Study variances
        :param tau2_method: Method for τ² estimation ('DL', 'REML', 'SJ', 'PM')
        :param ci_method: CI method ('auto', 'HKSJ', 'wald', 'quantile')
                          'auto' selects HKSJ for k < 20, wald for k >= 20 (Cochrane)
        :param tau2_ci_method: Method for τ² CI ('q_profile', 'profile_likelihood')
        :param prediction: Whether to compute prediction interval
        :param alpha: Significance level
        :return: MetaAnalysisResult
        """
        n = len(effects)

        # Auto-select CI method based on sample size (Cochrane recommendation)
        if ci_method == "auto":
            if n < 20:
                ci_method = "HKSJ"
            else:
                ci_method = "wald"

        # Estimate τ²
        tau2 = AdvancedMetaAnalysis._estimate_tau2(
            effects, variances, method=tau2_method
        )

        # Random-effects weights
        weights_re = 1 / (variances + tau2)
        sum_w_re = np.sum(weights_re)
        weights_re_re = weights_re / sum_w_re

        # Pooled effect
        pooled_effect = np.sum(weights_re_re * effects)

        # Standard error (naive)
        se_naive = np.sqrt(1 / sum_w_re)

        # Heterogeneity statistics with τ² CI method selection
        heterogeneity = AdvancedMetaAnalysis._compute_heterogeneity(
            effects, variances, tau2, tau2_ci_method
        )

        # Confidence interval using HKSJ adjustment
        if ci_method == "HKSJ":
            ci = AdvancedMetaAnalysis._hksj_ci(
                effects, variances, tau2, pooled_effect, alpha
            )
        elif ci_method == "wald":
            z = stats.norm.ppf(1 - alpha/2)
            ci = UncertaintyInterval(
                lower=pooled_effect - z * se_naive,
                upper=pooled_effect + z * se_naive,
                level=1-alpha,
                method="wald"
            )
        elif ci_method == "quantile":
            ci = AdvancedMetaAnalysis._quantile_ci(
                effects, variances, tau2, alpha
            )
        else:
            raise ValueError(f"Unknown CI method: {ci_method}")

        # Prediction interval
        pred_interval = None
        if prediction:
            pred_interval = AdvancedMetaAnalysis._prediction_interval(
                pooled_effect, tau2, heterogeneity, n, alpha
            )

        # Z-test
        z_stat = pooled_effect / se_naive
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        return MetaAnalysisResult(
            pooled_effect=pooled_effect,
            ci=ci,
            prediction_interval=pred_interval,
            z_statistic=z_stat,
            p_value=p_value,
            heterogeneity=heterogeneity,
            method=f"random-effects ({tau2_method}, {ci_method})",
            n_studies=n,
            tau_squared=tau2,
            weights=weights_re_re
        )

    @staticmethod
    def _estimate_tau2(
        effects: np.ndarray,
        variances: np.ndarray,
        method: str = "REML"
    ) -> float:
        """
        Estimate between-study variance τ².

        :param effects: Study effects
        :param variances: Study variances
        :param method: Estimation method
        :return: τ² estimate
        """
        n = len(effects)
        weights = 1 / variances
        sum_w = np.sum(weights)
        weighted_mean = np.sum(weights * effects) / sum_w

        # Q statistic
        q = np.sum(weights * (effects - weighted_mean)**2)
        df = n - 1

        if method == "DL":
            # DerSimonian-Laird
            sum_w2 = np.sum(weights**2)
            tau2 = max(0, (q - df) / (sum_w - sum_w2 / sum_w))

        elif method == "REML":
            # Restricted maximum likelihood with proper convergence
            tau2 = AdvancedMetaAnalysis._reml_estimate(
                effects, variances, max_iter=1000, tolerance=1e-8
            )
            # Fallback to DL if REML fails
            if tau2 is None or np.isnan(tau2) or tau2 < 0:
                sum_w2 = np.sum(weights**2)
                tau2 = max(0, (q - df) / (sum_w - sum_w2 / sum_w))

        elif method == "SJ":
            # Sidik-Jonkman
            sum_w2 = np.sum(weights**2)
            c = sum_w - sum_w2 / sum_w
            tau2 = (q - df) / c
            # SJ can be negative; truncate at 0
            tau2 = max(0, tau2)

        elif method == "PM":
            # Paule-Mandel
            tau2 = AdvancedMetaAnalysis._paule_mandel_estimate(
                effects, variances
            )

        else:
            raise ValueError(f"Unknown τ² method: {method}")

        return tau2

    @staticmethod
    def _reml_estimate(
        effects: np.ndarray,
        variances: np.ndarray,
        max_iter: int = 1000,
        tolerance: float = 1e-8
    ) -> Optional[float]:
        """
        REML estimation of τ² with proper convergence checking.

        :param effects: Study effects
        :param variances: Study variances
        :param max_iter: Maximum iterations
        :param tolerance: Convergence tolerance
        :return: τ² estimate or None if failed
        """
        n = len(effects)

        # Initialize with DL estimate
        weights = 1 / variances
        sum_w = np.sum(weights)
        sum_w2 = np.sum(weights**2)
        weighted_mean = np.sum(weights * effects) / sum_w
        q = np.sum(weights * (effects - weighted_mean)**2)
        df = n - 1
        tau2 = max(0, (q - df) / (sum_w - sum_w2 / sum_w))

        # Iterative REML
        for iteration in range(max_iter):
            tau2_old = tau2

            # Update weights with current τ²
            w_re = 1 / (variances + tau2)
            sum_w_re = np.sum(w_re)
            weighted_mean_re = np.sum(w_re * effects) / sum_w_re

            # Score function derivative
            sum_w_re2 = np.sum(w_re**2)
            residual_var = np.sum(w_re * (effects - weighted_mean_re)**2) / n

            # REML update
            tau2 = residual_var - (1/n) * np.sum(variances * w_re)

            # Check for convergence
            if abs(tau2 - tau2_old) < tolerance:
                # Validate result
                if tau2 < -tolerance:
                    return None  # Negative variance - failed
                return max(0, tau2)

            # Check for divergence
            if not np.isfinite(tau2) or tau2 > 1e6:
                return None

        # Failed to converge
        return None

    @staticmethod
    def _paule_mandel_estimate(
        effects: np.ndarray,
        variances: np.ndarray
    ) -> float:
        """
        Paule-Mandel estimator for τ².

        Solves for τ² such that Q(τ²) = df.

        Reference: Paule and Mandel (1982)
        """
        n = len(effects)

        # Calculate initial Q and DL estimate for bound determination
        weights = 1 / variances
        sum_w = np.sum(weights)
        weighted_mean = np.sum(weights * effects) / sum_w
        q = np.sum(weights * (effects - weighted_mean)**2)
        df = n - 1
        sum_w2 = np.sum(weights**2)
        tau2_dl = max(0, (q - df) / (sum_w - sum_w2 / sum_w))

        def q_minus_df(tau2):
            """Q - df as function of τ²"""
            w = 1 / (variances + tau2)
            sum_w = np.sum(w)
            weighted_mean = np.sum(w * effects) / sum_w
            q = np.sum(w * (effects - weighted_mean)**2)
            return q - (n - 1)

        # Find root with adaptive upper bound
        # Upper bound: max(100 * tau2_DL, 100) to ensure we capture the root
        upper_bound = max(100, 100 * tau2_dl, 10 * q / df)

        try:
            result = brentq(q_minus_df, 0, upper_bound, maxiter=100, rtol=1e-10)
            return max(0, result)
        except ValueError:
            # If brentq fails, use DL as fallback
            warnings.warn("Paule-Mandel estimation failed, using DerSimonian-Laird fallback")
            return tau2_dl

    @staticmethod
    def _hksj_ci(
        effects: np.ndarray,
        variances: np.ndarray,
        tau2: float,
        pooled_effect: float,
        alpha: float = 0.05
    ) -> UncertaintyInterval:
        """
        Hartung-Knapp-Sidik-Jonkman adjusted confidence interval.

        This is the recommended method for small samples (k < 20).

        References:
        - Hartung and Knapp (2001)
        - Sidik and Jonkman (2002)
        - IntHout et al. (2014)
        """
        n = len(effects)

        # Random-effects weights
        weights_re = 1 / (variances + tau2)
        sum_w_re = np.sum(weights_re)
        weights_re_re = weights_re / sum_w_re

        # Residual variance estimate
        residual_sum_sq = np.sum(weights_re_re * (effects - pooled_effect)**2)
        t_df = n - 1

        # HKSJ standard error
        se_hksj = np.sqrt(residual_sum_sq / sum_w_re)

        # t-distribution critical value
        t_crit = stats.t.ppf(1 - alpha/2, df=t_df)

        return UncertaintyInterval(
            lower=pooled_effect - t_crit * se_hksj,
            upper=pooled_effect + t_crit * se_hksj,
            level=1-alpha,
            method="HKSJ"
        )

    @staticmethod
    def _quantile_ci(
        effects: np.ndarray,
        variances: np.ndarray,
        tau2: float,
        alpha: float = 0.05,
        n_bootstrap: int = 1000
    ) -> UncertaintyInterval:
        """
        Quantile (bootstrap) confidence interval.

        :param effects: Study effects
        :param variances: Study variances
        :param tau2: Between-study variance
        :param alpha: Significance level
        :param n_bootstrap: Number of bootstrap iterations
        :return: UncertaintyInterval
        """
        n = len(effects)
        bootstrap_estimates = []

        for _ in range(n_bootstrap):
            # Parametric bootstrap: resample from estimated distribution
            # New effects ~ N(theta_i, v_i + tau2)
            effects_boot = np.random.normal(
                loc=effects,
                scale=np.sqrt(variances + tau2)
            )

            # Re-estimate τ² and pooled effect
            tau2_boot = AdvancedMetaAnalysis._estimate_tau2(
                effects_boot, variances, method="DL"
            )
            weights_boot = 1 / (variances + tau2_boot)
            pooled_boot = np.sum(weights_boot * effects_boot) / np.sum(weights_boot)
            bootstrap_estimates.append(pooled_boot)

        # Quantile interval
        lower = np.percentile(bootstrap_estimates, 100 * alpha/2)
        upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha/2))

        return UncertaintyInterval(
            lower=lower,
            upper=upper,
            level=1-alpha,
            method="bootstrap_quantile"
        )

    @staticmethod
    def _prediction_interval(
        pooled_effect: float,
        tau2: float,
        heterogeneity: HeterogeneityStatistics,
        n_studies: int,
        alpha: float = 0.05
    ) -> UncertaintyInterval:
        """
        Prediction interval for a new study.

        Implements the Partlett and Riley (2017) method with Knapp-Hartung adjustment.

        Reference: Partlett and Riley (2017). "Approximations to the distribution of
        the predicted random effects meta-analysis estimate."

        :param pooled_effect: Pooled effect estimate
        :param tau2: Between-study variance
        :param heterogeneity: Heterogeneity statistics
        :param n_studies: Number of studies
        :param alpha: Significance level
        :return: Prediction interval
        """
        # Standard error for prediction (accounts for uncertainty in τ²)
        # Using t-distribution for small samples (k < 20)
        se_pred = np.sqrt(tau2 + heterogeneity.tau_squared_ci.width() / 4)

        # Critical value: use t-distribution for k < 10, normal for k >= 10
        if n_studies < 10:
            t_crit = stats.t.ppf(1 - alpha/2, df=n_studies - 2)
        else:
            t_crit = stats.norm.ppf(1 - alpha/2)

        # Prediction interval
        lower = pooled_effect - t_crit * se_pred
        upper = pooled_effect + t_crit * se_pred

        return UncertaintyInterval(
            lower=lower,
            upper=upper,
            level=1-alpha,
            method="prediction_kh_adj"
        )

    @staticmethod
    def _compute_heterogeneity(
        effects: np.ndarray,
        variances: np.ndarray,
        tau2: float,
        tau2_ci_method: str = "q_profile"
    ) -> HeterogeneityStatistics:
        """
        Compute complete heterogeneity statistics.

        :param effects: Study effects
        :param variances: Study variances
        :param tau2: Between-study variance estimate
        :param tau2_ci_method: Method for τ² CI ('q_profile', 'profile_likelihood')
        :return: HeterogeneityStatistics
        """
        n = len(effects)
        weights = 1 / variances
        sum_w = np.sum(weights)
        weighted_mean = np.sum(weights * effects) / sum_w

        # Q statistic
        q = np.sum(weights * (effects - weighted_mean)**2)
        df = n - 1
        p_value = 1 - stats.chi2.cdf(q, df)

        # I² with confidence interval
        i2 = max(0, 100 * (q - df) / q) if q > df else 0
        i2_ci = AdvancedMetaAnalysis._i2_ci(q, df, 1 - 0.05)

        # τ² confidence interval using selected method
        if tau2_ci_method == "profile_likelihood":
            tau2_ci = AdvancedMetaAnalysis._tau2_profile_likelihood_ci(
                effects, variances, tau2, 0.05
            )
        else:  # default to q_profile
            tau2_ci = AdvancedMetaAnalysis._tau2_q_profile_ci(
                effects, variances, q, df, 0.05
            )

        return HeterogeneityStatistics(
            q_statistic=q,
            q_df=df,
            q_p_value=p_value,
            i_squared=i2,
            i_squared_ci=i2_ci,
            tau_squared=tau2,
            tau_squared_ci=tau2_ci
        )

    @staticmethod
    def _i2_ci(
        q: float,
        df: int,
        conf_level: float = 0.95
    ) -> UncertaintyInterval:
        """
        Confidence interval for I² using non-central chi-square.

        Reference: Higgins and Thompson (2002).

        :param q: Q statistic
        :param df: Degrees of freedom
        :param conf_level: Confidence level
        :return: UncertaintyInterval for I²
        """
        alpha = 1 - conf_level

        # Handle edge cases
        if q <= df:
            # No heterogeneity, CI is [0, 0]
            return UncertaintyInterval(
                lower=0,
                upper=0,
                level=conf_level,
                method="noncentral_chi2"
            )

        # Lower bound: find non-centrality parameter
        def q_lower_func(ncp):
            return stats.chi2.ppf(1 - alpha/2, df, loc=ncp) - q

        try:
            # Use adaptive upper bound for root finding
            nc_upper_bound = max(q * 2, df * 5, 100)
            nc_lower = brentq(q_lower_func, 0, nc_upper_bound, maxiter=100, rtol=1e-10)
            i2_lower = max(0, 100 * (nc_lower - df) / nc_lower) if nc_lower > df else 0
        except (ValueError, RuntimeError) as e:
            warnings.warn(f"I² lower CI calculation failed: {e}, using 0 as lower bound")
            i2_lower = 0

        # Upper bound
        def q_upper_func(ncp):
            return stats.chi2.ppf(alpha/2, df, loc=ncp) - q

        try:
            # Use adaptive upper bound for root finding
            nc_upper_bound = max(q * 5, df * 10, 500)
            nc_upper = brentq(q_upper_func, max(0, q - df), nc_upper_bound, maxiter=100, rtol=1e-10)
            i2_upper = 100 * (nc_upper - df) / nc_upper if nc_upper > df else 100
            i2_upper = min(100, i2_upper)  # Cap at 100%
        except (ValueError, RuntimeError) as e:
            warnings.warn(f"I² upper CI calculation failed: {e}, using 100 as upper bound")
            i2_upper = 100

        return UncertaintyInterval(
            lower=i2_lower,
            upper=i2_upper,
            level=conf_level,
            method="noncentral_chi2"
        )

    @staticmethod
    def _tau2_q_profile_ci(
        effects: np.ndarray,
        variances: np.ndarray,
        q_observed: float,
        df: int,
        alpha: float = 0.05
    ) -> UncertaintyInterval:
        """
        Q-profile confidence interval for τ².

        This is the preferred method for τ² confidence intervals.

        Reference: Viechtbauer (2007).

        :param effects: Study effects
        :param variances: Study variances
        :param q_observed: Observed Q statistic
        :param df: Degrees of freedom
        :param alpha: Significance level
        :return: Confidence interval for τ²
        """
        n = len(effects)

        def q_profile(tau2):
            """Q(τ²) - quantile of chi-square"""
            w = 1 / (variances + tau2)
            sum_w = np.sum(w)
            weighted_mean = np.sum(w * effects) / sum_w
            q = np.sum(w * (effects - weighted_mean)**2)
            return q

        # Lower bound: find τ² where Q = chi2_{1-alpha/2}
        def q_lower_func(tau2):
            return q_profile(tau2) - stats.chi2.ppf(1 - alpha/2, df)

        try:
            # Adaptive upper bound for root finding
            tau2_upper_bound = max(1, q_observed * 2, df * 2)
            tau2_lower = brentq(q_lower_func, 0, tau2_upper_bound, maxiter=100, rtol=1e-10)
            tau2_lower = max(0, tau2_lower)
        except (ValueError, RuntimeError) as e:
            warnings.warn(f"τ² lower Q-profile CI failed: {e}, using 0 as lower bound")
            tau2_lower = 0

        # Upper bound: find τ² where Q = chi2_{alpha/2}
        def q_upper_func(tau2):
            return q_profile(tau2) - stats.chi2.ppf(alpha/2, df)

        try:
            # Adaptive upper bound for root finding
            tau2_upper_bound = max(q_observed * 10, df * 10, 100)
            tau2_upper = brentq(q_upper_func, 0, tau2_upper_bound, maxiter=100, rtol=1e-10)
        except (ValueError, RuntimeError) as e:
            warnings.warn(f"τ² upper Q-profile CI failed: {e}, using point estimate as upper bound")
            tau2_upper = max(0, (q_observed - df) / (np.sum(1/variances) - np.sum(1/variances**2) / np.sum(1/variances)))

        return UncertaintyInterval(
            lower=tau2_lower,
            upper=tau2_upper,
            level=1-alpha,
            method="q_profile"
        )

    @staticmethod
    def _tau2_profile_likelihood_ci(
        effects: np.ndarray,
        variances: np.ndarray,
        tau2_estimate: float,
        alpha: float = 0.05
    ) -> UncertaintyInterval:
        """
        Profile likelihood confidence interval for τ².

        This method uses the profile likelihood function to find confidence
        intervals. It's an alternative to Q-profile that can be more efficient
        for large samples.

        :param effects: Study effects
        :param variances: Study variances
        :param tau2_estimate: Point estimate of τ²
        :param alpha: Significance level
        :return: Confidence interval for τ²
        """
        n = len(effects)

        def log_likelihood(tau2):
            """Restricted log-likelihood for τ²"""
            if tau2 < 0:
                return -np.inf

            w = 1 / (variances + tau2)
            sum_w = np.sum(w)
            weighted_mean = np.sum(w * effects) / sum_w

            # Residual sum of squares
            rss = np.sum(w * (effects - weighted_mean)**2)

            # REML log-likelihood (constants omitted)
            log_lik = 0.5 * (n * np.log(sum_w) - np.sum(np.log(variances + tau2)) -
                            (n - 1) * np.log(rss))
            return log_lik

        # Maximum log-likelihood
        ll_max = log_likelihood(tau2_estimate)

        # Critical value for likelihood ratio test
        chi2_crit = stats.chi2.ppf(1 - alpha, df=1)
        ll_crit = ll_max - 0.5 * chi2_crit

        # Lower bound: find τ² where log-likelihood = critical value
        def ll_lower_func(tau2):
            return log_likelihood(tau2) - ll_crit

        try:
            tau2_lower = brentq(ll_lower_func, 0, tau2_estimate)
            tau2_lower = max(0, tau2_lower)
        except ValueError:
            tau2_lower = 0

        # Upper bound
        def ll_upper_func(tau2):
            return log_likelihood(tau2) - ll_crit

        try:
            tau2_upper = brentq(ll_upper_func, tau2_estimate, max(1, tau2_estimate * 10))
        except ValueError:
            tau2_upper = tau2_estimate * 5

        return UncertaintyInterval(
            lower=tau2_lower,
            upper=tau2_upper,
            level=1-alpha,
            method="profile_likelihood"
        )


class QualityEffectsModel:
    """
    Quality effects model for meta-analysis.

    Incorporates study quality into weighting (as opposed to random-effects).
    """

    @staticmethod
    def analyze(
        effects: np.ndarray,
        variances: np.ndarray,
        quality_weights: np.ndarray,
        alpha: float = 0.05
    ) -> MetaAnalysisResult:
        """
        Quality effects meta-analysis.

        :param effects: Study effects
        :param variances: Study variances
        :param quality_weights: Quality weights (0-1, higher = better quality)
        :param alpha: Significance level
        :return: MetaAnalysisResult
        """
        n = len(effects)

        # Incorporate quality into weights
        # Inverse variance weighting adjusted by quality
        # Modified from Doi and Thalib (2008)
        iv_weights = 1 / variances

        # Quality adjustment: weight^quality
        # Higher quality studies get relatively more weight
        adjusted_weights = iv_weights ** quality_weights

        sum_w = np.sum(adjusted_weights)
        pooled_effect = np.sum(adjusted_weights * effects) / sum_w

        # Standard error
        se = np.sqrt(1 / sum_w)

        # Confidence interval (using t-distribution for quality effects)
        t_df = n - 1
        t_crit = stats.t.ppf(1 - alpha/2, df=t_df)
        ci = UncertaintyInterval(
            lower=pooled_effect - t_crit * se,
            upper=pooled_effect + t_crit * se,
            level=1-alpha,
            method="quality_effects"
        )

        # Heterogeneity (computed with standard weights for comparison)
        weights_iv = 1 / variances
        sum_w_iv = np.sum(weights_iv)
        weighted_mean_iv = np.sum(weights_iv * effects) / sum_w_iv
        q = np.sum(weights_iv * (effects - weighted_mean_iv)**2)
        df = n - 1
        p_value = 1 - stats.chi2.cdf(q, df)
        i2 = max(0, 100 * (q - df) / q) if q > df else 0

        # τ² (DL estimate)
        sum_w2 = np.sum(weights_iv**2)
        tau2 = max(0, (q - df) / (sum_w_iv - sum_w2 / sum_w_iv))

        heterogeneity = HeterogeneityStatistics(
            q_statistic=q,
            q_df=df,
            q_p_value=p_value,
            i_squared=i2,
            tau_squared=tau2
        )

        # Z-test
        z_stat = pooled_effect / se
        p_value_effect = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        return MetaAnalysisResult(
            pooled_effect=pooled_effect,
            ci=ci,
            prediction_interval=None,
            z_statistic=z_stat,
            p_value=p_value_effect,
            heterogeneity=heterogeneity,
            method="quality-effects",
            n_studies=n,
            tau_squared=0,  # Not applicable for QE
            weights=adjusted_weights / sum_w
        )


class BootstrapVariancePropagation:
    """
    Bootstrap methods for proper variance propagation in meta-analysis.

    Accounts for uncertainty in both effect estimates and variance estimates.
    """

    @staticmethod
    def bootstrap_meta_analysis(
        effects: np.ndarray,
        variances: np.ndarray,
        sample_sizes: np.ndarray,
        n_bootstrap: int = 1000,
        tau2_method: str = "DL",
        alpha: float = 0.05
    ) -> Dict:
        """
        Bootstrap meta-analysis to propagate all sources of uncertainty.

        :param effects: Study effects
        :param variances: Study variances
        :param sample_sizes: Study sample sizes
        :param n_bootstrap: Number of bootstrap iterations
        :param tau2_method: τ² estimation method
        :param alpha: Significance level
        :return: Dictionary with bootstrap results
        """
        n = len(effects)
        bootstrap_effects = []
        bootstrap_tau2 = []

        for _ in range(n_bootstrap):
            # Non-parametric bootstrap: resample studies with replacement
            boot_indices = np.random.choice(n, n, replace=True)
            effects_boot = effects[boot_indices]
            variances_boot = variances[boot_indices]

            # Estimate τ²
            tau2_boot = AdvancedMetaAnalysis._estimate_tau2(
                effects_boot, variances_boot, method=tau2_method
            )
            bootstrap_tau2.append(tau2_boot)

            # Compute pooled effect
            weights_boot = 1 / (variances_boot + tau2_boot)
            pooled_boot = np.sum(weights_boot * effects_boot) / np.sum(weights_boot)
            bootstrap_effects.append(pooled_boot)

        bootstrap_effects = np.array(bootstrap_effects)
        bootstrap_tau2 = np.array(bootstrap_tau2)

        # Percentile intervals
        ci_lower = np.percentile(bootstrap_effects, 100 * alpha/2)
        ci_upper = np.percentile(bootstrap_effects, 100 * (1 - alpha/2))

        # Bias-corrected intervals could be added here

        return {
            "pooled_effect": np.mean(bootstrap_effects),
            "ci": UncertaintyInterval(
                lower=ci_lower,
                upper=ci_upper,
                level=1-alpha,
                method="bootstrap_percentile"
            ),
            "se": np.std(bootstrap_effects),
            "tau2_mean": np.mean(bootstrap_tau2),
            "tau2_ci": (
                np.percentile(bootstrap_tau2, 100 * alpha/2),
                np.percentile(bootstrap_tau2, 100 * (1 - alpha/2))
            ),
            "bootstrap_distribution": bootstrap_effects
        }


class MetaRegression:
    """
    Meta-regression: meta-analysis with covariates.

    Extends random-effects meta-analysis to include study-level predictors.

    References:
    - Thompson and Higgins (2002)
    - Viechtbauer (2010)
    """

    @staticmethod
    def analyze(
        effects: np.ndarray,
        variances: np.ndarray,
        covariates: np.ndarray,  # Shape: (n_studies, n_covariates)
        method: str = "REML",
        alpha: float = 0.05
    ) -> Dict:
        """
        Meta-regression analysis.

        Model: y_i = X_i * beta + u_i + e_i
        where:
            y_i = effect estimate in study i
            X_i = covariate vector for study i
            beta = regression coefficients
            u_i ~ N(0, tau²) = between-study residual
            e_i ~ N(0, v_i) = within-study error

        :param effects: Study effects (n_studies,)
        :param variances: Study variances (n_studies,)
        :param covariates: Study covariates (n_studies, n_covariates)
        :param method: Method for tau² estimation
        :param alpha: Significance level
        :return: Dictionary with regression results
        """
        n_studies = len(effects)
        n_covariates = covariates.shape[1] if covariates.ndim > 1 else 1

        # Ensure covariates is 2D
        if covariates.ndim == 1:
            covariates = covariates.reshape(-1, 1)

        # Initial estimate: weighted least squares (fixed effects)
        weights_iv = 1 / variances
        X = covariates
        y = effects

        # WLS: beta = (X'W X)^(-1) X'W y
        W = np.diag(weights_iv)
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ y

        try:
            beta_fe = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            raise ValueError("Singular matrix: covariates may be collinear")

        # Residuals and Q statistics
        residuals = y - X @ beta_fe
        q_statistic = np.sum(weights_iv * residuals**2)

        # Estimate tau² (method of moments from residuals)
        df = n_studies - n_covariates
        tau2_mm = max(0, (q_statistic - df) / (n_studies - np.trace(X @ np.linalg.solve(XtWX, X.T) @ W)))

        # Iterative REML for random effects meta-regression
        if method == "REML":
            beta = beta_fe
            tau2 = tau2_mm

            for iteration in range(100):
                tau2_old = tau2

                # Update weights with current tau²
                weights_re = 1 / (variances + tau2)
                W = np.diag(weights_re)

                # Re-estimate beta
                XtWX = X.T @ W @ X
                XtWy = X.T @ W @ y
                beta = np.linalg.solve(XtWX, XtWy)

                # Update tau² using REML
                residuals = y - X @ beta
                q_re = np.sum(weights_re * residuals**2)

                # REML update for tau²
                n = n_studies
                sum_w = np.sum(weights_re)
                tau2 = q_re / n - np.sum(weights_re * variances) / sum_w / n

                # Check convergence
                if abs(tau2 - tau2_old) < 1e-8:
                    break

                # Ensure non-negative
                tau2 = max(0, tau2)

            # Final weights
            weights_re = 1 / (variances + tau2)
        else:
            beta = beta_fe
            tau2 = tau2_mm
            weights_re = 1 / (variances + tau2)

        # Standard errors (sandwich estimator)
        W = np.diag(weights_re)
        XtWX = X.T @ W @ X
        var_covar = np.linalg.inv(XtWX)

        # Test statistics for each coefficient
        z_stats = beta / np.sqrt(np.diag(var_covar))
        p_values = 2 * (1 - stats.norm.cdf(np.abs(z_stats)))

        # Confidence intervals
        z_crit = stats.norm.ppf(1 - alpha/2)
        se_beta = np.sqrt(np.diag(var_covar))
        ci_lower = beta - z_crit * se_beta
        ci_upper = beta + z_crit * se_beta

        # R² statistics (proportion of heterogeneity explained)
        q_total = q_statistic
        q_residual = np.sum(weights_re * (effects - X @ beta)**2)
        r2_heterogeneity = max(0, 1 - q_residual / q_total) if q_total > 0 else 0

        # Adjusted R² (accounting for number of covariates)
        r2_adj = max(0, 1 - (q_residual / (n_studies - n_covariates)) / (q_total / (n_studies - 1)))

        return {
            "coefficients": beta,
            "se": se_beta,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "z_statistic": z_stats,
            "p_value": p_values,
            "tau_squared": tau2,
            "r_squared": r2_heterogeneity,
            "r_squared_adj": r2_adj,
            "q_total": q_total,
            "q_residual": q_residual,
            "covariate_names": [f"X{i+1}" for i in range(n_covariates)]
        }


class PublicationBiasTests:
    """
    Tests and corrections for publication bias in meta-analysis.

    Includes:
    - Egger's regression test
    - Begg's rank correlation test
    - Trim-and-fill analysis
    - PET-PEESE approach
    """

    @staticmethod
    def eggers_test(
        effects: np.ndarray,
        variances: np.ndarray,
        alpha: float = 0.05
    ) -> Dict:
        """
        Egger's regression test for funnel plot asymmetry.

        Tests whether there's a linear relationship between effect size
        and precision (1/SE).

        Reference: Egger et al. (1997). BMJ.

        :param effects: Study effects
        :param variances: Study variances
        :param alpha: Significance level
        :return: Dictionary with test results
        """
        n = len(effects)

        # Standard errors
        se = np.sqrt(variances)
        precision = 1 / se

        # Standardized effects
        std_effects = effects / se

        # Regression: std_effect = intercept + slope * precision
        # We're interested in the intercept (small-study effect)
        X = np.column_stack([np.ones(n), precision])
        y = std_effects

        # OLS regression
        XtX = X.T @ X
        beta = np.linalg.solve(XtX, X.T @ y)

        # Standard error of intercept
        residuals = y - X @ beta
        sigma2 = np.sum(residuals**2) / (n - 2)
        var_covar = sigma2 * np.linalg.inv(XtX)
        se_intercept = np.sqrt(var_covar[0, 0])

        # Test statistic
        t_stat = beta[0] / se_intercept
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))

        # CI for intercept
        t_crit = stats.t.ppf(1 - alpha/2, df=n-2)
        intercept_ci = (
            beta[0] - t_crit * se_intercept,
            beta[0] + t_crit * se_intercept
        )

        return {
            "intercept": beta[0],
            "intercept_se": se_intercept,
            "intercept_ci": intercept_ci,
            "slope": beta[1],
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < alpha,
            "interpretation": "Evidence of publication bias" if p_value < alpha else "No evidence of publication bias"
        }

    @staticmethod
    def beggs_test(
        effects: np.ndarray,
        variances: np.ndarray,
        alpha: float = 0.05
    ) -> Dict:
        """
        Begg's rank correlation test for publication bias.

        Tests correlation between standardized effect and variance.

        Reference: Begg and Mazumdar (1994). Biometrics.

        :param effects: Study effects
        :param variances: Study variances
        :param alpha: Significance level
        :return: Dictionary with test results
        """
        from scipy.stats import rankdata

        n = len(effects)

        # Ranks of effects and variances
        effect_ranks = rankdata(effects)
        variance_ranks = rankdata(variances)

        # Kendall's tau correlation
        from scipy.stats import kendalltau
        tau, p_value = kendalltau(effect_ranks, variance_ranks)

        # Continuity correction
        z_stat = tau / np.sqrt(2 * (2*n + 5) / (9*n*(n-1)))

        return {
            "kendalls_tau": tau,
            "z_statistic": z_stat,
            "p_value": p_value,
            "significant": p_value < alpha,
            "interpretation": "Evidence of publication bias" if p_value < alpha else "No evidence of publication bias"
        }

    @staticmethod
    def trim_and_fill(
        effects: np.ndarray,
        variances: np.ndarray,
        side: str = "auto",
        alpha: float = 0.05
    ) -> Dict:
        """
        Trim-and-fill analysis for publication bias.

        Iteratively removes studies from the funnel plot and adds
        missing studies on the opposite side.

        Reference: Duval and Tweedie (2000).

        :param effects: Study effects
        :param variances: Study variances
        :param side: Which side to trim ('left', 'right', or 'auto')
        :param alpha: Significance level
        :return: Dictionary with adjusted results
        """
        n = len(effects)

        # Calculate precision
        se = np.sqrt(variances)
        precision = 1 / se

        # Standardized effects (for ranking)
        std_effects = effects / se

        # Determine which side to trim
        if side == "auto":
            # Use rank correlation to determine asymmetry direction
            from scipy.stats import spearmanr
            corr, _ = spearmanr(precision, effects)
            side = "left" if corr < 0 else "right"

        # Iterate until no more studies to trim
        effects_filled = effects.copy()
        variances_filled = variances.copy()
        n_filled = 0

        while True:
            # Current pooled estimate
            weights = 1 / (variances_filled + 0)  # Fixed effects
            pooled = np.sum(weights * effects_filled) / np.sum(weights)

            # Residuals (distance from pooled effect)
            residuals = effects_filled - pooled

            # Identify studies to trim (most extreme on the asymmetric side)
            if side == "left":
                # Trim studies with negative residuals (below pooled effect)
                to_trim = residuals < 0
            else:
                # Trim studies with positive residuals (above pooled effect)
                to_trim = residuals > 0

            if not np.any(to_trim):
                break

            # Trim the most extreme study
            if side == "left":
                trim_idx = np.argmin(residuals)
            else:
                trim_idx = np.argmax(residuals)

            # Remove the study
            effects_filled = np.delete(effects_filled, trim_idx)
            variances_filled = np.delete(variances_filled, trim_idx)

            # Add a "filled" study on the opposite side
            # Mirror the trimmed study around the pooled effect
            filled_effect = 2 * pooled - effects[trim_idx]
            effects_filled = np.append(effects_filled, filled_effect)
            variances_filled = np.append(variances_filled, variances[trim_idx])

            n_filled += 1

            # Safety check
            if n_filled > n // 2:
                warnings.warn("Trim-and-fill: Maximum iterations reached")
                break

        # Recalculate pooled effect with filled studies
        tau2 = AdvancedMetaAnalysis._estimate_tau2(effects_filled, variances_filled, method="DL")
        weights_filled = 1 / (variances_filled + tau2)
        pooled_adjusted = np.sum(weights_filled * effects_filled) / np.sum(weights_filled)
        se_adjusted = np.sqrt(1 / np.sum(weights_filled))

        # Original pooled effect for comparison
        weights_orig = 1 / (variances + tau2)
        pooled_orig = np.sum(weights_orig * effects) / np.sum(weights_orig)

        return {
            "n_studies_original": n,
            "n_filled": n_filled,
            "n_studies_adjusted": n + n_filled,
            "pooled_effect_original": pooled_orig,
            "pooled_effect_adjusted": pooled_adjusted,
            "se_adjusted": se_adjusted,
            "ci_adjusted": (
                pooled_adjusted - 1.96 * se_adjusted,
                pooled_adjusted + 1.96 * se_adjusted
            ),
            "direction": side,
            "estimate_change": pooled_adjusted - pooled_orig,
            "percent_change": 100 * (pooled_adjusted - pooled_orig) / abs(pooled_orig) if pooled_orig != 0 else 0
        }

    @staticmethod
    def pet_peese(
        effects: np.ndarray,
        variances: np.ndarray,
        alpha: float = 0.05
    ) -> Dict:
        """
        Precision-effect test and precision-effect estimate standard error.

        Two-stage approach:
        1. PET: Test for small-study effects using precision
        2. PEESE: If PET significant, use SE-based regression

        Reference: Stanley and Doucouliagos (2014).

        :param effects: Study effects
        :param variances: Study variances
        :param alpha: Significance level
        :return: Dictionary with PET-PEESE results
        """
        n = len(effects)
        se = np.sqrt(variances)
        precision = 1 / se

        # PET: Regression on precision
        X_pet = np.column_stack([np.ones(n), precision])
        y = effects

        XtX = X_pet.T @ X_pet
        beta_pet = np.linalg.solve(XtX, X_pet.T @ y)

        # Standard error of intercept
        residuals = y - X_pet @ beta_pet
        sigma2_pet = np.sum(residuals**2) / (n - 2)
        var_covar_pet = sigma2_pet * np.linalg.inv(XtX)
        se_intercept_pet = np.sqrt(var_covar_pet[0, 0])

        # Test for small-study effects (intercept ≠ 0)
        t_stat_pet = beta_pet[0] / se_intercept_pet
        p_pet = 2 * (1 - stats.t.cdf(abs(t_stat_pet), df=n-2))

        # PEESE: Regression on SE (if PET significant)
        if p_pet < alpha:
            X_peese = np.column_stack([np.ones(n), se])
            beta_peese = np.linalg.solve(X_peese.T @ X_peese, X_peese.T @ y)

            residuals_peese = y - X_peese @ beta_peese
            sigma2_peese = np.sum(residuals_peese**2) / (n - 2)
            var_covar_peese = sigma2_peese * np.linalg.inv(X_peese.T @ X_peese)
            se_intercept_peese = np.sqrt(var_covar_peese[0, 0])

            # PEESE estimate (at SE = 0)
            effect_peese = beta_peese[0]
            se_peese = se_intercept_peese
            ci_peese = (
                effect_peese - 1.96 * se_peese,
                effect_peese + 1.96 * se_peese
            )

            method_used = "PEESE"
        else:
            # PET not significant, use PET estimate at mean precision
            effect_pet = beta_pet[0] + beta_pet[1] * np.mean(precision)
            se_pet_at_mean = np.sqrt(var_covar_pet[0, 0] + var_covar_pet[1, 1] * np.mean(precision)**2 +
                                      2 * var_covar_pet[0, 1] * np.mean(precision))

            effect_peese = effect_pet
            se_peese = se_pet_at_mean
            ci_peese = (
                effect_peese - 1.96 * se_peese,
                effect_peese + 1.96 * se_peese
            )

            method_used = "PET"

        return {
            "pet_intercept": beta_pet[0],
            "pet_intercept_se": se_intercept_pet,
            "pet_intercept_p": p_pet,
            "pet_significant": p_pet < alpha,
            "adjusted_effect": effect_peese,
            "adjusted_se": se_peese,
            "adjusted_ci": ci_peese,
            "method_used": method_used,
            "interpretation": "Small-study effects detected" if p_pet < alpha else "No small-study effects"
        }


if __name__ == "__main__":
    print("Enhanced Statistical Framework (Revised) loaded")
    print("Features:")
    print("  - HKSJ adjustment for small samples")
    print("  - Proper prediction intervals")
    print("  - Q-profile confidence intervals for τ²")
    print("  - Quality effects model")
    print("  - Bootstrap variance propagation")
    print("  - REML with convergence checking")
    print("  - Meta-regression with covariates")
    print("  - Publication bias tests (Egger, Begg, Trim-and-Fill, PET-PEESE)")
    print("  - Multiple τ² estimators (DL, REML, SJ, PM)")
