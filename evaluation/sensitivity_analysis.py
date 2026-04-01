"""
Subgroup Analysis for Meta-Analysis

Implements formal subgroup comparisons with proper statistical inference.

References:
- Borenstein et al. (2009). Introduction to Meta-Analysis.
- Deeks et al. (2019). Cochrane Handbook.
- Riley et al. (2011). Statistics in Medicine.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings


@dataclass
class SubgroupResult:
    """Results from subgroup analysis."""
    subgroup_name: str
    n_studies: int
    pooled_effect: float
    ci_lower: float
    ci_upper: float
    p_value: float
    tau_squared: float
    i_squared: float
    q_statistic: float


@dataclass
class SubgroupComparison:
    """Results from formal subgroup comparison."""
    test_statistic: float
    df: int
    p_value: float
    significant: bool
    interpretation: str


class SubgroupAnalysis:
    """
    Subgroup analysis with formal statistical tests.

    Tests whether effect sizes differ significantly between subgroups.
    """

    @staticmethod
    def analyze(
        effects: np.ndarray,
        variances: np.ndarray,
        subgroups: np.ndarray,
        tau2_method: str = "DL",
        alpha: float = 0.05
    ) -> Dict:
        """
        Perform subgroup meta-analysis.

        Model: Within each subgroup g, use random-effects model:
            y_ig ~ N(theta_g, tau_g^2 + v_ig)

        Test: H0: theta_1 = theta_2 = ... = theta_G

        :param effects: Study effects
        :param variances: Study variances
        :param subgroups: Subgroup labels (same length as effects)
        :param tau2_method: Method for τ² estimation within subgroups
        :param alpha: Significance level
        :return: Dictionary with subgroup results
        """
        unique_subgroups = np.unique(subgroups)
        n_subgroups = len(unique_subgroups)

        # Import main analysis functions
        from evaluation.statistical_framework_v2 import AdvancedMetaAnalysis

        # Analyze each subgroup separately
        subgroup_results = {}
        all_effects = []
        all_weights = []

        for subgroup in unique_subgroups:
            mask = subgroups == subgroup
            effects_g = effects[mask]
            variances_g = variances[mask]
            n_g = len(effects_g)

            # Random-effects meta-analysis within subgroup
            result_g = AdvancedMetaAnalysis.random_effects_analysis(
                effects_g,
                variances_g,
                tau2_method=tau2_method,
                ci_method="wald",
                prediction=False,
                alpha=alpha
            )

            subgroup_results[subgroup] = SubgroupResult(
                subgroup_name=str(subgroup),
                n_studies=n_g,
                pooled_effect=result_g.pooled_effect,
                ci_lower=result_g.ci.lower,
                ci_upper=result_g.ci.upper,
                p_value=result_g.p_value,
                tau_squared=result_g.tau_squared,
                i_squared=result_g.heterogeneity.i_squared,
                q_statistic=result_g.heterogeneity.q_statistic
            )

            all_effects.append(result_g.pooled_effect)
            # Use inverse variance weights for between-subgroup test
            all_weights.append(1 / result_g.ci.width()**2)

        all_effects = np.array(all_effects)
        all_weights = np.array(all_weights)

        # Formal test for subgroup differences
        # Q_between = sum(w_g * (theta_g - theta_pooled)^2)
        pooled_overall = np.sum(all_weights * all_effects) / np.sum(all_weights)
        q_between = np.sum(all_weights * (all_effects - pooled_overall)**2)
        df_between = n_subgroups - 1
        p_between = 1 - stats.chi2.cdf(q_between, df_between)

        comparison = SubgroupComparison(
            test_statistic=q_between,
            df=df_between,
            p_value=p_between,
            significant=p_between < alpha,
            interpretation=f"Subgroup difference is {'significant' if p_between < alpha else 'not significant'} (p={p_between:.4f})"
        )

        # Test for residual heterogeneity within subgroups
        # This tests whether there's remaining heterogeneity not explained by subgroups
        q_within = 0
        df_within = 0
        for subgroup in unique_subgroups:
            mask = subgroups == subgroup
            effects_g = effects[mask]
            variances_g = variances[mask]
            result_g = subgroup_results[subgroup]
            q_within += result_g.q_statistic
            df_within += len(effects_g) - 1

        p_within = 1 - stats.chi2.cdf(q_within, df_within)

        # I² for subgroups (proportion of variance explained)
        q_total = q_between + q_within
        i_squared_between = (q_between / q_total * 100) if q_total > 0 else 0
        i_squared_within = (q_within / q_total * 100) if q_total > 0 else 0

        return {
            "subgroups": subgroup_results,
            "n_subgroups": n_subgroups,
            "between_subgroups": {
                "q_statistic": q_between,
                "df": df_between,
                "p_value": p_between,
                "interpretation": comparison.interpretation
            },
            "within_subgroups": {
                "q_statistic": q_within,
                "df": df_within,
                "p_value": p_within
            },
            "i_squared_explained": i_squared_between,
            "i_squared_residual": i_squared_within,
            "recommendation": "Subgroup analysis explains substantial heterogeneity" if i_squared_between > 50 else
                             "Subgroup differences are minimal"
        }

    @staticmethod
    def meta_regression_subgroup(
        effects: np.ndarray,
        variances: np.ndarray,
        subgroups: np.ndarray,
        alpha: float = 0.05
    ) -> Dict:
        """
        Subgroup analysis using meta-regression approach.

        More flexible than traditional subgroup analysis:
        - Can handle continuous subgroup variables
        - Can test for trend across ordered subgroups
        - Provides similar results with better properties

        :param effects: Study effects
        :param variances: Study variances
        :param subgroups: Subgroup variable (categorical or continuous)
        :param alpha: Significance level
        :return: Dictionary with regression results
        """
        from evaluation.statistical_framework_v2 import MetaRegression

        # Convert categorical subgroups to dummy variables
        unique_subgroups = np.unique(subgroups)
        n_subgroups = len(unique_subgroups)

        if n_subgroups == 2:
            # Binary subgroup: single indicator
            covariates = (subgroups == unique_subgroups[1]).astype(float)
            method = "binary"
        elif n_subgroups <= 5:
            # Create dummy variables for categorical subgroups
            covariates = np.zeros((len(subgroups), n_subgroups - 1))
            for i, subgroup in enumerate(unique_subgroups[:-1]):
                covariates[:, i] = (subgroups == subgroup).astype(float)
            method = "categorical"
        else:
            # Too many subgroups, treat as continuous if possible
            try:
                # Try to convert to numeric
                covariates = pd.Categorical(subgroups).codes.astype(float)
                method = "ordinal"
            except:
                # Fall back to using first few categories
                covariates = np.zeros((len(subgroups), n_subgroups - 1))
                for i, subgroup in enumerate(unique_subgroups[:-1]):
                    covariates[:, i] = (subgroups == subgroup).astype(float)
                method = "categorical_truncated"

        # Run meta-regression
        result = MetaRegression.analyze(
            effects,
            variances,
            covariates,
            method="REML",
            alpha=alpha
        )

        # Extract subgroup effect test
        if method == "binary":
            # Single coefficient test
            coef = result["coefficients"][1] if len(result["coefficients"]) > 1 else result["coefficients"][0]
            se = result["se"][1] if len(result["se"]) > 1 else result["se"][0]
            p_val = result["p_value"][1] if len(result["p_value"]) > 1 else result["p_value"][0]
            z_stat = coef / se

            interpretation = {
                "method": method,
                "subgroup_effect": coef,
                "se": se,
                "z_statistic": z_stat,
                "p_value": p_val,
                "significant": p_val < alpha,
                "interpretation": f"Subgroup difference is {'significant' if p_val < alpha else 'not significant'}"
            }
        else:
            # Multiple coefficients - omnibus test
            interpretation = {
                "method": method,
                "n_coefficients": len(result["coefficients"]),
                "p_values": result["p_value"].tolist(),
                "significant": any(p < alpha for p in result["p_value"]),
                "interpretation": f"At least one subgroup differs" if any(p < alpha for p in result["p_value"]) else "No significant subgroup differences"
            }

        return {
            "regression_result": result,
            "subgroup_test": interpretation,
            "r_squared": result["r_squared"],
            "r_squared_adj": result["r_squared_adj"]
        }


class SensitivityAnalysis:
    """
    Sensitivity analysis for meta-analysis.

    Assesses robustness of findings to various assumptions and individual studies.
    """

    @staticmethod
    def leave_one_out(
        effects: np.ndarray,
        variances: np.ndarray,
        study_names: Optional[List[str]] = None,
        tau2_method: str = "DL",
        alpha: float = 0.05
    ) -> pd.DataFrame:
        """
        Leave-one-out sensitivity analysis.

        Recomputes meta-analysis omitting each study in turn.
        Identifies influential studies that substantially change results.

        :param effects: Study effects
        :param variances: Study variances
        :param study_names: Optional study names
        :param tau2_method: Method for τ² estimation
        :param alpha: Significance level
        :return: DataFrame with leave-one-out results
        """
        from evaluation.statistical_framework_v2 import AdvancedMetaAnalysis

        n = len(effects)

        if study_names is None:
            study_names = [f"Study {i+1}" for i in range(n)]

        # Overall analysis
        overall = AdvancedMetaAnalysis.random_effects_analysis(
            effects, variances, tau2_method=tau2_method,
            ci_method="wald", prediction=False, alpha=alpha
        )

        results = []
        for i in range(n):
            # Omit study i
            mask = np.arange(n) != i
            effects_omit = effects[mask]
            variances_omit = variances[mask]

            # Recompute meta-analysis
            result_omit = AdvancedMetaAnalysis.random_effects_analysis(
                effects_omit, variances_omit, tau2_method=tau2_method,
                ci_method="wald", prediction=False, alpha=alpha
            )

            # Calculate influence
            influence = result_omit.pooled_effect - overall.pooled_effect
            influence_pct = 100 * influence / abs(overall.pooled_effect) if overall.pooled_effect != 0 else 0

            results.append({
                "omitted_study": study_names[i],
                "n_remaining": n - 1,
                "pooled_effect": result_omit.pooled_effect,
                "ci_lower": result_omit.ci.lower,
                "ci_upper": result_omit.ci.upper,
                "tau_squared": result_omit.tau_squared,
                "i_squared": result_omit.heterogeneity.i_squared,
                "influence": influence,
                "influence_pct": influence_pct,
                "p_value": result_omit.p_value
            })

        df = pd.DataFrame(results)

        # Add overall row for comparison
        overall_row = {
            "omitted_study": "Overall (all studies)",
            "n_remaining": n,
            "pooled_effect": overall.pooled_effect,
            "ci_lower": overall.ci.lower,
            "ci_upper": overall.ci.upper,
            "tau_squared": overall.tau_squared,
            "i_squared": overall.heterogeneity.i_squared,
            "influence": 0,
            "influence_pct": 0,
            "p_value": overall.p_value
        }

        return pd.concat([pd.DataFrame([overall_row]), df], ignore_index=True)

    @staticmethod
    def cumulative_meta_analysis(
        effects: np.ndarray,
        variances: np.ndarray,
        study_names: Optional[List[str]] = None,
        order_by: str = "input",
        tau2_method: str = "DL",
        alpha: float = 0.05
    ) -> pd.DataFrame:
        """
        Cumulative meta-analysis.

        Recomputes meta-analysis adding studies one at a time.
        Shows how evidence accumulates over time.

        :param effects: Study effects
        :param variances: Study variances
        :param study_names: Optional study names
        :param order_by: How to order studies ('input', 'effect', 'precision', 'year')
        :param tau2_method: Method for τ² estimation
        :param alpha: Significance level
        :return: DataFrame with cumulative results
        """
        from evaluation.statistical_framework_v2 import AdvancedMetaAnalysis

        n = len(effects)

        if study_names is None:
            study_names = [f"Study {i+1}" for i in range(n)]

        # Determine ordering
        if order_by == "input":
            order = np.arange(n)
        elif order_by == "effect":
            order = np.argsort(effects)
        elif order_by == "precision":
            order = np.argsort(variances)  # Lower variance = higher precision
        elif order_by == "year":
            # Assume study names contain years or use input order
            order = np.arange(n)
        else:
            order = np.arange(n)

        # Reorder data
        effects_ordered = effects[order]
        variances_ordered = variances[order]
        names_ordered = [study_names[i] for i in order]

        results = []
        for i in range(1, n + 1):
            # Include first i studies
            effects_cumul = effects_ordered[:i]
            variances_cumul = variances_ordered[:i]

            # Compute meta-analysis
            result_cumul = AdvancedMetaAnalysis.random_effects_analysis(
                effects_cumul, variances_cumul, tau2_method=tau2_method,
                ci_method="wald", prediction=False, alpha=alpha
            )

            results.append({
                "n_studies": i,
                "last_added": names_ordered[i - 1],
                "pooled_effect": result_cumul.pooled_effect,
                "ci_lower": result_cumul.ci.lower,
                "ci_upper": result_cumul.ci.upper,
                "se": (result_cumul.ci.upper - result_cumul.ci.lower) / (2 * stats.norm.ppf(1 - alpha/2)),
                "tau_squared": result_cumul.tau_squared,
                "i_squared": result_cumul.heterogeneity.i_squared,
                "z_statistic": result_cumul.z_statistic,
                "p_value": result_cumul.p_value,
                "significant": result_cumul.p_value < alpha
            })

        return pd.DataFrame(results)

    @staticmethod
    def influence_diagnostics(
        effects: np.ndarray,
        variances: np.ndarray,
        study_names: Optional[List[str]] = None,
        alpha: float = 0.05
    ) -> Dict:
        """
        Comprehensive influence diagnostics.

        Computes multiple measures of study influence:
        - Cook's distance
        - DFBETAS (change in coefficients when study omitted)
        - Covariance ratio
        - Standardized residual
        - Hat value (leverage)

        :param effects: Study effects
        :param variances: Study variances
        :param study_names: Optional study names
        :param alpha: Significance level
        :return: Dictionary with diagnostic results
        """
        from evaluation.statistical_framework_v2 import AdvancedMetaAnalysis

        n = len(effects)

        if study_names is None:
            study_names = [f"Study {i+1}" for i in range(n)]

        # Overall analysis
        overall = AdvancedMetaAnalysis.random_effects_analysis(
            effects, variances, tau2_method="DL",
            ci_method="wald", prediction=False, alpha=alpha
        )

        tau2 = overall.tau_squared
        weights = 1 / (variances + tau2)
        sum_weights = np.sum(weights)
        weights_rel = weights / sum_weights

        diagnostics = []
        for i in range(n):
            # Leave-one-out
            mask = np.arange(n) != i
            effects_omit = effects[mask]
            variances_omit = variances[mask]

            result_omit = AdvancedMetaAnalysis.random_effects_analysis(
                effects_omit, variances_omit, tau2_method="DL",
                ci_method="wald", prediction=False, alpha=alpha
            )

            # Influence measures
            theta_overall = overall.pooled_effect
            theta_omit = result_omit.pooled_effect

            # Cook's distance (simplified)
            cook_d = (theta_omit - theta_overall)**2 / (overall.ci.width()**2)

            # DFBETAS
            dfbetas = (theta_omit - theta_overall) / np.sqrt(variances[i])

            # Covariance ratio
            var_overall = (1 / sum_weights)**2
            var_omit = (1 / np.sum(1 / (variances_omit + result_omit.tau_squared)))**2
            cov_ratio = var_omit / var_overall

            # Standardized residual
            residual = effects[i] - theta_overall
            se_i = np.sqrt(variances[i] + tau2)
            std_residual = residual / se_i

            # Hat value (leverage in meta-analysis)
            # Using relative weight as leverage measure
            hat_i = weights_rel[i]

            # Influence summary
            is_influential = (
                abs(dfbetas) > 1 or  # DFBETAS threshold
                cov_ratio < 0.8 or cov_ratio > 1.2 or  # Covariance ratio thresholds
                abs(std_residual) > 2  # Outlier threshold
            )

            diagnostics.append({
                "study": study_names[i],
                "weight": weights_rel[i],
                "standardized_residual": std_residual,
                "hat_value": hat_i,
                "cook_distance": cook_d,
                "dfbetas": dfbetas,
                "covariance_ratio": cov_ratio,
                "is_influential": is_influential,
                "interpretation": "Influential study" if is_influential else "Not influential"
            })

        df = pd.DataFrame(diagnostics)

        # Summary
        n_influential = df["is_influential"].sum()

        return {
            "diagnostics": df,
            "n_influential": n_influential,
            "interpretation": f"{n_influential} influential study{'ies' if n_influential != 1 else ''} identified",
            "recommendation": "Consider sensitivity analysis excluding influential studies" if n_influential > 0 else "No concerning influential studies"
        }

    @staticmethod
    def subset_analysis(
        effects: np.ndarray,
        variances: np.ndarray,
        criteria: np.ndarray,
        criterion_name: str = "subset",
        alpha: float = 0.05
    ) -> Dict:
        """
        Sensitivity analysis based on study characteristics.

        Compare meta-analysis results across different subsets of studies.

        :param effects: Study effects
        :param variances: Study variances
        :param criteria: Binary criteria for inclusion (True = include)
        :param criterion_name: Name of the criterion
        :param alpha: Significance level
        :return: Dictionary with subset comparison results
        """
        from evaluation.statistical_framework_v2 import AdvancedMetaAnalysis

        # Analysis with all studies
        all_result = AdvancedMetaAnalysis.random_effects_analysis(
            effects, variances, tau2_method="DL",
            ci_method="wald", prediction=False, alpha=alpha
        )

        # Analysis with subset only
        effects_subset = effects[criteria]
        variances_subset = variances[criteria]

        if len(effects_subset) < 2:
            return {
                "criterion": criterion_name,
                "n_total": len(effects),
                "n_subset": len(effects_subset),
                "error": "Not enough studies in subset"
            }

        subset_result = AdvancedMetaAnalysis.random_effects_analysis(
            effects_subset, variances_subset, tau2_method="DL",
            ci_method="wald", prediction=False, alpha=alpha
        )

        # Compare results
        diff = subset_result.pooled_effect - all_result.pooled_effect
        diff_pct = 100 * diff / abs(all_result.pooled_effect) if all_result.pooled_effect != 0 else 0

        # Check if CI overlap
        ci_overlap = not (subset_result.ci.upper < all_result.ci.lower or
                          subset_result.ci.lower > all_result.ci.upper)

        return {
            "criterion": criterion_name,
            "n_total": len(effects),
            "n_subset": len(effects_subset),
            "n_excluded": len(effects) - len(effects_subset),
            "all_studies": {
                "effect": all_result.pooled_effect,
                "ci": [all_result.ci.lower, all_result.ci.upper],
                "tau2": all_result.tau_squared,
                "i2": all_result.heterogeneity.i_squared
            },
            "subset_only": {
                "effect": subset_result.pooled_effect,
                "ci": [subset_result.ci.lower, subset_result.ci.upper],
                "tau2": subset_result.tau_squared,
                "i2": subset_result.heterogeneity.i_squared
            },
            "difference": {
                "absolute": diff,
                "percent": diff_pct
            },
            "ci_overlap": ci_overlap,
            "robust": abs(diff_pct) < 20,  # Less than 20% change
            "interpretation": f"Results are {'robust' if abs(diff_pct) < 20 else 'sensitive'} to this criterion"
        }


if __name__ == "__main__":
    print("Subgroup and Sensitivity Analysis Module loaded")
    print("Features:")
    print("  - Formal subgroup comparison tests")
    print("  - Meta-regression approach to subgroups")
    print("  - Leave-one-out sensitivity analysis")
    print("  - Cumulative meta-analysis")
    print("  - Influence diagnostics (Cook's D, DFBETAS, covariance ratio)")
    print("  - Subset analysis by study characteristics")
