"""
Meta-Regression and Sensitivity Analysis Toolkit (DEPRECATED)

⚠️ DEPRECATION WARNING (v2.0): This module is deprecated.

Please use meta_regression_v2.py instead, which includes:
- Knapp-Hartung adjustment for standard errors
- VIF multicollinearity detection
- Proper model selection with AIC/BIC
- F-tests for overall model significance

Migration: Replace `from meta_regression import` with `from meta_regression_v2 import`

This module will be removed in version 3.0.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from scipy import stats
import warnings

# Issue deprecation warning
warnings.warn(
    "meta_regression.py is deprecated. Use meta_regression_v2.py instead. "
    "This module will be removed in version 3.0.",
    DeprecationWarning,
    stacklevel=2
)


@dataclass
class MetaRegressionResult:
    """Results from meta-regression"""
    intercept: float
    intercept_se: float
    intercept_ci: Tuple[float, float]
    slope: Optional[float] = None  # For single covariate
    slope_se: Optional[float] = None
    slope_ci: Optional[Tuple[float, float]] = None
    covariate_effects: Optional[Dict[str, Dict[str, float]]] = None
    r_squared: float = 0
    adjusted_r_squared: float = 0
    q_residual: float = 0
    q_df: float = 0
    q_p_value: float = 1
    tau_squared_residual: float = 0
    n_studies: int = 0


@dataclass
class InfluenceAnalysisResult:
    """Results from influence analysis"""
    study_influence: Dict[str, float]  # Study ID -> influence statistic
    influential_studies: List[str]
    cook_distances: Dict[str, float]
    dfbetas: Dict[str, float]
    leverage: Dict[str, float]


@dataclass
class CumulativeMetaResult:
    """Results from cumulative meta-analysis"""
    study_order: List[str]
    cumulative_effects: List[float]
    cumulative_ci_lower: List[float]
    cumulative_ci_upper: List[float]
    z_statistics: List[float]
    timestamp_studies: List[str]
    final_estimate: float
    final_ci: Tuple[float, float]


@dataclass
class SensitivityAnalysisResult:
    """Results from sensitivity analysis"""
    analysis_type: str
    n_scenarios: int
    base_case: Dict[str, float]
    results: List[Dict[str, Dict[str, float]]]
    robust_to_exclusions: bool
    outliers: List[str]
    influential_studies: List[str]


class MetaRegressionAnalyzer:
    """
    Meta-regression for exploring heterogeneity using study-level covariates.

    Implements:
    - Weighted least squares meta-regression
    - Random-effects meta-regression
    - Multiple covariates
    - Prediction intervals
    """

    def __init__(self):
        self.effects = np.array([])
        self.variances = np.array([])
        self.covariates: Optional[pd.DataFrame] = None
        self.study_ids = []

    def load_data(
        self,
        effects: np.ndarray,
        variances: np.ndarray,
        covariates: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        study_ids: Optional[List[str]] = None
    ) -> None:
        """
        Load data for meta-regression.

        :param effects: Array of effect sizes
        :param variances: Array of variances
        :param covariates: Covariate matrix or DataFrame
        :param study_ids: Optional study identifiers
        """
        self.effects = np.array(effects)
        self.variances = np.array(variances)

        if covariates is not None:
            if isinstance(covariates, pd.DataFrame):
                self.covariates = covariates
            else:
                self.covariates = pd.DataFrame(covariates)

        if study_ids:
            self.study_ids = study_ids
        else:
            self.study_ids = [f"study_{i}" for i in range(len(effects))]

    def fit(
        self,
        method: str = "random",
        predictors: Optional[List[str]] = None
    ) -> MetaRegressionResult:
        """
        Fit meta-regression model.

        :param method: 'fixed' or 'random'
        :param predictors: List of covariate names to include
        :return: MetaRegressionResult
        """
        if len(self.effects) == 0:
            raise ValueError("No data loaded")

        # Prepare design matrix
        if self.covariates is not None:
            if predictors:
                X = self.covariates[predictors].values
                predictor_names = predictors
            else:
                X = self.covariates.values
                predictor_names = list(self.covariates.columns)
        else:
            # Intercept-only model
            X = np.ones((len(self.effects), 1))
            predictor_names = ['intercept']

        n_studies = len(self.effects)
        n_predictors = X.shape[1]

        # Add intercept column if not present
        if not np.allclose(X[:, 0], 1):
            X = np.column_stack([np.ones(n_studies), X])
            predictor_names = ['intercept'] + predictor_names
        else:
            X = X.copy()

        if method == "fixed":
            return self._fixed_effect_regression(X, predictor_names)
        else:
            return self._random_effects_regression(X, predictor_names)

    def _fixed_effect_regression(
        self,
        X: np.ndarray,
        predictor_names: List[str]
    ) -> MetaRegressionResult:
        """Fixed-effect meta-regression using WLS"""
        y = self.effects
        V = np.diag(self.variances)

        # Weighted least squares
        W = np.linalg.inv(V)
        XTW = X.T @ W
        XTWX_inv = np.linalg.pinv(XTW @ X)
        beta = XTWX_inv @ XTW @ y

        # Standard errors
        resid_variance = np.sum((y - X @ beta)**2) / (len(y) - len(beta))
        var_beta = resid_variance * XTWX_inv

        se = np.sqrt(np.diag(var_beta))

        # Confidence intervals
        t_crit = stats.t.ppf(0.975, df=len(y) - len(beta))
        ci_lower = beta - t_crit * se
        ci_upper = beta + t_crit * se

        # R-squared
        y_pred = X @ beta
        ss_tot = np.sum((y - np.mean(y))**2)
        ss_resid = np.sum((y - y_pred)**2)
        r_squared = 1 - ss_resid / ss_tot

        # Adjusted R-squared
        adj_r2 = 1 - (1 - r_squared) * (len(y) - 1) / (len(y) - len(beta))

        # Q statistic (residual heterogeneity)
        q = np.sum(W @ (y - X @ beta)**2)
        q_df = len(y) - len(beta)
        q_p_value = 1 - stats.chi2.cdf(q, q_df)

        # Extract results
        result = MetaRegressionResult(
            intercept=float(beta[0]),
            intercept_se=float(se[0]),
            intercept_ci=(float(ci_lower[0]), float(ci_upper[0])),
            n_studies=len(y)
        )

        # Single covariate
        if len(beta) == 2:
            result.slope = float(beta[1])
            result.slope_se = float(se[1])
            result.slope_ci = (float(ci_lower[1]), float(ci_upper[1]))

        # Multiple covariates
        if len(beta) > 2:
            cov_effects = {}
            for i, name in enumerate(predictor_names[1:], 1):
                cov_effects[name] = {
                    'coefficient': float(beta[i]),
                    'se': float(se[i]),
                    'ci_lower': float(ci_lower[i]),
                    'ci_upper': float(ci_upper[i]),
                    'p_value': 2 * (1 - stats.norm.cdf(abs(beta[i] / se[i])))
                }
            result.covariate_effects = cov_effects

        result.r_squared = r_squared
        result.adjusted_r_squared = adj_r2
        result.q_residual = q
        result.q_df = q_df
        result.q_p_value = q_p_value

        return result

    def _random_effects_regression(
        self,
        X: np.ndarray,
        predictor_names: List[str]
    ) -> MetaRegressionResult:
        """Random-effects meta-regression using DL estimator"""
        y = self.effects
        v = self.variances

        n_studies = len(y)
        n_predictors = X.shape[1]

        # Iteratively estimate tau² and beta
        tau2 = 0
        max_iter = 100
        tolerance = 1e-6

        for iteration in range(max_iter):
            tau2_old = tau2

            # Weights with current tau²
            w = 1 / (v + tau2)
            W = np.diag(w)
            XTW = X.T @ W
            XTWX_inv = np.linalg.pinv(XTW @ X)
            beta = XTWX_inv @ XTW @ y

            # Update tau² (DL for residuals)
            resid = y - X @ beta
            q = np.sum(w * resid**2)

            df = n_studies - n_predictors
            sum_w = np.sum(w)
            sum_w2 = np.sum(w**2)

            if q > df:
                tau2 = (q - df) / (sum_w - sum_w2 / sum_w)
            else:
                tau2 = 0

            if abs(tau2 - tau2_old) < tolerance:
                break

        # Standard errors
        var_beta = XTWX_inv
        se = np.sqrt(np.diag(var_beta))

        # Confidence intervals
        z_crit = stats.norm.ppf(0.975)
        ci_lower = beta - z_crit * se
        ci_upper = beta + z_crit * se

        # R-squared (conditional on covariates)
        y_pred = X @ beta
        ss_tot = np.sum((y - np.mean(y))**2)
        ss_resid = np.sum((y - y_pred)**2)
        r_squared = 1 - ss_resid / ss_tot if ss_tot > 0 else 0

        # Adjusted R²
        adj_r2 = 1 - (1 - r_squared) * (n_studies - 1) / (n_studies - n_predictors)

        result = MetaRegressionResult(
            intercept=float(beta[0]),
            intercept_se=float(se[0]),
            intercept_ci=(float(ci_lower[0]), float(ci_upper[0])),
            n_studies=n_studies,
            r_squared=r_squared,
            adjusted_r_squared=adj_r2,
            tau_squared_residual=tau2,
            q_residual=q,
            q_df=df,
            q_p_value=1 - stats.chi2.cdf(q, df)
        )

        # Add slope/covariates
        if len(beta) == 2:
            result.slope = float(beta[1])
            result.slope_se = float(se[1])
            result.slope_ci = (float(ci_lower[1]), float(ci_upper[1]))

        if len(beta) > 2:
            cov_effects = {}
            for i, name in enumerate(predictor_names[1:], 1):
                cov_effects[name] = {
                    'coefficient': float(beta[i]),
                    'se': float(se[i]),
                    'ci_lower': float(ci_lower[i]),
                    'ci_upper': float(ci_upper[i]),
                    'p_value': 2 * (1 - stats.norm.cdf(abs(beta[i] / se[i])))
                }
            result.covariate_effects = cov_effects

        return result

    def predict(
        self,
        result: MetaRegressionResult,
        new_covariates: Union[pd.DataFrame, np.ndarray],
        tau_squared: Optional[float] = None
    ) -> Tuple[float, float, Tuple[float, float]]:
        """
        Predict effect for new studies using the meta-regression model.

        :param result: Fitted meta-regression result
        :param new_covariates: Covariate values for prediction
        :param tau_squared: Between-study variance (use residual if None)
        :return: Tuple of (predicted_effect, se, ci_lower, ci_upper)
        """
        # Extract beta and prepare covariate row
        if result.covariate_effects:
            # Multiple covariates
            beta = np.array([result.intercept] +
                          [result.covariate_effects[k]['coefficient']
                           for k in sorted(result.covariate_effects.keys())])
        else:
            beta = np.array([result.intercept, result.slope or 0])

        if isinstance(new_covariates, pd.DataFrame):
            x_new = np.array([1] + [new_covariates[k].iloc[0]
                                          for k in sorted(new_covariates.columns)])
        else:
            x_new = np.array([1] + list(new_covariates))

        # Point prediction
        prediction = x_new @ beta

        # Use residual tau² if not provided
        if tau_squared is None:
            tau_squared = result.tau_squared_residual

        # Prediction variance = model variance + tau²
        # This is simplified; full calculation would use X * var_beta * X'
        model_var = tau_squared  # Simplified

        se = np.sqrt(model_var)

        # CI
        z = stats.norm.ppf(0.975)
        ci_lower = prediction - z * se
        ci_upper = prediction + z * se

        return float(prediction), float(se), (float(ci_lower), float(ci_upper))


class SensitivityAnalyzer:
    """
    Comprehensive sensitivity analysis for meta-analysis.

    Implements:
    - Leave-one-out analysis
    - Influence analysis (Cook's distance, DFBETAS)
    - Cumulative meta-analysis
    - Subgroup sensitivity
    - Model selection sensitivity
    """

    def __init__(self, effects: np.ndarray, variances: np.ndarray):
        self.effects = effects
        self.variances = variances
        self.study_indices = np.arange(len(effects))

    def leave_one_out_analysis(
        self,
        method: str = "dl"
    ) -> pd.DataFrame:
        """
        Perform leave-one-out cross-validation.

        :param method: Method for pooling ('dl', 'reml', 'fe')
        :return: DataFrame with leave-one-out results
        """
        results = []

        for leave_out_idx in self.study_indices:
            mask = np.ones(len(self.effects), dtype=bool)
            mask[leave_out_idx] = False

            effects_loo = self.effects[mask]
            variances_loo = self.variances[mask]

            if len(effects_loo) < 2:
                continue

            # Re-fit model without this study
            result = self._pool_effects(effects_loo, variances_loo, method)
            result['left_out_study'] = leave_out_idx
            result['n_studies_included'] = len(effects_loo)

            results.append(result)

        df = pd.DataFrame(results)
        return df

    def _pool_effects(
        self,
        effects: np.ndarray,
        variances: np.ndarray,
        method: str
    ) -> Dict[str, float]:
        """Pool effects using specified method"""
        if method == "fe":
            weights = 1 / variances
            pooled = np.sum(weights * effects) / np.sum(weights)
            se = np.sqrt(1 / np.sum(weights))
            tau2 = 0
        else:  # DL
            weights = 1 / variances
            weighted_mean = np.sum(weights * effects) / np.sum(weights)
            q = np.sum(weights * (effects - weighted_mean)**2)
            df = len(effects) - 1

            if q > df:
                sum_w = np.sum(weights)
                sum_w2 = np.sum(weights**2)
                tau2 = (q - df) / (sum_w - sum_w2 / sum_w)
            else:
                tau2 = 0

            re_weights = 1 / (variances + tau2)
            pooled = np.sum(re_weights * effects) / np.sum(re_weights)
            se = np.sqrt(1 / np.sum(re_weights))

        ci_lower = pooled - 1.96 * se
        ci_upper = pooled + 1.96 * se

        return {
            'pooled_effect': pooled,
            'se': se,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'tau_squared': tau2,
            'i_squared': max(0, 100 * (q - df) / q) if q > df and method != "fe" else 0
        }

    def influence_analysis(
        self,
        base_result: Optional[Dict[str, float]] = None
    ) -> InfluenceAnalysisResult:
        """
        Perform influence analysis to identify influential studies.

        :param base_result: Base case result (all studies)
        :return: InfluenceAnalysisResult
        """
        if base_result is None:
            base_result = self._pool_effects(self.effects, self.variances, "dl")

        base_effect = base_result['pooled_effect']

        influence = {}
        cooks = {}
        dfbetas = {}
        leverage = []

        for i in self.study_indices:
            # Leave-one-out
            mask = np.ones(len(self.effects), dtype=bool)
            mask[i] = False

            effects_loo = self.effects[mask]
            variances_loo = self.variances[mask]

            loo_result = self._pool_effects(effects_loo, variances_loo, "dl")
            loo_effect = loo_result['pooled_effect']

            # DFBETA: Change in effect when study removed
            dfbeta = loo_effect - base_effect
            dfbetas[i] = dfbeta

            # Influence as absolute standardized difference
            influence[i] = abs(dfbeta) / np.sqrt(base_result['tau_squared'] + base_result['se']**2)

            # Cook's distance analog
            cooks[i] = dfbeta**2

            # Leverage (hat values) - approximation
            w = 1 / self.variances[i]
            w_sum = np.sum(1 / self.variances)
            leverage.append(w / w_sum)

        # Identify influential studies
        # Criteria: |DFBETA| > 1 or top 10% of Cook's distance
        cook_threshold = np.percentile(list(cooks.values()), 90)
        influential = [i for i, d in cooks.items() if d > cook_threshold]

        return InfluenceAnalysisResult(
            study_influence=influence,
            influential_studies=influential,
            cook_distances=cooks,
            dfbetas=dfbetas,
            leverage=dict(zip(self.study_indices, leverage))
        )

    def cumulative_meta_analysis(
        self,
        order: Optional[List[int]] = None,
        method: str = "dl"
    ) -> CumulativeMetaResult:
        """
        Perform cumulative meta-analysis (scan statistic).

        :param order: Order of studies to add (None = by index)
        :param method: Pooling method
        :return: CumulativeMetaResult
        """
        if order is None:
            order = list(range(len(self.effects)))
        else:
            # Validate order
            if set(order) != set(self.study_indices):
                raise ValueError("Order must include all study indices")

        cumulative_effects = []
        cumulative_cis_lower = []
        cumulative_cis_upper = []
        z_stats = []
        study_ids_in_order = []

        effects_so_far = []
        variances_so_far = []

        for idx in order:
            effects_so_far.append(self.effects[idx])
            variances_so_far.append(self.variances[idx])

            result = self._pool_effects(
                np.array(effects_so_far),
                np.array(variances_so_far),
                method
            )

            cumulative_effects.append(result['pooled_effect'])
            cumulative_cis_lower.append(result['ci_lower'])
            cumulative_cis_upper.append(result['ci_upper'])

            # Z-statistic
            z = result['pooled_effect'] / result['se']
            z_stats.append(z)

            study_ids_in_order.append(idx)

        return CumulativeMetaResult(
            study_order=study_ids_in_order,
            cumulative_effects=cumulative_effects,
            cumulative_ci_lower=cumulative_cis_lower,
            cumulative_ci_upper=cumulative_cis_upper,
            z_statistics=z_stats,
            timestamp_studies=study_ids_in_order,
            final_estimate=cumulative_effects[-1],
            final_ci=(cumulative_cis_lower[-1], cumulative_cis_upper[-1])
        )

    def subgroup_sensitivity(
        self,
        subgroups: Dict[str, List[int]],
        subgroup_names: Optional[List[str]] = None
    ) -> SensitivityAnalysisResult:
        """
        Assess sensitivity to subgroup definitions.

        :param subgroups: Dictionary mapping subgroup names to study indices
        :param subgroup_names: Names for subgroups
        :return: SensitivityAnalysisResult
        """
        if subgroup_names is None:
            subgroup_names = list(subgroups.keys())

        results = []
        base_result = self._pool_effects(self.effects, self.variances, "dl")

        for name, indices in subgroups.items():
            if len(indices) < 2:
                continue

            sub_effects = self.effects[list(indices)]
            sub_variances = self.variances[list(indices)]

            result = self._pool_effects(sub_effects, sub_variances, "dl")
            result['subgroup'] = name
            result['n_studies'] = len(indices)
            result['diff_from_overall'] = result['pooled_effect'] - base_result['pooled_effect']

            results.append(result)

        # Check robustness
        diffs = [r['diff_from_overall'] for r in results]
        robust = all(abs(d) < 0.5 for d in diffs) if diffs else True

        return SensitivityAnalysisResult(
            analysis_type="subgroup",
            n_scenarios=len(results),
            base_case=base_result,
            results=[{'subgroup': r['subgroup'], 'result': r} for r in results],
            robust_to_exclusions=robust,
            outliers=[],
            influential_studies=[]
        )

    def contour_enhanced_plot(
        self,
        output_path: Optional[str] = None
    ) -> None:
        """
        Create contour-enhanced funnel plot.

        Shows how effect size changes with study precision and other characteristics.
        """
        import matplotlib.pyplot as plt
        from matplotlib import cm

        # Calculate effects and precisions
        effects = self.effects
        se = np.sqrt(self.variances)
        precision = 1 / se

        # Create grid
        n_grid = 100
        x = np.linspace(precision.min() * 0.9, precision.max() * 1.1, n_grid)
        y = np.linspace(effects.min() - 1, effects.max() + 1, n_grid)
        X, Y = np.meshgrid(x, y)

        # Compute kernel density estimate
        from scipy.stats import gaussian_kde
        if len(effects) > 1:
            kde = gaussian_kde(np.vstack([precision, effects]))
            Z = kde(np.vstack([X.ravel(), Y.ravel()]))
            Z = Z.reshape(X.shape)
        else:
            Z = np.zeros_like(X)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))

        contour = ax.contourf(X, Y, Z, levels=20, cmap=cm.Blues, alpha=0.6)
        ax.colorbar(contour, label='Density')

        ax.scatter(precision, effects, s=100, alpha=0.8, edgecolors='black', linewidths=1.5)
        ax.axvline(x=1/se.mean(), color='red', linestyle='--', alpha=0.5, label='Mean precision')
        ax.axhline(y=np.mean(effects), color='green', linestyle='--', alpha=0.5, label='Mean effect')

        ax.set_xlabel('Precision (1/SE)')
        ax.set_ylabel('Effect Size')
        ax.set_title('Contour-Enhanced Funnel Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
        else:
            plt.show()

    def generate_sensitivity_report(
        self,
        results: Dict[str, Union[pd.DataFrame, Dict]]
    ) -> str:
        """
        Generate comprehensive sensitivity analysis report.

        :param results: Dictionary of all sensitivity analysis results
        :return: Formatted report string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("SENSITIVITY ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Leave-one-out summary
        if 'leave_one_out' in results:
            loo = results['leave_one_out']
            lines.append("LEAVE-ONE-OUT ANALYSIS")
            lines.append("-" * 80)
            lines.append(f"N studies: {len(loo)}")
            lines.append(f"Effect range: [{loo['pooled_effect'].min():.4f}, {loo['pooled_effect'].max():.4f}]")
            lines.append("")

        # Influence analysis
        if 'influence' in results:
            inf = results['influence']
            lines.append("INFLUENCE ANALYSIS")
            lines.append("-" * 80)
            lines.append(f"Influential studies: {inf.influential_studies}")
            lines.append("")

        # Cumulative analysis
        if 'cumulative' in results:
            cum = results['cumulative']
            lines.append("CUMULATIVE META-ANALYSIS")
            lines.append("-" * 80)
            lines.append(f"Final estimate: {cum.final_estimate:.4f}")
            lines.append(f"Final CI: ({cum.final_ci[0]:.4f}, {cum.final_ci[1]:.4f})")
            lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)


def create_regression_dataframe(
    effects: np.ndarray,
    variances: np.ndarray,
    covariate_data: Dict[str, List]
) -> pd.DataFrame:
    """
    Create DataFrame suitable for meta-regression.

    :param effects: Effect sizes
    :param variances: Variances
    :param covariate_data: Dictionary of covariates
    :return: DataFrame
    """
    data = {
        'effect': effects,
        'variance': variances
    }
    data.update(covariate_data)
    return pd.DataFrame(data)


if __name__ == "__main__":
    print("Meta-Regression and Sensitivity Analysis module loaded")
    print("Features:")
    print("  - Meta-regression with single/multiple covariates")
    print("  - Leave-one-out cross-validation")
    print("  - Influence analysis (Cook's distance, DFBETAS)")
    print("  - Cumulative meta-analysis")
    print("  - Subgroup sensitivity analysis")
    print("  - Contour-enhanced funnel plots")
    print("  - Comprehensive sensitivity reports")
