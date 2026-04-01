"""
Multivariate Meta-Analysis Module

Implements multivariate meta-analysis for multiple correlated outcomes.
Handles within-study correlation and accounts for dependency structures.

References:
- van Houwelingen et al. (2002) Multivariate meta-analysis
- Riley et al. (2007) Multivariate meta-analysis of diagnostic tests
- Jackson et al. (2011) Multivariate meta-analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats
from scipy.stats import multivariate_normal
from scipy.linalg import pinvh


@dataclass
class MultivariateResult:
    """Results from multivariate meta-analysis"""
    n_studies: int
    n_outcomes: int
    pooled_effects: np.ndarray  # Vector of pooled effects
    covariance_matrix: np.ndarray  # Covariance matrix of pooled effects
    standard_errors: np.ndarray
    correlations: np.ndarray  # Correlation matrix
    within_study_correlation: np.ndarray
    between_study_covariance: np.ndarray
    chi_sq_test: float
    chi_sq_p_value: float
    wald_statistics: np.ndarray


@dataclass
class StudyMultivariateData:
    """Data for a single study with multiple outcomes"""
    study_id: str
    outcomes: Dict[str, np.ndarray]  # outcome_name -> effect estimates
    variances: Dict[str, np.ndarray]  # outcome_name -> variances
    sample_sizes: Dict[str, int]
    within_study_correlations: Optional[np.ndarray] = None  # Correlation matrix


class MultivariateMetaAnalyzer:
    """
    Multivariate meta-analysis for correlated outcomes.

    Handles:
    - Multiple outcomes from same studies
    - Within-study correlation
    - Between-study covariance
    - Joint inference
    - Subset analysis
    """

    def __init__(self):
        self.studies: List[StudyMultivariateData] = []
        self.outcome_names: List[str] = []
        self.result: Optional[MultivariateResult] = None

    def add_study(
        self,
        study_id: str,
        outcomes: Dict[str, float],
        variances: Dict[str, float],
        sample_sizes: Dict[str, int],
        correlation_matrix: Optional[np.ndarray] = None
    ) -> None:
        """
        Add a study with multiple outcomes.

        :param study_id: Study identifier
        :param outcomes: Dictionary mapping outcome names to effect estimates
        :param variances: Dictionary mapping outcome names to variances
        :param sample_sizes: Dictionary mapping outcome names to sample sizes
        :param correlation_matrix: Within-study correlation matrix (k x k)
        """
        # Convert to arrays for vectorized operations
        outcome_array = np.array([outcomes[name] for name in sorted(outcomes.keys())])
        variance_array = np.array([variances[name] for name in sorted(variances.keys())])
        sample_size_dict = sample_sizes

        study = StudyMultivariateData(
            study_id=study_id,
            outcomes={name: np.array([val]) for name, val in outcomes.items()},
            variances={name: np.array([var]) for name, var in variances.items()},
            sample_sizes=sample_size_dict,
            within_study_correlations=correlation_matrix
        )

        self.studies.append(study)

        # Update outcome names
        if not self.outcome_names:
            self.outcome_names = sorted(outcomes.keys())
        else:
            self.outcome_names = sorted(set(self.outcome_names) | set(outcomes.keys()))

    def estimate_within_study_correlation(
        self,
        assume_independence: bool = False
    ) -> np.ndarray:
        """
        Estimate or assume within-study correlation.

        :param assume_independence: If True, assume diagonal correlation matrix
        :return: Estimated correlation matrix
        """
        k = len(self.outcome_names)

        if assume_independence:
            return np.eye(k)

        # Try to extract from studies that provided correlations
        corrs = []
        for study in self.studies:
            if study.within_study_correlations is not None:
                corrs.append(study.within_study_correlations)

        if corrs:
            # Average the correlations
            return np.mean(corrs, axis=0)

        # Estimate from outcome correlations across studies
        # This is a simplified approach
        all_effects = []
        for name in self.outcome_names:
            effects = []
            for study in self.studies:
                if name in study.outcomes:
                    effects.append(study.outcomes[name][0])
            all_effects.append(effects)

        # Compute correlation matrix from outcome vectors
        if all(len(effects) == len(all_effects[0]) for effects in all_effects):
            corr_matrix = np.corrcoef(all_effects)
            return corr_matrix

        # Default: assume moderate correlation (0.5)
        corr_matrix = np.eye(k)
        corr_matrix[corr_matrix == 0] = 0.5
        return corr_matrix

    def analyze(
        self,
        method: str = "reml",
        assume_independence: bool = False
    ) -> MultivariateResult:
        """
        Perform multivariate meta-analysis.

        :param method: Estimation method ('reml', 'ml', 'fixed')
        :param assume_independence: Whether to assume within-study independence
        :return: MultivariateResult
        """
        k = len(self.outcome_names)
        n_studies = len(self.studies)

        if n_studies == 0:
            raise ValueError("No studies added")

        # Get within-study correlation
        within_corr = self.estimate_within_study_correlation(assume_independence)

        # Build the block diagonal covariance matrix for each study
        # and construct the design matrices

        # Extract data matrices
        Y = []  # Outcomes (n_studies x k)
        V_list = []  # Within-study covariances
        sample_sizes = []

        for study in self.studies:
            study_effects = []
            study_variances = []

            for name in self.outcome_names:
                if name in study.outcomes:
                    study_effects.append(study.outcomes[name][0])
                    study_variances.append(study.variances[name][0])
                else:
                    # Missing outcome - use imputation or skip
                    study_effects.append(np.nan)
                    study_variances.append(np.nan)

            if all(not np.isnan(v) for v in study_variances):
                Y.append(study_effects)

                # Build within-study covariance matrix
                D = np.diag(study_variances)
                R = within_corr
                V = np.sqrt(D) @ R @ np.sqrt(D)

                # Ensure positive definiteness
                V = (V + V.T) / 2
                eigenvals = np.linalg.eigvals(V)
                if np.any(eigenvals <= 0):
                    # Add ridge regularization
                    V = V + np.eye(k) * 1e-6

                V_list.append(V)

                # Get average sample size
                if study.sample_sizes:
                    sample_sizes.append(np.mean(list(study.sample_sizes.values())))

        Y = np.array(Y)
        n_studies_valid = len(Y)

        if n_studies_valid == 0:
            raise ValueError("No valid studies for analysis")

        if method == "fixed":
            # Fixed-effect multivariate meta-analysis
            return self._fixed_effect_analysis(Y, V_list, within_corr)
        else:
            # Random-effects using REML or ML
            return self._random_effects_analysis(Y, V_list, within_corr, method)

    def _fixed_effect_analysis(
        self,
        Y: np.ndarray,
        V_list: List[np.ndarray],
        within_corr: np.ndarray
    ) -> MultivariateResult:
        """Fixed-effect multivariate meta-analysis"""
        n = len(Y)
        k = Y.shape[1]

        # Inverse variance weights (block diagonal)
        W_blocks = [np.linalg.inv(V) for V in V_list]

        # Pooled effect: W * Y
        # Need to construct block diagonal W
        W_nk = np.zeros((n, n))
        row_idx = 0
        for W in W_blocks:
            size = W.shape[0]
            W_nk[row_idx:row_idx+size, row_idx:row_idx+size] = W
            row_idx += size

        # Reshape Y for matrix multiplication
        Y_long = Y.flatten(order='F')  # Column-major (Fortran-style)

        # Solve W * Y = beta
        try:
            beta_long = np.linalg.solve(W_nk, Y_long)
        except np.linalg.LinAlgError:
            # Use pseudoinverse if singular
            beta_long = np.linalg.lstsq(W_nk, Y_long, rcond=None)[0]

        beta = beta_long.reshape((k, k), order='F')[:k, 0]

        # Covariance of beta
        # Extract diagonal blocks corresponding to beta
        beta_cov = np.zeros((k, k))
        row_idx = 0
        for W_inv in W_blocks:
            size = W_inv.shape[0]
            # For the first outcome
            if row_idx == 0:
                beta_cov[0, 0] = 1 / W_inv[0, 0] if size == k else 1 / W_inv[0, 0]
            row_idx += size

        # Standard errors
        se = np.sqrt(np.diag(beta_cov))

        # Correlation matrix
        D = np.diag(se)
        D_inv = np.diag(1 / se)
        corr_matrix = D_inv @ beta_cov @ D_inv

        # Wald statistics
        wald = beta / se

        # Chi-square test (all effects = 0)
        if k > 1:
            chi_sq = beta @ np.linalg.pinv(beta_cov) @ beta
            p_value = 1 - stats.chi2.cdf(chi_sq, k)
        else:
            chi_sq = (beta / se)**2
            p_value = 2 * (1 - stats.norm.cdf(abs(wald[0])))

        return MultivariateResult(
            n_studies=n,
            n_outcomes=k,
            pooled_effects=beta,
            covariance_matrix=beta_cov,
            standard_errors=se,
            correlations=corr_matrix,
            within_study_correlation=within_corr,
            between_study_covariance=np.zeros((k, k)),
            chi_sq_test=chi_sq,
            chi_sq_p_value=p_value,
            wald_statistics=wald
        )

    def _random_effects_analysis(
        self,
        Y: np.ndarray,
        V_list: List[np.ndarray],
        within_corr: np.ndarray,
        method: str = "reml"
    ) -> MultivariateResult:
        """Random-effects multivariate meta-analysis using REML/ML"""
        n = len(Y)
        k = Y.shape[1]

        # Initialize between-study covariance
        Tau = np.eye(k) * 0.1  # Initial value

        # EM algorithm for estimating Tau
        max_iter = 100
        tolerance = 1e-6

        for iteration in range(max_iter):
            Tau_old = Tau.copy()

            # E-step: Compute weights with current Tau
            W_blocks = []
            for V in V_list:
                W = np.linalg.inv(V + Tau)
                W_blocks.append(W)

            # M-step: Update Tau
            # Compute residuals
            beta = self._compute_pooled_effect(Y, V_list, Tau)

            # Update Tau using method of moments
            sum_sq_residuals = np.zeros((k, k))
            for i, y in enumerate(Y):
                resid = y - beta
                outer = np.outer(resid, resid)
                sum_sq_residuals += outer

            # Method of moments estimator
            if method == "reml":
                # REML: adjust for degrees of freedom
                Tau_new = sum_sq_residuals / n
            else:
                # ML
                Tau_new = sum_sq_residuals / (n + 1)

            # Ensure positive definiteness
            Tau_new = (Tau_new + Tau_new.T) / 2
            eigenvals = np.linalg.eigvals(Tau_new)
            if np.any(eigenvals <= 0):
                Tau_new = Tau_new + np.eye(k) * 1e-6

            Tau = Tau_new

            # Check convergence
            if np.max(np.abs(Tau - Tau_old)) < tolerance:
                break

        # Final estimates with converged Tau
        beta = self._compute_pooled_effect(Y, V_list, Tau)

        # Compute covariance of beta
        # Total information = sum of (V_i + Tau)^(-1)
        total_info = np.zeros((k, k))
        for V in V_list:
            try:
                total_info += np.linalg.inv(V + Tau)
            except:
                continue

        beta_cov = np.linalg.pinv(total_info)
        se = np.sqrt(np.diag(beta_cov))

        # Correlation matrix
        D_inv = np.diag(1 / se)
        corr_matrix = D_inv @ beta_cov @ D_inv

        # Wald statistics
        wald = beta / se

        # Chi-square test
        if k > 1:
            chi_sq = beta @ np.linalg.pinv(beta_cov) @ beta
            p_value = 1 - stats.chi2.cdf(chi_sq, k)
        else:
            chi_sq = (beta / se)**2
            p_value = 2 * (1 - stats.norm.cdf(abs(wald[0])))

        return MultivariateResult(
            n_studies=n,
            n_outcomes=k,
            pooled_effects=beta,
            covariance_matrix=beta_cov,
            standard_errors=se,
            correlations=corr_matrix,
            within_study_correlation=within_corr,
            between_study_covariance=Tau,
            chi_sq_test=chi_sq,
            chi_sq_p_value=p_value,
            wald_statistics=wald
        )

    def _compute_pooled_effect(
        self,
        Y: np.ndarray,
        V_list: List[np.ndarray],
        Tau: np.ndarray
    ) -> np.ndarray:
        """Compute pooled effect given between-study covariance Tau"""
        k = Y.shape[1]

        # Inverse variance weights with Tau
        W_sum = np.zeros((k, k))
        weighted_sum = np.zeros(k)

        for y, V in zip(Y, V_list):
            W = np.linalg.inv(V + Tau)
            W_sum += W
            weighted_sum += W @ y

        beta = np.linalg.inv(W_sum) @ weighted_sum
        return beta

    def subset_analysis(
        self,
        outcome_subset: List[str]
    ) -> 'MultivariateMetaAnalyzer':
        """
        Create subset analysis for specific outcomes.

        :param outcome_subset: List of outcome names to include
        :return: New MultivariateMetaAnalyzer with subset of outcomes
        """
        subset_analyzer = MultivariateMetaAnalyzer()
        subset_analyzer.outcome_names = outcome_subset

        for study in self.studies:
            # Filter to requested outcomes
            filtered_outcomes = {k: v for k, v in study.outcomes.items() if k in outcome_subset}
            filtered_variances = {k: v for k, v in study.variances.items() if k in outcome_subset}
            filtered_sample_sizes = {k: v for k, v in study.sample_sizes.items() if k in outcome_subset}

            if filtered_outcomes:
                # Extract relevant part of correlation matrix
                if study.within_study_correlations is not None:
                    idx = [self.outcome_names.index(name) for name in outcome_subset]
                    corr_subset = study.within_study_correlations[np.ix_(idx, idx)]
                else:
                    corr_subset = None

                subset_analyzer.add_study(
                    study_id=study.study_id,
                    outcomes=filtered_outcomes,
                    variances=filtered_variances,
                    sample_sizes=filtered_sample_sizes,
                    correlation_matrix=corr_subset
                )

        return subset_analyzer

    def global_test(
        self,
        result: MultivariateResult
    ) -> Dict[str, Union[float, str]]:
        """
        Perform global test of whether any outcome shows an effect.

        :param result: Multivariate analysis result
        :return: Dictionary with test results
        """
        k = result.n_outcomes

        # Chi-square test from results
        chi_sq = result.chi_sq_test
        p_value = result.chi_sq_p_value

        return {
            "statistic": chi_sq,
            "df": k,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "interpretation": self._interpret_global_test(p_value, k)
        }

    def _interpret_global_test(self, p_value: float, df: int) -> str:
        """Interpret global test result"""
        if p_value < 0.001:
            return f"Strong evidence that at least one outcome differs from null (p < 0.001)"
        elif p_value < 0.01:
            return f"Very strong evidence against null (p < 0.01)"
        elif p_value < 0.05:
            return f"Significant evidence against null (p = {p_value:.4f})"
        elif p_value < 0.10:
            return f"Trend toward significance (p = {p_value:.4f})"
        else:
            return f"No significant evidence against null (p = {p_value:.4f})"


class LongitudinalMetaAnalyzer(MultivariateMetaAnalyzer):
    """
    Longitudinal meta-analysis for repeated measures.

    Extends multivariate analysis to handle time-course data
    with correlation across time points.
    """

    def __init__(self):
        super().__init__()
        self.time_points: List[float] = []

    def add_longitudinal_study(
        self,
        study_id: str,
        outcomes_over_time: Dict[float, Dict[str, float]],
        variances_over_time: Dict[float, Dict[str, float]],
        sample_size: int,
        time_structure: str = "exchangeable"
    ) -> None:
        """
        Add a study with repeated measures over time.

        :param study_id: Study identifier
        :param outcomes_over_time: Dict mapping time -> outcome_name -> effect
        :param variances_over_time: Dict mapping time -> outcome_name -> variance
        :param sample_size: Sample size
        :param time_structure: Correlation structure ('exchangeable', 'ar1', 'unstructured')
        """
        if not self.time_points:
            self.time_points = sorted(outcomes_over_time.keys())

        for t in self.time_points:
            if t not in outcomes_over_time:
                continue

            # Create correlation matrix for this time point
            # across all outcomes and previous time points

            outcomes = outcomes_over_time[t]
            variances = variances_over_time[t]

            self.add_study(
                study_id=f"{study_id}_t{t}",
                outcomes=outcomes,
                variances=variances,
                sample_sizes={name: sample_size for name in outcomes.keys()},
                correlation_matrix=None  # Build from time_structure
            )


def perform_multivariate_analysis(
    data: pd.DataFrame,
    outcome_cols: List[str],
    study_col: str = "study_id",
    effect_col: str = "effect",
    variance_col: str = "variance",
    sample_size_col: str = "n"
) -> MultivariateResult:
    """
    Convenience function for multivariate meta-analysis from DataFrame.

    :param data: DataFrame in long format
    :param outcome_cols: Columns that identify different outcomes
    :param study_col: Column with study IDs
    :param effect_col: Column with effect estimates
    :param variance_col: Column with variances
    :param sample_size_col: Column with sample sizes
    :return: MultivariateResult
    """
    analyzer = MultivariateMetaAnalyzer()

    # Group by study
    for study_id, group in data.groupby(study_col):
        outcomes = {}
        variances = {}
        sample_sizes = {}

        # Extract data for each outcome
        for _, row in group.iterrows():
            # Determine outcome identifier
            # This assumes data has columns to distinguish outcomes
            # Adapt based on actual data structure

            # Simple approach: use the first outcome column that exists
            for outcome in outcome_cols:
                if outcome in row and pd.notna(row[outcome]):
                    # Extract identifier
                    outcomes[outcome] = row[effect_col]
                    variances[outcome] = row[variance_col]
                    if sample_size_col in row:
                        sample_sizes[outcome] = int(row[sample_size_col])
                    break

        if outcomes:
            analyzer.add_study(
                study_id=str(study_id),
                outcomes=outcomes,
                variances=variances,
                sample_sizes=sample_sizes
            )

    return analyzer.analyze()


if __name__ == "__main__":
    print("Multivariate Meta-Analysis module loaded")
    print("Features:")
    print("  - Multiple correlated outcomes")
    print("  - Within-study correlation")
    print("  - Between-study covariance")
    print("  - Joint inference")
    print("  - Subset analysis")
    print("  - Longitudinal/repeated measures")
