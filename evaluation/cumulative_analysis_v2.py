"""
Cumulative Meta-Analysis and Scan Statistics (Revised)

Revisions based on editorial feedback:
- Explicit validation of temporal data requirements
- Proper error handling when publication dates are missing
- Improved TSA boundary calculations
- Lan-DeMets alpha-spending function
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats
from scipy.stats import norm, chi2
import warnings


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


class CumulativeMetaAnalyzer:
    """
    Cumulative meta-analysis for monitoring evidence accumulation.

    Tracks how pooled estimates evolve as new studies are added,
    helping identify when conclusions become stable.
    """

    def __init__(self, require_temporal_data: bool = True):
        """
        Initialize cumulative analyzer.

        :param require_temporal_data: If True, fail when temporal data is missing
        """
        self.require_temporal_data = require_temporal_data
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

        # Validate temporal data requirement
        if order_by == "chronological":
            if publication_dates is None:
                if self.require_temporal_data:
                    raise ValueError(
                        "publication_dates must be provided for chronological ordering. "
                        "Either provide publication dates or use order_by='precision'. "
                        "Cumulative meta-analysis by date requires temporal information."
                    )
                else:
                    warnings.warn(
                        "publication_dates not provided for chronological ordering. "
                        "Defaulting to input order (this may not reflect true temporal sequence)."
                    )
                    publication_dates = [f"t{i}" for i in range(n_studies)]

            # Validate dates are not all identical
            if self._all_dates_identical(publication_dates):
                if self.require_temporal_data:
                    raise ValueError(
                        "All publication dates are identical or could not be parsed. "
                        "Cannot perform meaningful temporal analysis. "
                        "Ensure proper date formatting or use order_by='precision'."
                    )
                else:
                    warnings.warn(
                        "All publication dates appear identical. "
                        "Cumulative analysis may not reflect true temporal sequence."
                    )

        # Determine order
        if order_by == "chronological" and publication_dates is not None:
            order = self._parse_and_sort_dates(publication_dates)
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

    def _all_dates_identical(self, dates: List[str]) -> bool:
        """
        Check if all dates are identical or invalid.

        :param dates: List of date strings
        :return: True if all identical
        """
        # Try to parse as dates
        parsed_dates = []
        for d in dates:
            try:
                # Try common date formats
                if isinstance(d, str):
                    if d.isdigit():
                        parsed_dates.append(int(d))
                    else:
                        import datetime
                        parsed_dates.append(datetime.datetime.strptime(d, "%Y-%m-%d").year)
                else:
                    parsed_dates.append(d)
            except Exception:
                # If we can't parse, check if all are the same string
                parsed_dates.append(d)

        # Check if all identical
        if len(set(parsed_dates)) == 1:
            return True
        return False

    def _parse_and_sort_dates(self, dates: List[str]) -> np.ndarray:
        """
        Parse dates and return sorting indices.

        :param dates: List of date strings
        :return: Sorting indices
        """
        parsed = []

        for d in dates:
            try:
                # Try year format
                if isinstance(d, (int, float)):
                    parsed.append(int(d))
                elif d.isdigit():
                    parsed.append(int(d))
                else:
                    # Try datetime formats
                    import datetime
                    parsed_dates.append(datetime.datetime.strptime(d, "%Y-%m-%d"))
            except Exception:
                # If can't parse, use position
                parsed.append(len(parsed))

        return np.argsort(parsed)

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

        # Pooled effect (fixed effect for cumulative)
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
            sum_w2 = np.sum(weights**2)
            tau_squared = max(0, (q_statistic - df) / (sum_weights - sum_w2 / sum_weights))
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


class TrialSequentialAnalyzer:
    """
    Trial Sequential Analysis (TSA) with improved boundary calculations.

    Uses Lan-DeMets alpha-spending for sequential monitoring.
    """

    def __init__(self):
        """Initialize TSA analyzer"""
        self.boundaries: Optional[Dict] = None

    def calculate_information_size(
        self,
        effect_size: float,
        variance: float,
        alpha: float = 0.05,
        beta: float = 0.20,
        delta: float = None,
        ratio: float = 1.0
    ) -> float:
        """
        Calculate required information size (optimal sample size).

        :param effect_size: True effect size
        :param variance: Variance of effect size
        :param alpha: Type I error rate
        :param beta: Type II error rate
        :param delta: Minimal clinically important difference
        :param ratio: Ratio of sample sizes (intervention:control)
        :return: Required information size
        """
        if delta is None:
            delta = abs(effect_size)

        # Z values
        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(1 - beta)

        # Required information size
        # Based on Wetterslev et al. formula
        n_is = 4 * (z_alpha + z_beta)**2 * variance / (delta**2 * ratio)

        return n_is

    def construct_ld_boundaries(
        self,
        n_analyses: int,
        information_size: float,
        alpha: float = 0.05,
        beta: float = 0.20,
        spending_function: str = "obrien_fleming",
        side: str = "two-sided"
    ) -> Dict:
        """
        Construct Lan-DeMets sequential boundaries.

        :param n_analyses: Number of interim analyses
        :param information_size: Total required information
        :param alpha: Type I error
        :param beta: Type II error
        :param spending_function: Alpha-spending function
        :param side: 'two-sided' or 'one-sided'
        :return: Dictionary with boundaries
        """
        # Information fractions
        information_fractions = np.linspace(0, 1, n_analyses + 1)[1:]

        if side == "two-sided":
            alpha_total = alpha
        else:
            alpha_total = 2 * alpha

        # Alpha-spending function
        alpha_spent = np.zeros(n_analyses)

        for i, t in enumerate(information_fractions):
            if spending_function == "obrien_fleming":
                # O'Brien-Fleming type spending
                alpha_spent[i] = 2 * (1 - norm.ppf(1 - alpha_total/2 / np.sqrt(t)))
                alpha_spent[i] = min(alpha_spent[i], alpha_total)
            elif spending_function == "pocock":
                # Pocock type spending
                alpha_spent[i] = alpha_total * t
            elif spending_function == "power_family":
                # Power family: alpha * t^rho
                rho = 3  # Closer to O'Brien-Fleming
                alpha_spent[i] = alpha_total * (t ** rho)
            else:
                raise ValueError(f"Unknown spending function: {spending_function}")

        # Compute boundary values
        critical_values = []
        for i, t in enumerate(information_fractions):
            # Incremental alpha spent
            if i == 0:
                alpha_inc = alpha_spent[0]
            else:
                alpha_inc = alpha_spent[i] - alpha_spent[i-1]

            # Two-sided boundary
            z_boundary = norm.ppf(1 - alpha_inc/2)
            critical_values.append(z_boundary)

        # Futility boundaries (non-binding)
        z_beta = norm.ppf(1 - beta)
        futility_values = -z_beta / np.sqrt(information_fractions)

        # Adjusted boundary values (scaled by information fraction)
        if spending_function == "obrien_fleming":
            # O'Brien-Fleming: boundaries are more conservative early
            monitoring_boundaries = np.array(critical_values) / np.sqrt(information_fractions)
        else:
            monitoring_boundaries = critical_values

        self.boundaries = {
            "n_analyses": n_analyses,
            "information_fractions": information_fractions,
            "efficacy_boundaries": monitoring_boundaries,
            "futility_boundaries": futility_values,
            "alpha_spent": alpha_spent,
            "alpha_total": alpha_total,
            "spending_function": spending_function,
            "information_size": information_size
        }

        return self.boundaries

    def assess_trial_status(
        self,
        cumulative_z_score: float,
        information_fraction: float,
        boundary_type: str = "efficacy"
    ) -> str:
        """
        Assess trial status against boundaries.

        :param cumulative_z_score: Cumulative Z statistic
        :param information_fraction: Proportion of information accrued
        :param boundary_type: 'efficacy' or 'futility'
        :return: Status ('continue', 'efficacy', 'futility')
        """
        if self.boundaries is None:
            raise ValueError("Boundaries must be computed first")

        # Find nearest information fraction
        idx = np.argmin(np.abs(
            self.boundaries["information_fractions"] - information_fraction
        ))
        idx = max(0, min(idx, len(self.boundaries["efficacy_boundaries"]) - 1))

        if boundary_type == "efficacy":
            bound = self.boundaries["efficacy_boundaries"][idx]
            if abs(cumulative_z_score) >= bound:
                return "efficacy"
        elif boundary_type == "futility":
            bound = self.boundaries["futility_boundaries"][idx]
            if cumulative_z_score < bound:
                return "futility"

        return "continue"


if __name__ == "__main__":
    print("Cumulative Meta-Analysis (Revised) loaded")
    print("Features:")
    print("  - Explicit temporal data validation")
    print("  - Proper error handling for missing dates")
    print("  - Lan-DeMets alpha-spending boundaries")
    print("  - Multiple spending functions (O'Brien-Fleming, Pocock)")
