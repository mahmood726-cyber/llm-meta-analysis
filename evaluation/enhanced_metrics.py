"""
Enhanced Metrics Calculator with Uncertainty Quantification

Implements rigorous statistical validation for extracted data following
Cochrane Handbook and Research Synthesis Methods standards.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from scipy import stats
from sklearn.metrics import f1_score, cohen_kappa_score

from statistical_framework import (
    UncertaintyQuantifier,
    UncertaintyInterval,
    EnhancedHeterogeneity,
    PublicationBiasAssessor
)


@dataclass
class ExtractionWithUncertainty:
    """Data extraction result with proper uncertainty quantification"""
    value: float
    standard_error: float
    confidence_interval: UncertaintyInterval
    is_missing: bool = False
    is_imputed: bool = False


@dataclass
class InterRaterReliability:
    """Inter-rater reliability statistics for chunked aggregation"""
    kappa: float
    kappa_se: float
    kappa_ci: UncertaintyInterval
    agreement_percentage: float
    n_observations: int
    n_raters: int
    interpretation: str


@dataclass
class ClinicalRelevance:
    """Clinical relevance assessment of extraction errors"""
    error_type: str
    clinical_impact: str  # 'critical', 'major', 'moderate', 'minor'
    potential_consequence: str
    requires_verification: bool


class EnhancedMetricsCalculator:
    """
    Enhanced metrics calculator with proper uncertainty quantification
    and clinical relevance assessment.
    """

    def __init__(self, task: str):
        self.task = task

    def calculate_extraction_with_uncertainty(
        self,
        actual_value: Any,
        predicted_value: Any,
        sample_size: Optional[int] = None,
        outcome_type: str = "binary"
    ) -> ExtractionWithUncertainty:
        """
        Calculate extraction metrics with proper uncertainty quantification.

        :param actual_value: Reference (actual) value
        :param predicted_value: Model-predicted value
        :param sample_size: Sample size for CI calculation
        :param outcome_type: Type of outcome ('binary' or 'continuous')
        :return: ExtractionWithUncertainty object
        """
        uq = UncertaintyQuantifier()

        # Handle missing values
        if actual_value in ["x", "unknown", None, ""] or predicted_value in ["x", "unknown", None, ""]:
            return ExtractionWithUncertainty(
                value=np.nan,
                standard_error=np.nan,
                confidence_interval=UncertaintyInterval(np.nan, np.nan, method="missing"),
                is_missing=True,
                is_imputed=False
            )

        try:
            if outcome_type == "binary":
                # For binary outcomes, calculate as proportion
                actual_float = float(actual_value)
                predicted_float = float(predicted_value)

                # If these are counts, need total for CI
                if sample_size and sample_size > 0:
                    actual_ci = uq.calculate_proportion_ci(int(actual_float), sample_size)
                    predicted_ci = uq.calculate_proportion_ci(int(predicted_float), sample_size)

                    # Calculate difference with uncertainty
                    comparison = uq.compare_with_uncertainty(
                        actual_float, actual_ci,
                        predicted_float, predicted_ci
                    )

                    return ExtractionWithUncertainty(
                        value=predicted_float,
                        standard_error=comparison["difference"] / 1.96 if not np.isnan(comparison["difference"]) else np.nan,
                        confidence_interval=predicted_ci,
                        is_missing=False,
                        is_imputed=False
                    )
                else:
                    # Without sample size, treat as direct comparison
                    return ExtractionWithUncertainty(
                        value=predicted_float,
                        standard_error=0,
                        confidence_interval=UncertaintyInterval(predicted_float, predicted_float),
                        is_missing=False
                    )

            else:  # continuous
                actual_float = float(actual_value)
                predicted_float = float(predicted_value)

                # For continuous outcomes, need SD and n for proper CI
                # Without additional information, use simple difference
                difference = abs(actual_float - predicted_float)

                return ExtractionWithUncertainty(
                    value=predicted_float,
                    standard_error=np.nan,
                    confidence_interval=UncertaintyInterval(predicted_float, predicted_float),
                    is_missing=False,
                    is_imputed=False
                )

        except (ValueError, TypeError):
            return ExtractionWithUncertainty(
                value=np.nan,
                standard_error=np.nan,
                confidence_interval=UncertaintyInterval(np.nan, np.nan),
                is_missing=True
            )

    def calculate_variance_weighted_metrics(
        self,
        data: List[Dict],
        actual_field: str,
        predicted_field: str,
        variance_field: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate variance-weighted accuracy metrics.

        :param data: List of data dictionaries
        :param actual_field: Field name for actual values
        :param predicted_field: Field name for predicted values
        :param variance_field: Optional field for variances
        :return: Dictionary with weighted metrics
        """
        values_actual = []
        values_predicted = []
        weights = []

        for item in data:
            try:
                actual = item.get(actual_field)
                predicted = item.get(predicted_field)

                if actual in ["x", "unknown", None, ""] or predicted in ["x", "unknown", None, ""]:
                    continue

                actual_val = float(actual)
                predicted_val = float(predicted)

                # Calculate weight (inverse of variance)
                if variance_field and variance_field in item:
                    var = float(item[variance_field])
                    if var > 0:
                        weight = 1 / var
                    else:
                        weight = 1.0
                else:
                    # Use equal weights if no variance provided
                    weight = 1.0

                values_actual.append(actual_val)
                values_predicted.append(predicted_val)
                weights.append(weight)

            except (ValueError, TypeError):
                continue

        if len(values_actual) == 0:
            return {"error": "No valid data points"}

        values_actual = np.array(values_actual)
        values_predicted = np.array(values_predicted)
        weights = np.array(weights)

        # Variance-weighted accuracy
        is_correct = (values_actual == values_predicted).astype(float)
        weighted_accuracy = np.sum(weights * is_correct) / np.sum(weights)

        # Standard error of weighted proportion
        n = len(weights)
        se_weighted = np.sqrt(weighted_accuracy * (1 - weighted_accuracy) / n)

        # Confidence interval
        uq = UncertaintyQuantifier()
        ci = uq.calculate_proportion_ci(int(weighted_accuracy * n), n)

        return {
            "weighted_accuracy": weighted_accuracy,
            "se": se_weighted,
            "ci_lower": ci.lower,
            "ci_upper": ci.upper,
            "n_studies": n,
            "total_weight": np.sum(weights)
        }

    def calculate_point_estimates_with_uncertainty(
        self,
        data: List[Dict],
        outcome_type: str = "binary"
    ) -> Dict[str, Any]:
        """
        Calculate point estimates (LOR, SMD) with proper uncertainty quantification.

        :param data: List of data dictionaries
        :param outcome_type: 'binary' or 'continuous'
        :return: Dictionary with point estimates and CIs
        """
        from utils import calculate_log_odds_ratio, calculate_standardized_mean_difference

        estimates = []
        variances = []

        if outcome_type == "binary":
            for item in data:
                try:
                    ie = item.get("intervention_events_output", item.get("intervention_events", "x"))
                    ce = item.get("comparator_events_output", item.get("comparator_events", "x"))
                    it = item.get("intervention_group_size_output", item.get("intervention_group_size", "x"))
                    ct = item.get("comparator_group_size_output", item.get("comparator_group_size", "x"))

                    if "x" in [ie, ce, it, ct]:
                        continue

                    lor, var_lor = calculate_log_odds_ratio(ie, ce, it, ct)
                    if lor is not None:
                        estimates.append(lor)
                        variances.append(var_lor)
                except:
                    continue

            metric_name = "log_odds_ratio"
            display_name = "Log Odds Ratio"

        else:  # continuous
            for item in data:
                try:
                    im = item.get("intervention_mean_output", item.get("intervention_mean", "x"))
                    cm = item.get("comparator_mean_output", item.get("comparator_mean", "x"))
                    isd = item.get("intervention_standard_deviation_output",
                                   item.get("intervention_standard_deviation", "x"))
                    csd = item.get("comparator_standard_deviation_output",
                                   item.get("comparator_standard_deviation", "x"))
                    igs = item.get("intervention_group_size_output",
                                   item.get("intervention_group_size", "x"))
                    cgs = item.get("comparator_group_size_output",
                                   item.get("comparator_group_size", "x"))

                    if "x" in [im, cm, isd, csd, igs, cgs]:
                        continue

                    smd, var_smd = calculate_standardized_mean_difference(im, cm, isd, csd, igs, cgs)
                    if smd is not None:
                        estimates.append(smd)
                        variances.append(var_smd)
                except:
                    continue

            metric_name = "standardized_mean_difference"
            display_name = "Standardized Mean Difference"

        if len(estimates) == 0:
            return {"error": "No valid estimates"}

        estimates = np.array(estimates)
        variances = np.array(variances)

        # Variance-weighted pooling
        uq = UncertaintyQuantifier()
        pooled, se, ci = uq.variance_weighted_mean(estimates, variances)

        # Calculate heterogeneity
        heterogeneity = EnhancedHeterogeneity.full_heterogeneity_assessment(
            estimates, variances
        )

        return {
            "metric_name": metric_name,
            "display_name": display_name,
            "n_studies": len(estimates),
            "pooled_estimate": pooled,
            "pooled_se": se,
            "pooled_ci_lower": ci.lower,
            "pooled_ci_upper": ci.upper,
            "individual_estimates": estimates.tolist(),
            "individual_variances": variances.tolist(),
            "heterogeneity": {
                "q": heterogeneity.q_statistic,
                "q_p": heterogeneity.q_p_value,
                "i2": heterogeneity.i_squared,
                "i2_ci_lower": heterogeneity.i_squared_ci.lower,
                "i2_ci_upper": heterogeneity.i_squared_ci.upper,
                "tau2": heterogeneity.tau_squared,
                "tau": heterogeneity.tau
            }
        }


class ImprovedChunkedAggregation:
    """
    Improved aggregation for chunked outputs with inter-rater reliability.

    Implements:
    - Fleiss' kappa for multiple raters
    - Majority voting with confidence weighting
    - Inter-rater reliability assessment
    """

    @staticmethod
    def aggregate_with_reliability(
        values_per_chunk: List[List[Any]],
        confidence_scores: Optional[List[float]] = None
    ) -> Tuple[Any, InterRaterReliability]:
        """
        Aggregate multiple chunked values with reliability assessment.

        :param values_per_chunk: List of value lists from each chunk
        :param confidence_scores: Optional confidence scores for each chunk
        :return: Tuple of (aggregated_value, reliability_stats)
        """
        # Flatten all values
        all_values = []
        for chunk_values in values_per_chunk:
            all_values.extend(chunk_values)

        # Get unique possible values
        unique_values = list(set(all_values))

        if len(unique_values) <= 1:
            # All chunks agree
            return unique_values[0] if unique_values else "x", InterRaterReliability(
                kappa=1.0,
                kappa_se=0.0,
                kappa_ci=UncertaintyInterval(1.0, 1.0),
                agreement_percentage=100.0,
                n_observations=len(all_values),
                n_raters=len(values_per_chunk),
                interpretation="Perfect agreement"
            )

        # Calculate Fleiss' kappa for multiple raters
        kappa_result = ImprovedChunkedAggregation._fleiss_kappa(
            values_per_chunk, unique_values
        )

        # Aggregate using weighted voting if confidence scores provided
        if confidence_scores:
            aggregated = ImprovedChunkedAggregation._weighted_voting(
                values_per_chunk, unique_values, confidence_scores
            )
        else:
            # Use mode (majority voting)
            from collections import Counter
            counts = Counter(all_values)
            aggregated = counts.most_common(1)[0][0]

        return aggregated, kappa_result

    @staticmethod
    def _fleiss_kappa(
        values_per_chunk: List[List[Any]],
        categories: List[Any]
    ) -> InterRaterReliability:
        """
        Calculate Fleiss' kappa for inter-rater reliability.

        :param values_per_chunk: List of value lists from each rater (chunk)
        :param categories: List of all possible categories
        :return: InterRaterReliability object
        """
        n = len(values_per_chunk)  # Number of raters
        k = len(categories)  # Number of categories
        n_items = max(len(v) for v in values_per_chunk)  # Number of items

        # Build assignment matrix
        category_map = {cat: i for i, cat in enumerate(categories)}

        # Count assignments per category per item
        assignments = np.zeros((n_items, k))

        for item_idx in range(n_items):
            for rater_idx, rater_values in enumerate(values_per_chunk):
                if item_idx < len(rater_values):
                    value = rater_values[item_idx]
                    if value in category_map:
                        assignments[item_idx, category_map[value]] += 1

        # Calculate Fleiss' kappa
        n_raters = n  # Each item rated by n raters (chunks)

        # Proportion of assignments for each category
        p_j = np.sum(assignments, axis=0) / (n_items * n_raters)

        # Proportion of agreement for each item
        p_i = (np.sum(assignments ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))

        # Overall agreement
        p_bar = np.mean(p_i)

        # Expected agreement
        p_e_bar = np.sum(p_j ** 2)

        # Fleiss' kappa
        if p_e_bar == 1:
            kappa = 1.0
        else:
            kappa = (p_bar - p_e_bar) / (1 - p_e_bar)

        # Standard error of kappa
        se_kappa = np.sqrt(2 * (1 - p_bar) ** 2 / (n_items * n_raters * (n_raters - 1))) if n_items > 1 else 0

        # Confidence interval
        z = 1.96
        ci_lower = max(-1, kappa - z * se_kappa)
        ci_upper = min(1, kappa + z * se_kappa)

        # Interpretation
        if kappa < 0:
            interpretation = "Poor agreement"
        elif kappa < 0.2:
            interpretation = "Slight agreement"
        elif kappa < 0.4:
            interpretation = "Fair agreement"
        elif kappa < 0.6:
            interpretation = "Moderate agreement"
        elif kappa < 0.8:
            interpretation = "Substantial agreement"
        else:
            interpretation = "Almost perfect agreement"

        # Overall agreement percentage
        agreement_pct = p_bar * 100

        return InterRaterReliability(
            kappa=kappa,
            kappa_se=se_kappa,
            kappa_ci=UncertaintyInterval(ci_lower, ci_upper),
            agreement_percentage=agreement_pct,
            n_observations=n_items,
            n_raters=n,
            interpretation=interpretation
        )

    @staticmethod
    def _weighted_voting(
        values_per_chunk: List[List[Any]],
        categories: List[Any],
        confidence_scores: List[float]
    ) -> Any:
        """
        Aggregate using confidence-weighted voting.

        :param values_per_chunk: List of value lists from each chunk
        :param categories: List of possible categories
        :param confidence_scores: Confidence scores for each chunk
        :return: Aggregated value
        """
        category_scores = {cat: 0.0 for cat in categories}

        for chunk_idx, chunk_values in enumerate(values_per_chunk):
            conf = confidence_scores[chunk_idx] if chunk_idx < len(confidence_scores) else 1.0
            for value in chunk_values:
                if value in category_scores:
                    category_scores[value] += conf

        return max(category_scores, key=category_scores.get)


class ClinicalRelevanceAssessor:
    """
    Assess clinical relevance of extraction errors.

    Categorizes errors based on potential impact on clinical decision-making.
    """

    ERROR_TAXONOMY = {
        # Critical errors - could change clinical decision
        "critical": [
            "reversed_direction",  # Effect direction reversed
            "events_exceed_total",  # Impossible values
            "negative_counts",  # Invalid negative counts
            "extreme_outlier",  # Values far outside plausible range
        ],
        # Major errors - likely to affect pooled estimate
        "major": [
            "large_value_error",  # Error > 50% of true value
            "wrong_category",  # Binary vs continuous mismatch
            "hallucinated_value",  # Value not in source text
        ],
        # Moderate errors - may affect precision
        "moderate": [
            "moderate_value_error",  # Error 20-50% of true value
            "rounding_difference",  # Rounding vs exact value
            "unknown_when_available",  # Model predicted "x" when value present
        ],
        # Minor errors - unlikely to affect conclusions
        "minor": [
            "small_value_error",  # Error < 20% of true value
            "formatting_difference",  # Decimal places, etc.
            "unknown_when_unavailable",  # Correctly identified missing data
        ]
    }

    @staticmethod
    def assess_error(
        actual_value: Any,
        predicted_value: Any,
        field_type: str,
        context: Optional[Dict] = None
    ) -> ClinicalRelevance:
        """
        Assess the clinical relevance of an extraction error.

        :param actual_value: True value
        :param predicted_value: Predicted value
        :param field_type: Type of field (events, mean, etc.)
        :param context: Additional context (sample size, etc.)
        :return: ClinicalRelevance object
        """
        # Check for critical errors
        try:
            actual = float(actual_value) if actual_value not in ["x", "unknown", None, ""] else None
            predicted = float(predicted_value) if predicted_value not in ["x", "unknown", None, ""] else None
        except (ValueError, TypeError):
            actual = None
            predicted = None

        # Critical: Impossible values
        if predicted is not None and predicted < 0 and field_type in ["events", "group_size", "mean", "n"]:
            return ClinicalRelevance(
                error_type="negative_counts",
                clinical_impact="critical",
                potential_consequence="Invalid negative value could cause calculation errors",
                requires_verification=True
            )

        # Check if predicted exceeds total
        if context and "group_size" in str(context):
            # Would need to extract group_size from context
            pass

        # Predicted unknown when available
        if actual is not None and predicted is None:
            return ClinicalRelevance(
                error_type="unknown_when_available",
                clinical_impact="moderate",
                potential_consequence="Missing data reduces precision of meta-analysis",
                requires_verification=False
            )

        # Calculate relative error if both values present
        if actual is not None and predicted is not None:
            if actual == 0:
                if predicted != 0:
                    rel_error = float('inf')
                else:
                    rel_error = 0
            else:
                rel_error = abs((predicted - actual) / actual)

            if rel_error > 0.5:
                impact = "major"
                consequence = f"Large error ({rel_error:.1%}) could significantly affect pooled estimate"
            elif rel_error > 0.2:
                impact = "moderate"
                consequence = f"Moderate error ({rel_error:.1%}) may affect precision"
            elif rel_error > 0.01:
                impact = "minor"
                consequence = f"Small error ({rel_error:.1%}) unlikely to affect conclusions"
            else:
                impact = "minor"
                consequence = "Negligible error (within rounding)"

            return ClinicalRelevance(
                error_type="value_error",
                clinical_impact=impact,
                potential_consequence=consequence,
                requires_verification=impact in ["major", "critical"]
            )

        # Default
        return ClinicalRelevance(
            error_type="unknown",
            clinical_impact="minor",
            potential_consequence="Unable to assess",
            requires_verification=False
        )


if __name__ == "__main__":
    print("Enhanced Metrics Calculator loaded successfully")
    print("Available classes:")
    print("  - EnhancedMetricsCalculator")
    print("  - ImprovedChunkedAggregation")
    print("  - ClinicalRelevanceAssessor")
