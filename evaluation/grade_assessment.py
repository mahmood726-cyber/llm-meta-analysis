"""
GRADE Assessment Module for Systematic Reviews

Implements GRADE (Grading of Recommendations Assessment, Development and Evaluation)
approach to assessing certainty of evidence.

Reference:
- Guyatt et al. (2011) GRADE guidelines
- Balshem et al. (2011) GRADE guidelines
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum


class CertaintyLevel(Enum):
    """GRADE certainty of evidence levels"""
    HIGH = "High"
    MODERATE = "Moderate"
    LOW = "Low"
    VERY_LOW = "Very Low"


class StudyDesign(Enum):
    """Study design types"""
    RCT = "Randomized Trial"
    OBSERVATIONAL = "Observational Study"
    DIAGNOSTIC_ACCURACY = "Diagnostic Accuracy Study"


@dataclass
class GRADEDomain:
    """Individual GRADE domain assessment"""
    domain: str
    rating: str  # 'serious', 'very_serious', 'not_serious', 'no_downgrade'
    reason: str
    downgrade_levels: int  # 0, 1, or 2


@dataclass
class GRADEAssessment:
    """Complete GRADE assessment"""
    outcome_name: str
    study_design: StudyDesign
    n_studies: int
    n_participants: int
    effect_estimate: Optional[float]
    effect_ci: Optional[Tuple[float, float]]
    certainty_level: CertaintyLevel
    domains: List[GRADEDomain] = field(default_factory=list)
    summary: str = ""
    requires_update: bool = False


class GRADEAssessor:
    """
    GRADE assessment for systematic reviews and meta-analyses.

    Domains that can downgrade certainty:
    1. Risk of bias
    2. Inconsistency
    3. Indirectness
    4. Imprecision
    5. Publication bias

    Domains that can upgrade certainty (for observational studies):
    1. Large effect
    2. Dose-response gradient
    3. Plausible confounding
    """

    # Criteria for each domain
    INCONSISTENCY_CRITERIA = {
        'i2_low': 25,      # I² < 25%: no concerns
        'i2_moderate': 50,  # I² 25-50%: serious concern
        'i2_substantial': 75,  # I² 50-75%: very serious concern
        'i2_considerable': 100,  # I² > 75%: very serious concern
    }

    PRECISION_CRITERIA = {
        'optimal_information_size_multiplier': 2,  # OIS = 2 * total sample size
        'ci_width_threshold': 0.75,  # CI width > 0.75 suggests imprecision
    }

    def __init__(self):
        self.assessments = []

    def assess_risk_of_bias(
        self,
        risk_of_bias_data: Dict[str, Dict[str, str]]
    ) -> GRADEDomain:
        """
        Assess risk of bias domain.

        :param risk_of_bias_data: Dictionary mapping study_id to bias assessment
        :return: GRADEDomain for risk of bias
        """
        # Count studies with different risk levels
        n_high_risk = sum(1 for bias in risk_of_bias_data.values()
                         if bias.get('overall', '').lower() in ['high', 'serious'])

        n_total = len(risk_of_bias_data)

        if n_total == 0:
            return GRADEDomain(
                domain="Risk of Bias",
                rating="not_serious",
                reason="No studies available",
                downgrade_levels=0
            )

        proportion_high_risk = n_high_risk / n_total

        if proportion_high_risk > 0.5:
            # More than 50% high risk - very serious concern
            return GRADEDomain(
                domain="Risk of Bias",
                rating="very_serious",
                reason=f"{n_high_risk}/{n_total} studies at high risk of bias",
                downgrade_levels=2
            )
        elif proportion_high_risk > 0.2:
            # More than 20% high risk - serious concern
            return GRADEDomain(
                domain="Risk of Bias",
                rating="serious",
                reason=f"{n_high_risk}/{n_total} studies at high risk of bias",
                downgrade_levels=1
            )
        else:
            return GRADEDomain(
                domain="Risk of Bias",
                rating="not_serious",
                reason=f"Only {n_high_risk}/{n_total} studies at high risk of bias",
                downgrade_levels=0
            )

    def assess_inconsistency(
        self,
        i_squared: float,
        i_squared_ci: Optional[Tuple[float, float]] = None,
        q_p_value: float = None,
        prediction_interval: Optional[Tuple[float, float]] = None,
        direction_of_effects: Optional[List[str]] = None
    ) -> GRADEDomain:
        """
        Assess inconsistency domain.

        :param i_squared: I² statistic
        :param i_squared_ci: Confidence interval for I²
        :param q_p_value: P-value for Q statistic
        :param prediction_interval: Prediction interval
        :param direction_of_effects: List showing direction of effects in each study
        :return: GRADEDomain for inconsistency
        """
        downgrade = 0
        reasons = []

        # Assess based on I²
        if i_squared < self.INCONSISTENCY_CRITERIA['i2_moderate']:
            reasons.append("Low heterogeneity (I² < 25%)")
        elif i_squared < self.INCONSISTENCY_CRITERIA['i2_substantial']:
            downgrade += 1
            reasons.append(f"Moderate heterogeneity (I² = {i_squared:.1f}%)")
        elif i_squared < self.INCONSISTENCY_CRITERIA['i2_considerable']:
            downgrade += 1
            reasons.append(f"Substantial heterogeneity (I² = {i_squared:.1f}%)")
        else:
            downgrade = 2
            reasons.append(f"Considerable heterogeneity (I² = {i_squared:.1f}%)")

        # Check for inconsistency of effects
        if direction_of_effects and len(direction_of_effects) > 1:
            positive = sum(1 for d in direction_of_effects if d == "positive")
            negative = sum(1 for d in direction_of_effects if d == "negative")

            # If directions disagree, additional downgrading may be warranted
            if positive > 0 and negative > 0:
                if downgrade < 2:
                    downgrade += 1
                reasons.append("Inconsistency in direction of effects")

        # Check prediction interval
        if prediction_interval:
            # If prediction interval includes both benefit and harm, serious concern
            if prediction_interval[0] < 0 and prediction_interval[1] > 0:
                if downgrade < 2:
                    downgrade = 2
                reasons.append("Prediction interval includes both benefit and harm")

        # Determine rating
        if downgrade == 0:
            rating = "not_serious"
        elif downgrade == 1:
            rating = "serious"
        else:
            rating = "very_serious"

        return GRADEDomain(
            domain="Inconsistency",
            rating=rating,
            reason="; ".join(reasons),
            downgrade_levels=downgrade
        )

    def assess_indirectness(
        self,
        population_match: str = "direct",
        intervention_match: str = "direct",
        outcome_match: str = "direct",
        comparator_match: str = "direct"
    ) -> GRADEDomain:
        """
        Assess indirectness domain.

        :param population_match: How well population matches PICO question
        :param intervention_match: How well intervention matches PICO question
        :param outcome_match: How well outcome matches PICO question
        :param comparator_match: How well comparator matches PICO question
        :return: GRADEDomain for indirectness
        """
        downgrade = 0
        reasons = []

        # Check each PICO element
        pico_elements = {
            'Population': population_match,
            'Intervention': intervention_match,
            'Outcome': outcome_match,
            'Comparator': comparator_match
        }

        indirect_count = sum(1 for v in pico_elements.values() if v.lower() != 'direct')

        if indirect_count == 0:
            return GRADEDomain(
                domain="Indirectness",
                rating="not_serious",
                reason="Direct evidence across all PICO elements",
                downgrade_levels=0
            )

        # Assess seriousness based on which elements are indirect
        critical_elements = ['Population', 'Intervention', 'Outcome']
        serious_indirect = sum(1 for elem in critical_elements
                             if pico_elements.get(elem, '').lower() != 'direct')

        if serious_indirect >= 2:
            downgrade = 2
            reasons.append(f"{serious_indirect} critical PICO elements are indirect")
        elif serious_indirect == 1:
            downgrade = 1
            reasons.append(f"{serious_indirect} critical PICO element is indirect")
        else:
            downgrade = 1
            reasons.append(f"{indirect_count} non-critical PICO elements are indirect")

        rating = "very_serious" if downgrade == 2 else "serious"

        return GRADEDomain(
            domain="Indirectness",
            rating=rating,
            reason="; ".join(reasons),
            downgrade_levels=downgrade
        )

    def assess_imprecision(
        self,
        n_participants: int,
        effect_ci: Tuple[float, float],
        optimal_information_size: Optional[int] = None,
        min_important_effect: Optional[float] = None
    ) -> GRADEDomain:
        """
        Assess imprecision domain.

        :param n_participants: Total number of participants
        :param effect_ci: Confidence interval (lower, upper)
        :param optimal_information_size: OIS for this outcome
        :param min_important_effect: Minimal clinically important effect
        :return: GRADEDomain for imprecision
        """
        downgrade = 0
        reasons = []

        # Calculate OIS if not provided
        if optimal_information_size is None:
            # Rule of thumb: OIS = 2 * total sample size
            optimal_information_size = self.PRECISION_CRITERIA['optimal_information_size_multiplier'] * n_participants

        # Check if total sample size < OIS
        if n_participants < optimal_information_size:
            downgrade += 1
            reasons.append(f"Sample size ({n_participants}) below optimal information size ({optimal_information_size})")

        # Check if CI crosses null
        if effect_ci[0] < 0 < effect_ci[1]:
            # CI crosses null - assess if this is problematic
            if min_important_effect:
                # Check if CI excludes important effect
                if abs(effect_ci[0]) < min_important_effect and abs(effect_ci[1]) < min_important_effect:
                    downgrade += 1
                    reasons.append("Confidence interval excludes clinically important effect")
            else:
                # CI crosses null and no min effect specified - serious concern
                downgrade += 1
                reasons.append("Confidence interval crosses null effect")

        # Check CI width
        ci_width = abs(effect_ci[1] - effect_ci[0])
        if ci_width > self.PRECISION_CRITERIA['ci_width_threshold']:
            if downgrade < 1:
                downgrade = 1
            reasons.append(f"Wide confidence interval (width = {ci_width:.2f})")

        # Determine rating
        if downgrade == 0:
            rating = "not_serious"
        elif downgrade == 1:
            rating = "serious"
        else:
            rating = "very_serious"

        return GRADEDomain(
            domain="Imprecision",
            rating=rating,
            reason="; ".join(reasons),
            downgrade_levels=downgrade
        )

    def assess_publication_bias(
        self,
        egger_p_value: Optional[float] = None,
        beggs_p_value: Optional[float] = None,
        n_missing_studies: Optional[int] = None,
        funnel_plot_asymmetry: Optional[str] = None
    ) -> GRADEDomain:
        """
        Assess publication bias domain.

        :param egger_p_value: P-value from Egger's test
        :param beggs_p_value: P-value from Begg's test
        :param n_missing_studies: Number of missing studies from trim-and-fill
        :param funnel_plot_asymmetry: Assessment of funnel plot asymmetry
        :return: GRADEDomain for publication bias
        """
        downgrade = 0
        reasons = []

        # Check statistical tests
        if egger_p_value and egger_p_value < 0.1:
            downgrade += 1
            reasons.append(f"Egger's test suggests asymmetry (p = {egger_p_value:.3f})")

        if beggs_p_value and beggs_p_value < 0.1:
            if downgrade < 2:
                downgrade += 1
            reasons.append(f"Begg's test suggests asymmetry (p = {beggs_p_value:.3f})")

        # Check trim-and-fill
        if n_missing_studies and n_missing_studies > 0:
            proportion_missing = n_missing_studies / max(10, n_missing_studies)  # Normalize
            if proportion_missing > 0.2:
                if downgrade < 2:
                    downgrade = 2
                reasons.append(f"Trim-and-fill suggests {n_missing_studies} missing studies")
            elif downgrade < 1:
                downgrade = 1
                reasons.append(f"Trim-and-fill suggests {n_missing_studies} missing studies")

        # Check funnel plot assessment
        if funnel_plot_asymmetry:
            if "strong" in funnel_plot_asymmetry.lower():
                if downgrade < 2:
                    downgrade = 2
                reasons.append(f"Funnel plot shows {funnel_plot_asymmetry}")
            elif "moderate" in funnel_plot_asymmetry.lower():
                if downgrade < 1:
                    downgrade = 1
                reasons.append(f"Funnel plot shows {funnel_plot_asymmetry}")

        # Determine rating
        if downgrade == 0:
            rating = "not_serious"
            reason = "No clear evidence of publication bias"
        elif downgrade == 1:
            rating = "serious"
            reason = "; ".join(reasons)
        else:
            rating = "very_serious"
            reason = "; ".join(reasons)

        return GRADEDomain(
            domain="Publication Bias",
            rating=rating,
            reason=reason,
            downgrade_levels=downgrade
        )

    def assess_large_effect(
        self,
        effect_estimate: float,
        effect_ci: Tuple[float, float],
        minimal_important_effect: float
    ) -> GRADEDomain:
        """
        Assess large effect (for upgrading observational studies).

        :param effect_estimate: Point estimate (e.g., risk ratio, SMD)
        :param effect_ci: Confidence interval
        :param minimal_important_effect: Threshold for important effect
        :return: GRADEDomain for large effect (upgrade domain)
        """
        # Criteria for large effect:
        # 1. Point estimate > 2 * minimal important effect (RR > 2 or RR < 0.5)
        # 2. Confidence interval excludes 1 (RR) or 0 (SMD)

        # For risk ratios (log scale)
        if effect_estimate > 1:
            large_effect = effect_estimate >= 2
            ci_excludes_null = effect_ci[0] > 1
        else:
            large_effect = effect_estimate <= 0.5
            ci_excludes_null = effect_ci[1] < 1

        if large_effect and ci_excludes_null:
            return GRADEDomain(
                domain="Large Effect",
                rating="upgrade",
                reason=f"Large effect (estimate = {effect_estimate:.2f}) with CI excluding null",
                downgrade_levels=-1  # Negative means upgrade
            )
        elif large_effect:
            return GRADEDomain(
                domain="Large Effect",
                rating="potential_upgrade",
                reason=f"Large effect (estimate = {effect_estimate:.2f}) but CI includes null",
                downgrade_levels=0
            )
        else:
            return GRADEDomain(
                domain="Large Effect",
                rating="no_upgrade",
                reason="Effect size not large enough to warrant upgrading",
                downgrade_levels=0
            )

    def calculate_certainty(
        self,
        study_design: StudyDesign,
        domains: List[GRADEDomain]
    ) -> CertaintyLevel:
        """
        Calculate overall GRADE certainty level.

        :param study_design: Type of study design
        :param domains: List of domain assessments
        :return: CertaintyLevel
        """
        # Start with base certainty
        if study_design == StudyDesign.RCT:
            certainty = CertaintyLevel.HIGH
        else:
            certainty = CertaintyLevel.LOW

        # Apply downgrades
        total_downgrade = sum(d.downgrade_levels for d in domains if d.downgrade_levels > 0)

        # Apply upgrades (for observational studies only)
        total_upgrade = sum(abs(d.downgrade_levels) for d in domains if d.downgrade_levels < 0)

        if study_design != StudyDesign.RCT:
            # Can upgrade observational studies
            certainty = self._apply_certainty_steps(
                CertaintyLevel.LOW,
                upgrade_steps=min(total_upgrade, 2)  # Max 2 upgrades
            )

        # Apply downgrades
        final_certainty = self._apply_certainty_steps(
            certainty,
            downgrade_steps=total_downgrade
        )

        return final_certainty

    def _apply_certainty_steps(
        self,
        starting_level: CertaintyLevel,
        downgrade_steps: int,
        upgrade_steps: int = 0
    ) -> CertaintyLevel:
        """Apply downgrade/upgrade steps to certainty level"""
        levels = [
            CertaintyLevel.HIGH,
            CertaintyLevel.MODERATE,
            CertaintyLevel.LOW,
            CertaintyLevel.VERY_LOW
        ]

        current_idx = levels.index(starting_level)

        if downgrade_steps > 0:
            current_idx = min(len(levels) - 1, current_idx + downgrade_steps)

        if upgrade_steps > 0:
            current_idx = max(0, current_idx - upgrade_steps)

        return levels[current_idx]

    def full_assessment(
        self,
        outcome_name: str,
        study_design: StudyDesign,
        n_studies: int,
        n_participants: int,
        effect_estimate: Optional[float],
        effect_ci: Optional[Tuple[float, float]],
        risk_of_bias_data: Optional[Dict[str, Dict[str, str]]] = None,
        heterogeneity_data: Optional[Dict] = None,
        indirectness_data: Optional[Dict[str, str]] = None,
        publication_bias_data: Optional[Dict] = None
    ) -> GRADEAssessment:
        """
        Perform full GRADE assessment for an outcome.

        :param outcome_name: Name of the outcome
        :param study_design: Study design type
        :param n_studies: Number of studies
        :param n_participants: Total participants
        :param effect_estimate: Pooled effect estimate
        :param effect_ci: Confidence interval
        :param risk_of_bias_data: Risk of bias data for each study
        :param heterogeneity_data: Heterogeneity statistics (I², Q, etc.)
        :param indirectness_data: PICO element matching
        :param publication_bias_data: Publication bias test results
        :return: GRADEAssessment object
        """
        domains = []

        # Risk of bias
        if risk_of_bias_data:
            rob_domain = self.assess_risk_of_bias(risk_of_bias_data)
            domains.append(rob_domain)

        # Inconsistency
        if heterogeneity_data:
            inc_domain = self.assess_inconsistency(
                i_squared=heterogeneity_data.get('i_squared', 0),
                i_squared_ci=heterogeneity_data.get('i_squared_ci'),
                q_p_value=heterogeneity_data.get('q_p_value'),
                prediction_interval=heterogeneity_data.get('prediction_interval'),
                direction_of_effects=heterogeneity_data.get('direction_of_effects')
            )
            domains.append(inc_domain)

        # Indirectness
        if indirectness_data:
            ind_domain = self.assess_indirectness(**indirectness_data)
            domains.append(ind_domain)

        # Imprecision
        if effect_ci:
            imp_domain = self.assess_imprecision(
                n_participants=n_participants,
                effect_ci=effect_ci,
                min_important_effect=heterogeneity_data.get('min_important_effect') if heterogeneity_data else None
            )
            domains.append(imp_domain)

        # Publication bias
        if publication_bias_data:
            pb_domain = self.assess_publication_bias(**publication_bias_data)
            domains.append(pb_domain)

        # Calculate certainty
        certainty = self.calculate_certainty(study_design, domains)

        # Generate summary
        summary = self._generate_summary(certainty, domains, study_design)

        return GRADEAssessment(
            outcome_name=outcome_name,
            study_design=study_design,
            n_studies=n_studies,
            n_participants=n_participants,
            effect_estimate=effect_estimate,
            effect_ci=effect_ci,
            certainty_level=certainty,
            domains=domains,
            summary=summary
        )

    def _generate_summary(
        self,
        certainty: CertaintyLevel,
        domains: List[GRADEDomain],
        study_design: StudyDesign
    ) -> str:
        """Generate text summary of GRADE assessment"""
        summary_parts = [f"Certainty: {certainty.value}"]

        for domain in domains:
            if domain.downgrade_levels > 0:
                summary_parts.append(
                    f"- {domain.domain}: {domain.rating} ({domain.reason})"
                )
            elif domain.downgrade_levels < 0:
                summary_parts.append(
                    f"- {domain.domain}: Upgrade ({domain.reason})"
                )

        return "\n".join(summary_parts)

    def create_summary_table(
        self,
        assessments: List[GRADEAssessment]
    ) -> pd.DataFrame:
        """
        Create GRADE summary table (Evidence Profile).

        :param assessments: List of GRADE assessments
        :return: DataFrame with evidence profile
        """
        rows = []

        for assessment in assessments:
            row = {
                'Outcome': assessment.outcome_name,
                'Study Design': assessment.study_design.value,
                'N Studies': assessment.n_studies,
                'N Participants': assessment.n_participants,
                'Effect': f"{assessment.effect_estimate:.2f}" if assessment.effect_estimate else "N/A",
                'Certainty': assessment.certainty_level.value
            }

            # Add domain ratings
            for domain in assessment.domains:
                if domain.downgrade_levels > 0:
                    symbol = "-" * domain.downgrade_levels
                elif domain.downgrade_levels < 0:
                    symbol = "+" * abs(domain.downgrade_levels)
                else:
                    symbol = "⊝"

                row[domain.domain] = symbol

            rows.append(row)

        return pd.DataFrame(rows)

    def create_grade_summary_of_findings(
        self,
        assessments: List[GRADEAssessment]
    ) -> str:
        """
        Create Summary of Findings table in GRADE format.

        :param assessments: List of GRADE assessments
        :return: Formatted string for Summary of Findings
        """
        lines = []
        lines.append("Summary of Findings")
        lines.append("=" * 80)
        lines.append("")

        for assessment in assessments:
            lines.append(f"Outcome: {assessment.outcome_name}")
            lines.append(f"Certainty: {assessment.certainty_level.value}")
            lines.append(f"N Studies: {assessment.n_studies}, N Participants: {assessment.n_participants}")

            if assessment.effect_estimate:
                effect = assessment.effect_estimate
                ci = assessment.effect_ci
                lines.append(f"Effect: {effect:.2f} (95% CI: {ci[0]:.2f} to {ci[1]:.2f})")

            lines.append("")
            lines.append("GRADE Assessment:")
            for domain in assessment.domains:
                if domain.downgrade_levels != 0:
                    lines.append(f"  {domain.domain}: {domain.reason}")

            lines.append("")
            lines.append("-" * 80)
            lines.append("")

        return "\n".join(lines)


if __name__ == "__main__":
    print("GRADE Assessment module loaded")
    print("Features:")
    print("  - Risk of bias assessment")
    print("  - Inconsistency assessment")
    print("  - Indirectness assessment")
    print("  - Imprecision assessment")
    print("  - Publication bias assessment")
    print("  - Large effect assessment (upgrading)")
    print("  - Full GRADE certainty calculation")
    print("  - Summary of Findings tables")
