"""
Clinical Decision Support Integration

Integrates meta-analysis results with clinical decision making,
providing actionable recommendations based on evidence synthesis.

References:
- GRADE working group recommendations
- BMJ Clinical Evidence
- UpToDate methodology
- Institute of Medicine (IOM) standards
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings


class RecommendationStrength(Enum):
    """Strength of recommendation"""
    STRONG = "Strong"
    MODERATE = "Moderate"
    WEAK = "Weak"
    INSUFFICIENT = "Insufficient evidence"


class Direction(Enum):
    """Direction of recommendation"""
    FOR = "Recommend for"
    AGAINST = "Recommend against"
    NEUTRAL = "No recommendation"


class Certainty(Enum):
    """Certainty of evidence (GRADE)"""
    VERY_HIGH = "Very high"
    HIGH = "High"
    MODERATE = "Moderate"
    LOW = "Low"
    VERY_LOW = "Very low"


@dataclass
class ClinicalRecommendation:
    """Structured clinical recommendation"""
    population: str
    intervention: str
    comparator: str
    outcome: str
    recommendation_strength: RecommendationStrength
    direction: Direction
    certainty: Certainty
    effect_size: float
    confidence_interval: Tuple[float, float]
    absolute_risk_reduction: Optional[float]
    number_needed_to_treat: Optional[float]
    number_needed_to_harm: Optional[float]
    baseline_risk: float
    patient_values: str
    resource_use: str
    justification: str
    implementation_considerations: List[str]
    monitoring_recommendations: List[str]
    shared_decision_making_tips: List[str]


@dataclass
class BenefitHarmAssessment:
    """Assessment of benefits and harms"""
    outcome: str
    benefit_effect: float
    benefit_ci: Tuple[float, float]
    harm_effect: float
    harm_ci: Tuple[float, float]
    baseline_risk_benefit: float
    baseline_risk_harm: float
    arr_benefit: float  # Absolute risk reduction (benefit)
    arr_harm: float  # Absolute risk increase (harm)
    nnt: float  # Number needed to treat
    nnh: float  # Number needed to harm
    net_benefit: str  # "benefit", "harm", "uncertain"


@dataclass
class PatientProfile:
    """Individual patient characteristics for personalization"""
    age: int
    sex: str
    comorbidities: List[str]
    baseline_risk: float
    life_expectancy: float
    values_preferences: Dict[str, str]  # outcome -> importance
    contraindications: List[str]
    concurrent_medications: List[str]
    treatment_goals: List[str]


@dataclass
class PersonalizedRecommendation:
    """Personalized recommendation for specific patient"""
    recommendation: ClinicalRecommendation
    patient_profile: PatientProfile
    personalized_benefits: Dict[str, float]
    personalized_harms: Dict[str, float]
    individual_nnt: float
    individual_nnhs: List[float]  # Multiple harms
    decision_aid_score: float
    recommendation_for_patient: str
    discussion_points: List[str]
    uncertainty_explanation: str


class ClinicalDecisionSupportEngine:
    """
    Engine for generating clinical recommendations from meta-analysis.

    Follows GRADE methodology for developing recommendations.
    """

    def __init__(self):
        """Initialize decision support engine"""
        self.benefit_harm_thresholds = {
            "trivial": 0.01,  # < 1% absolute difference
            "small": 0.05,    # 1-5%
            "moderate": 0.10,  # 5-10%
            "large": 0.20     # > 20%
        }

        self.certainty_mappings = {
            "very_high": 1.0,
            "high": 0.9,
            "moderate": 0.7,
            "low": 0.5,
            "very_low": 0.3
        }

    def generate_recommendation(
        self,
        effect_size: float,
        confidence_interval: Tuple[float, float],
        p_value: float,
        certainty: str,
        baseline_risk: float,
        outcome_type: str,
        n_studies: int,
        n_participants: int,
        i2: float,
        serious_harms: Optional[List[str]] = None,
        cost_considerations: Optional[str] = None,
        patient_values: Optional[str] = None
    ) -> ClinicalRecommendation:
        """
        Generate clinical recommendation based on meta-analysis results.

        :param effect_size: Pooled effect estimate
        :param confidence_interval: 95% confidence interval
        :param p_value: Statistical significance
        :param certainty: GRADE certainty level
        :param baseline_risk: Baseline risk in control group
        :param outcome_type: Type of outcome ('binary', 'continuous', 'survival')
        :param n_studies: Number of studies
        :param n_participants: Total participants
        :param i2: Heterogeneity I²
        :param serious_harms: List of serious harms identified
        :param cost_considerations: Cost/resource considerations
        :param patient_values: Patient values and preferences
        :return: ClinicalRecommendation
        """
        # Determine direction
        if effect_size > 0:
            direction = Direction.FOR
        elif effect_size < 0:
            direction = Direction.AGAINST
        else:
            direction = Direction.NEUTRAL

        # Determine strength based on GRADE factors
        strength = self._determine_strength(
            effect_size, confidence_interval, certainty, p_value,
            n_studies, n_participants, i2, serious_harms
        )

        # Calculate absolute effects
        arr = None
        nnt = None
        nnh = None

        if outcome_type == "binary" and baseline_risk > 0:
            # Calculate ARR and NNT for binary outcomes
            if direction == Direction.FOR:
                # Risk ratio
                rr = np.exp(effect_size) if effect_size < 5 else effect_size
                intervention_risk = baseline_risk * rr
                arr = intervention_risk - baseline_risk

                if arr != 0:
                    nnt = abs(1 / arr)
            else:
                # Odds/risk ratio for harmful outcomes
                rr = np.exp(abs(effect_size)) if abs(effect_size) < 5 else abs(effect_size)
                harm_risk = baseline_risk * rr
                arr = harm_risk - baseline_risk

                if arr != 0:
                    nnh = abs(1 / arr)

        # Justification
        justification = self._generate_justification(
            effect_size, confidence_interval, certainty, strength,
            n_studies, n_participants, i2
        )

        # Implementation considerations
        implementation = self._generate_implementation_considerations(
            direction, strength, serious_harms, cost_considerations
        )

        # Monitoring recommendations
        monitoring = self._generate_monitoring_recommendations(
            direction, outcome_type, serious_harms
        )

        # Shared decision making tips
        sdm_tips = self._generate_sdm_tips(
            direction, certainty, outcome_type
        )

        return ClinicalRecommendation(
            population="Patients with the condition",
            intervention="The intervention",
            comparator="Standard care or placebo",
            outcome="The primary outcome",
            recommendation_strength=strength,
            direction=direction,
            certainty=Certainty[certainty.upper()] if certainty.upper() in Certainty.__members__ else Certainty.MODERATE,
            effect_size=effect_size,
            confidence_interval=confidence_interval,
            absolute_risk_reduction=arr,
            number_needed_to_treat=nnt,
            number_needed_to_harm=nnh,
            baseline_risk=baseline_risk,
            patient_values=patient_values or "Most patients value the outcome",
            resource_use=cost_considerations or "Not considered",
            justification=justification,
            implementation_considerations=implementation,
            monitoring_recommendations=monitoring,
            shared_decision_making_tips=sdm_tips
        )

    def _determine_strength(
        self,
        effect_size: float,
        ci: Tuple[float, float],
        certainty: str,
        p_value: float,
        n_studies: int,
        n_participants: int,
        i2: float,
        serious_harms: Optional[List[str]]
    ) -> RecommendationStrength:
        """Determine strength of recommendation"""
        # Start with certainty-based strength
        certainty_score = self.certainty_mappings.get(certainty.lower(), 0.5)

        # Assess evidence magnitude
        if abs(effect_size) < self.benefit_harm_thresholds["small"]:
            magnitude = "trivial"
        elif abs(effect_size) < self.benefit_harm_thresholds["moderate"]:
            magnitude = "small"
        elif abs(effect_size) < self.benefit_harm_thresholds["large"]:
            magnitude = "moderate"
        else:
            magnitude = "large"

        # Statistical significance
        is_significant = p_value < 0.05
        ci_crosses_null = ci[0] * ci[1] < 0

        # Heterogeneity
        high_heterogeneity = i2 > 50

        # Harms present
        has_serious_harms = serious_harms is not None and len(serious_harms) > 0

        # Determine strength
        if magnitude == "trivial" or not is_significant or ci_crosses_null:
            return RecommendationStrength.INSUFFICIENT

        if certainty_score >= 0.9 and magnitude in ["moderate", "large"] and not high_heterogeneity:
            if has_serious_harms:
                return RecommendationStrength.MODERATE
            else:
                return RecommendationStrength.STRONG

        if certainty_score >= 0.7 and magnitude in ["moderate", "large"]:
            if has_serious_harms:
                return RecommendationStrength.WEAK
            else:
                return RecommendationStrength.MODERATE

        if certainty_score >= 0.5 and magnitude == "moderate":
            if has_serious_harms:
                return RecommendationStrength.WEAK
            else:
                return RecommendationStrength.MODERATE

        if magnitude == "small":
            return RecommendationStrength.WEAK

        if high_heterogeneity and certainty_score < 0.7:
            return RecommendationStrength.WEAK

        return RecommendationStrength.MODERATE

    def _generate_justification(
        self,
        effect_size: float,
        ci: Tuple[float, float],
        certainty: str,
        strength: RecommendationStrength,
        n_studies: int,
        n_participants: int,
        i2: float
    ) -> str:
        """Generate justification for recommendation"""
        direction_text = "beneficial" if effect_size > 0 else "harmful"

        justification = f"""
Based on a meta-analysis of {n_studies} studies ({n_participants} participants),
the intervention shows a statistically {'significant' if min(ci) * max(ci) > 0 else 'non-significant'}
{direction_text} effect (effect size: {effect_size:.2f}, 95% CI: {ci[0]:.2f} to {ci[1]:.2f}).

The certainty of evidence is {certainty}. Heterogeneity is {'high' if i2 > 50 else 'moderate' if i2 > 25 else 'low'} (I² = {i2:.1f}%).
"""

        if strength == RecommendationStrength.STRONG:
            justification += "A strong recommendation is warranted because the benefits clearly outweigh harms in most patients."
        elif strength == RecommendationStrength.MODERATE:
            justification += "A moderate recommendation is made; most patients should choose this option, but alternative approaches may be reasonable for some."
        elif strength == RecommendationStrength.WEAK:
            justification += "A weak recommendation is made; the decision should be individualized based on patient preferences and context."
        else:
            justification += "Evidence is insufficient to make a recommendation; shared decision making is essential."

        return justification.strip()

    def _generate_implementation_considerations(
        self,
        direction: Direction,
        strength: RecommendationStrength,
        serious_harms: Optional[List[str]],
        cost: Optional[str]
    ) -> List[str]:
        """Generate implementation considerations"""
        considerations = []

        if direction == Direction.FOR:
            considerations.append("Ensure appropriate patient selection based on inclusion criteria from trials")
        else:
            considerations.append("Avoid this intervention in most patients")

        if strength == RecommendationStrength.WEAK:
            considerations.append("Shared decision making is particularly important")
            considerations.append("Consider patient values and preferences carefully")

        if serious_harms:
            considerations.append(f"Monitor for: {', '.join(serious_harms[:3])}")

        if cost:
            considerations.append(f"Cost considerations: {cost}")

        considerations.append("Reassess treatment response at appropriate intervals")

        return considerations

    def _generate_monitoring_recommendations(
        self,
        direction: Direction,
        outcome_type: str,
        serious_harms: Optional[List[str]]
    ) -> List[str]:
        """Generate monitoring recommendations"""
        monitoring = []

        if direction == Direction.FOR:
            monitoring.append("Monitor treatment response at regular intervals")
            monitoring.append("Assess for adverse effects at each visit")

        if serious_harms:
            monitoring.append("Specific monitoring for serious harms:")
            for harm in serious_harms[:3]:
                monitoring.append(f"  - {harm}")

        if outcome_type == "binary":
            monitoring.append("Track outcome events systematically")
        elif outcome_type == "continuous":
            monitoring.append("Use standardized measures to assess change over time")

        return monitoring

    def _generate_sdm_tips(
        self,
        direction: Direction,
        certainty: str,
        outcome_type: str
    ) -> List[str]:
        """Generate shared decision making tips"""
        tips = []

        tips.append("Discuss the certainty of evidence with the patient")
        tips.append("Explain the absolute benefits and harms in plain language")

        if direction == Direction.FOR:
            tips.append("Use decision aids to help patients understand trade-offs")

        if certainty in ["low", "very_low"]:
            tips.append("Acknowledge uncertainty in the evidence")
            tips.append("Consider reassessing as new evidence emerges")

        tips.append("Elicit patient values and preferences regarding outcomes")
        tips.append("Discuss patient's personal experience with similar treatments")

        return tips


class PersonalizedDecisionSupport:
    """
    Personalize recommendations based on individual patient characteristics.

    Adjusts population-level evidence for individual patients.
    """

    def __init__(self):
        """Initialize personalized decision support"""
        self.engine = ClinicalDecisionSupportEngine()

    def personalize_for_patient(
        self,
        recommendation: ClinicalRecommendation,
        patient: PatientProfile,
        relative_effects: Dict[str, float],  # outcome -> relative effect
        baseline_risks: Dict[str, float]  # outcome -> baseline risk
    ) -> PersonalizedRecommendation:
        """
        Personalize recommendation for specific patient.

        :param recommendation: Population-level recommendation
        :param patient: Patient profile
        :param relative_effects: Relative effects for different outcomes
        :param baseline_risks: Patient-specific baseline risks
        :return: PersonalizedRecommendation
        """
        # Calculate personalized absolute effects
        personalized_benefits = {}
        personalized_harms = {}

        for outcome, rel_effect in relative_effects.items():
            baseline = baseline_risks.get(outcome, 0.1)
            arr = baseline * (rel_effect - 1)

            if arr > 0:
                personalized_benefits[outcome] = arr
            else:
                personalized_harms[outcome] = abs(arr)

        # Calculate personalized NNT
        total_benefit = sum(personalized_benefits.values())
        individual_nnt = 1 / total_benefit if total_benefit > 0 else float('inf')

        # Calculate NNH for harms
        individual_nnhs = []
        for harm, arr_h in personalized_harms.items():
            if arr_h > 0:
                individual_nnhs.append(1 / arr_h)

        # Decision aid score
        decision_aid_score = self._calculate_decision_aid_score(
            personalized_benefits, personalized_harms, patient, recommendation
        )

        # Generate recommendation for patient
        rec_for_patient = self._generate_patient_recommendation(
            recommendation, decision_aid_score, personalized_benefits,
            personalized_harms, individual_nnt, individual_nnhs
        )

        # Discussion points
        discussion_points = self._generate_discussion_points(
            personalized_benefits, personalized_harms, patient,
            recommendation.certainty.value
        )

        # Uncertainty explanation
        uncertainty_explanation = self._explain_uncertainty_for_patient(
            recommendation.certainty.value, individual_nnt, individual_nnhs
        )

        return PersonalizedRecommendation(
            recommendation=recommendation,
            patient_profile=patient,
            personalized_benefits=personalized_benefits,
            personalized_harms=personalized_harms,
            individual_nnt=individual_nnt,
            individual_nnhs=individual_nnhs,
            decision_aid_score=decision_aid_score,
            recommendation_for_patient=rec_for_patient,
            discussion_points=discussion_points,
            uncertainty_explanation=uncertainty_explanation
        )

    def _calculate_decision_aid_score(
        self,
        benefits: Dict[str, float],
        harms: Dict[str, float],
        patient: PatientProfile,
        recommendation: ClinicalRecommendation
    ) -> float:
        """
        Calculate decision aid score (0-1, higher = more favorable).

        Weights benefits and harms according to patient values.
        """
        score = 0.5  # Start neutral

        # Weight benefits by patient values
        for outcome, arr in benefits.items():
            importance = patient.values_preferences.get(outcome, "moderate")
            weight = {"very important": 1.0, "moderate": 0.5, "less important": 0.2}
            score += arr * weight.get(importance, 0.5)

        # Subtract harms
        for outcome, arr_h in harms.items():
            importance = patient.values_preferences.get(outcome, "moderate")
            weight = {"very important": 1.0, "moderate": 0.5, "less important": 0.2}
            score -= arr_h * weight.get(importance, 0.5) * 2  # Harms weighted more

        return max(0, min(1, score))

    def _generate_patient_recommendation(
        self,
        recommendation: ClinicalRecommendation,
        score: float,
        benefits: Dict[str, float],
        harms: Dict[str, float],
        nnt: float,
        nnhs: List[float]
    ) -> str:
        """Generate patient-specific recommendation text"""
        if score > 0.7:
            rec_text = "Based on your specific situation and values, this treatment is likely to be beneficial for you."
        elif score > 0.4:
            rec_text = "This treatment may be beneficial for you, but the decision depends on your personal preferences."
        else:
            rec_text = "For your specific situation, the harms may outweigh the benefits."

        # Add specific numbers
        if benefits and nnt < float('inf'):
            main_benefit = max(benefits.items(), key=lambda x: x[1])
            rec_text += f"\n\nFor you, we expect about {main_benefit[1]*100:.1f}% more patients to benefit from this treatment compared to not having it."

        if harms and nnhs:
            rec_text += f"\n\nAbout {1/nnhs[0]:.0f} patients would experience a significant side effect."

        return rec_text

    def _generate_discussion_points(
        self,
        benefits: Dict[str, float],
        harms: Dict[str, float],
        patient: PatientProfile,
        certainty: str
    ) -> List[str]:
        """Generate discussion points for patient-clinician conversation"""
        points = []

        points.append(f"Your age ({patient.age}) and overall health may affect how this treatment works for you")

        if patient.comorbidities:
            points.append(f"Your conditions ({', '.join(patient.comorbidities[:2])}) may influence the decision")

        if benefits:
            main_benefit = max(benefits.items(), key=lambda x: x[1])
            points.append(f"The most likely benefit is {main_benefit[0]} (about {main_benefit[1]*100:.1f}% absolute increase)")

        if harms:
            points.append(f"Potential side effects include: {', '.join(harms.keys())}")

        if certainty in ["low", "very_low"]:
            points.append("The evidence for this treatment is not very certain, so your personal experience may differ")

        points.append("Your personal values and preferences are important in this decision")

        return points

    def _explain_uncertainty_for_patient(
        self,
        certainty: str,
        nnt: float,
        nnhs: List[float]
    ) -> str:
        """Explain uncertainty in patient-friendly terms"""
        if certainty == "very_high":
            return "We are very confident about these results."
        elif certainty == "high":
            return "We are confident about these results, though there is some uncertainty."
        elif certainty == "moderate":
            return "The results are moderately certain, but your actual outcome may differ."
        elif certainty == "low":
            return "There is considerable uncertainty about these results for your specific situation."
        else:
            return "The evidence is very uncertain; we cannot confidently predict whether this will help you."


def generate_decision_support(
    pooled_effect: float,
    confidence_interval: Tuple[float, float],
    p_value: float,
    i2: float,
    n_studies: int,
    n_participants: int,
    baseline_risk: float = 0.2,
    certainty: str = "moderate",
    outcome_type: str = "binary",
    serious_harms: Optional[List[str]] = None
) -> ClinicalRecommendation:
    """
    Convenience function to generate clinical recommendation.

    :param pooled_effect: Pooled effect estimate
    :param confidence_interval: 95% CI
    :param p_value: P-value
    :param i2: Heterogeneity I²
    :param n_studies: Number of studies
    :param n_participants: Total participants
    :param baseline_risk: Baseline risk
    :param certainty: GRADE certainty
    :param outcome_type: Type of outcome
    :param serious_harms: List of serious harms
    :return: ClinicalRecommendation
    """
    engine = ClinicalDecisionSupportEngine()

    return engine.generate_recommendation(
        effect_size=pooled_effect,
        confidence_interval=confidence_interval,
        p_value=p_value,
        certainty=certainty,
        baseline_risk=baseline_risk,
        outcome_type=outcome_type,
        n_studies=n_studies,
        n_participants=n_participants,
        i2=i2,
        serious_harms=serious_harms
    )


if __name__ == "__main__":
    print("Clinical Decision Support module loaded")
    print("Features:")
    print("  - GRADE-based recommendation generation")
    print("  - Benefit-harm assessment")
    print("  - Personalized recommendations")
    print("  - Shared decision making support")
    print("  - NNT/NNH calculations")
    print("  - Patient-friendly explanations")
