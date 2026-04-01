"""
Machine Learning-Based Study Quality Assessment

Implements automated quality assessment for clinical trials using ML models.
Combines rule-based scoring with ML prediction of study quality metrics.

References:
- Cochrane Risk of Bias tools (RoB 1, RoB 2)
- QUADAS-2 for diagnostic accuracy
- Newcastle-Ottawa Scale
- MINORS for non-randomized studies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from scipy import stats
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import warnings


@dataclass
class QualityIndicators:
    """Quality indicators for a single study"""
    random_sequence_generation: float  # 0-1 score
    allocation_concealment: float
    blinding_participants: float
    blinding_personnel: float
    blinding_outcome: float
    incomplete_outcome_data: float
    selective_reporting: float
    other_bias: float
    sample_size_adequate: float
    follow_up_complete: float
    analysis_intention_to_treat: float
    baseline_balanced: float
    confounding_controlled: float
    funding_source: float  # Industry vs independent
    journal_impact_factor: float
    publication_year: float


@dataclass
class QualityAssessmentResult:
    """Result from quality assessment"""
    study_id: str
    overall_quality_score: float  # 0-100
    quality_category: str  # "low", "moderate", "high", "very high"
    risk_of_bias: str  # "low", "high", "unclear", "critical"
    individual_scores: Dict[str, float]
    confidence: float
    quality_weight: float  # For meta-analysis weighting
    recommendations: List[str]
    ml_probability: float
    cluster_assignment: int


class RuleBasedQualityAssessor:
    """
    Rule-based quality assessment following Cochrane RoB2 guidelines.
    Provides transparent, interpretable quality scores.
    """

    # Weights for different quality domains
    DOMAIN_WEIGHTS = {
        "randomization": 0.20,
        "deviation_from_intervention": 0.15,
        "missing_outcome_data": 0.15,
        "outcome_measurement": 0.20,
        "selection_of_reported_result": 0.15,
        "overall": 0.15
    }

    def assess_study(
        self,
        indicators: QualityIndicators
    ) -> QualityAssessmentResult:
        """
        Assess study quality using rule-based approach.

        :param indicators: Quality indicators
        :return: QualityAssessmentResult
        """
        # Calculate domain scores
        randomization_score = (
            indicators.random_sequence_generation * 0.6 +
            indicators.allocation_concealment * 0.4
        )

        deviation_score = (
            indicators.blinding_participants * 0.4 +
            indicators.blinding_personnel * 0.3 +
            indicators.analysis_intention_to_treat * 0.3
        )

        missing_data_score = indicators.incomplete_outcome_data

        measurement_score = (
            indicators.blinding_outcome * 0.7 +
            indicators.follow_up_complete * 0.3
        )

        selection_score = indicators.selective_reporting

        overall_score = (
            indicators.sample_size_adequate * 0.3 +
            indicators.baseline_balanced * 0.3 +
            indicators.confounding_controlled * 0.2 +
            (1 - abs(indicators.funding_source - 0.5) * 2) * 0.2
        )

        # Weighted overall quality
        overall_quality = (
            randomization_score * self.DOMAIN_WEIGHTS["randomization"] +
            deviation_score * self.DOMAIN_WEIGHTS["deviation_from_intervention"] +
            missing_data_score * self.DOMAIN_WEIGHTS["missing_outcome_data"] +
            measurement_score * self.DOMAIN_WEIGHTS["outcome_measurement"] +
            selection_score * self.DOMAIN_WEIGHTS["selection_of_reported_result"] +
            overall_score * self.DOMAIN_WEIGHTS["overall"]
        )

        # Convert to 0-100 scale
        overall_quality_100 = overall_quality * 100

        # Determine category
        if overall_quality_100 >= 85:
            category = "very high"
            risk = "low"
        elif overall_quality_100 >= 70:
            category = "high"
            risk = "low"
        elif overall_quality_100 >= 50:
            category = "moderate"
            risk = "some concerns"
        elif overall_quality_100 >= 30:
            category = "low"
            risk = "high"
        else:
            category = "very low"
            risk = "critical"

        # Generate recommendations
        recommendations = self._generate_recommendations(indicators)

        # Calculate quality weight for meta-analysis
        # Higher quality studies get more weight
        quality_weight = overall_quality ** 2

        return QualityAssessmentResult(
            study_id="",
            overall_quality_score=overall_quality_100,
            quality_category=category,
            risk_of_bias=risk,
            individual_scores={
                "randomization": randomization_score,
                "deviation": deviation_score,
                "missing_data": missing_data_score,
                "measurement": measurement_score,
                "selection": selection_score,
                "overall": overall_score
            },
            confidence=1.0,  # Rule-based is deterministic
            quality_weight=quality_weight,
            recommendations=recommendations,
            ml_probability=overall_quality,
            cluster_assignment=0
        )

    def _generate_recommendations(
        self,
        indicators: QualityIndicators
    ) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []

        if indicators.random_sequence_generation < 0.7:
            recommendations.append(
                "Random sequence generation method should be clearly described"
            )

        if indicators.allocation_concealment < 0.7:
            recommendations.append(
                "Allocation concealment method should be detailed"
            )

        if indicators.blinding_participants < 0.5:
            recommendations.append(
                "Participant blinding should be implemented or justified"
            )

        if indicators.blinding_outcome < 0.5:
            recommendations.append(
                "Outcome assessor blinding is recommended"
            )

        if indicators.incomplete_outcome_data < 0.7:
            recommendations.append(
                "Address incomplete outcome data with appropriate methods"
            )

        if not indicators.analysis_intention_to_treat:
            recommendations.append(
                "Intention-to-treat analysis should be performed"
            )

        if indicators.selective_reporting < 0.7:
            recommendations.append(
                "Ensure all pre-specified outcomes are reported"
            )

        return recommendations


class MLQualityAssessor:
    """
    Machine learning-based quality assessment.

    Uses ensemble methods to predict study quality from features.
    Can be trained on expert-rated studies for domain-specific performance.
    """

    def __init__(self, model_type: str = "ensemble"):
        """
        Initialize ML quality assessor.

        :param model_type: Type of model ('ensemble', 'random_forest', 'gradient_boosting')
        """
        self.model_type = model_type
        self.models = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            "random_sequence_generation",
            "allocation_concealment",
            "blinding_participants",
            "blinding_personnel",
            "blinding_outcome",
            "incomplete_outcome_data",
            "selective_reporting",
            "sample_size_adequate",
            "follow_up_complete",
            "intention_to_treat",
            "baseline_balanced",
            "confounding_controlled",
            "funding_source",
            "journal_impact",
            "log_sample_size",
            "study_age"
        ]

    def _create_ensemble(self) -> Dict:
        """Create ensemble of models"""
        return {
            "rf": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            ),
            "gb": GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            "lr": LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )
        }

    def prepare_features(
        self,
        indicators: QualityIndicators,
        sample_size: int,
        publication_year: int
    ) -> np.ndarray:
        """
        Prepare feature vector from quality indicators.

        :param indicators: Quality indicators
        :param sample_size: Study sample size
        :param publication_year: Year of publication
        :return: Feature array
        """
        features = np.array([
            indicators.random_sequence_generation,
            indicators.allocation_concealment,
            indicators.blinding_participants,
            indicators.blinding_personnel,
            indicators.blinding_outcome,
            indicators.incomplete_outcome_data,
            indicators.selective_reporting,
            indicators.sample_size_adequate,
            indicators.follow_up_complete,
            indicators.analysis_intention_to_treat,
            indicators.baseline_balanced,
            indicators.confounding_controlled,
            indicators.funding_source,
            indicators.journal_impact_factor,
            np.log(sample_size + 1),
            2024 - publication_year  # Study age
        ])

        return features.reshape(1, -1)

    def train(
        self,
        training_data: List[Tuple[QualityIndicators, int, int, str]],
        quality_labels: np.ndarray,
        test_size: float = 0.2
    ) -> Dict[str, float]:
        """
        Train ML models on expert-rated studies.

        :param training_data: List of (indicators, sample_size, year, study_id)
        :param quality_labels: Quality category labels (0=low, 1=moderate, 2=high, 3=very high)
        :param test_size: Proportion for testing
        :return: Training metrics
        """
        # Prepare features
        X = np.vstack([
            self.prepare_features(ind, n, year)
            for ind, n, year, _ in training_data
        ])

        y = np.array(quality_labels)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Create models
        self.models = self._create_ensemble()

        # Train each model
        metrics = {}
        for name, model in self.models.items():
            model.fit(X_train_scaled, y_train)

            # Evaluate
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)

            cv_scores = cross_val_score(
                model, X_train_scaled, y_train, cv=5, scoring='f1_weighted'
            )

            try:
                auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
            except:
                auc = None

            metrics[name] = {
                "cv_score": cv_scores.mean(),
                "auc": auc
            }

        self.is_trained = True
        return metrics

    def predict_quality(
        self,
        indicators: QualityIndicators,
        sample_size: int,
        publication_year: int
    ) -> Tuple[float, float, int]:
        """
        Predict quality score using trained models.

        :param indicators: Quality indicators
        :param sample_size: Study sample size
        :param publication_year: Publication year
        :return: (quality_score, confidence, category)
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")

        # Prepare features
        X = self.prepare_features(indicators, sample_size, publication_year)
        X_scaled = self.scaler.transform(X)

        # Get predictions from all models
        predictions = []
        probabilities = []

        for model in self.models.values():
            pred = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0]

            predictions.append(pred)
            probabilities.append(proba)

        # Ensemble: weighted average of probabilities
        avg_proba = np.mean(probabilities, axis=0)
        ensemble_pred = np.argmax(avg_proba)

        # Convert to 0-100 score
        quality_score = (ensemble_pred / 3) * 100

        # Confidence: probability of predicted class
        confidence = avg_proba[ensemble_pred]

        return quality_score, confidence, ensemble_pred

    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """
        Get feature importance from trained models.

        :return: Dictionary of model -> feature_importance
        """
        if not self.is_trained:
            raise ValueError("Models must be trained first")

        importance = {}

        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance[name] = dict(zip(
                    self.feature_names,
                    model.feature_importances_
                ))
            elif hasattr(model, 'coef_'):
                # For logistic regression, use absolute coefficient magnitude
                importance[name] = dict(zip(
                    self.feature_names,
                    np.abs(model.coef_[0])
                ))

        return importance


class ClusteringQualityAssessor:
    """
    Unsupervised clustering to identify quality patterns.

    Useful when expert ratings are not available.
    """

    def __init__(self, method: str = "kmeans"):
        """
        Initialize clustering assessor.

        :param method: Clustering method ('kmeans', 'dbscan')
        """
        self.method = method
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(
        self,
        indicators_list: List[QualityIndicators],
        sample_sizes: List[int],
        publication_years: List[int],
        n_clusters: int = 3
    ) -> None:
        """
        Fit clustering model to studies.

        :param indicators_list: List of quality indicators
        :param sample_sizes: Sample sizes
        :param publication_years: Publication years
        :param n_clusters: Number of clusters (for K-means)
        """
        # Prepare features
        X = np.vstack([
            self._prepare_features(ind, n, year)
            for ind, n, year in zip(indicators_list, sample_sizes, publication_years)
        ])

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Fit clustering model
        if self.method == "kmeans":
            self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=50)
        elif self.method == "dbscan":
            self.model = DBSCAN(eps=0.5, min_samples=3)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")

        self.model.fit(X_scaled)
        self.is_fitted = True

    def _prepare_features(
        self,
        indicators: QualityIndicators,
        sample_size: int,
        publication_year: int
    ) -> np.ndarray:
        """Prepare feature vector"""
        return np.array([
            indicators.random_sequence_generation,
            indicators.allocation_concealment,
            indicators.blinding_participants,
            indicators.blinding_personnel,
            indicators.blinding_outcome,
            indicators.incomplete_outcome_data,
            indicators.selective_reporting,
            indicators.sample_size_adequate,
            indicators.follow_up_complete,
            indicators.analysis_intention_to_treat,
            indicators.baseline_balanced,
            indicators.confounding_controlled,
            np.log(sample_size + 1)
        ])

    def predict_cluster(
        self,
        indicators: QualityIndicators,
        sample_size: int,
        publication_year: int
    ) -> int:
        """
        Predict cluster assignment for a study.

        :param indicators: Quality indicators
        :param sample_size: Sample size
        :param publication_year: Publication year
        :return: Cluster label
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X = self._prepare_features(indicators, sample_size, publication_year)
        X_scaled = self.scaler.transform(X.reshape(1, -1))

        if self.method == "kmeans":
            return self.model.predict(X_scaled)[0]
        else:
            # DBSCAN doesn't have predict, use nearest neighbor
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=1).fit(self.scaler.transform(
                self.model.components_
            ))
            _, idx = nbrs.kneighbors(X_scaled)
            return self.model.labels_[idx[0][0]]

    def get_cluster_centers(self) -> np.ndarray:
        """Get cluster centers (K-means only)"""
        if self.method == "kmeans" and self.is_fitted:
            return self.scaler.inverse_transform(self.model.cluster_centers_)
        return None


class IntegratedQualityAssessor:
    """
    Integrated quality assessment combining multiple approaches.

    Combines:
    1. Rule-based scoring (transparent, consistent)
    2. ML prediction (data-driven, can capture patterns)
    3. Clustering (identifies natural groupings)
    """

    def __init__(self):
        """Initialize integrated assessor"""
        self.rule_assessor = RuleBasedQualityAssessor()
        self.ml_assessor = MLQualityAssessor()
        self.cluster_assessor = ClusteringQualityAssessor()

    def assess_study(
        self,
        indicators: QualityIndicators,
        sample_size: int,
        publication_year: int,
        use_ml: bool = True,
        use_clustering: bool = True,
        ml_weight: float = 0.3
    ) -> QualityAssessmentResult:
        """
        Comprehensive quality assessment.

        :param indicators: Quality indicators
        :param sample_size: Study sample size
        :param publication_year: Publication year
        :param use_ml: Whether to use ML prediction
        :param use_clustering: Whether to use clustering
        :param ml_weight: Weight for ML component (0-1)
        :return: QualityAssessmentResult
        """
        # Rule-based assessment
        rule_result = self.rule_assessor.assess_study(indicators)

        ml_score = 0
        ml_conf = 0
        cluster = 0

        # ML assessment
        if use_ml and self.ml_assessor.is_trained:
            try:
                ml_score, ml_conf, _ = self.ml_assessor.predict_quality(
                    indicators, sample_size, publication_year
                )
            except Exception as e:
                warnings.warn(f"ML prediction failed: {e}")

        # Clustering
        if use_clustering and self.cluster_assessor.is_fitted:
            try:
                cluster = self.cluster_assessor.predict_cluster(
                    indicators, sample_size, publication_year
                )
            except Exception as e:
                warnings.warn(f"Clustering failed: {e}")

        # Combine scores
        if use_ml and self.ml_assessor.is_trained:
            combined_score = (
                (1 - ml_weight) * rule_result.overall_quality_score +
                ml_weight * ml_score
            )
            confidence = ml_conf
        else:
            combined_score = rule_result.overall_quality_score
            confidence = 1.0

        # Determine category
        if combined_score >= 85:
            category = "very high"
        elif combined_score >= 70:
            category = "high"
        elif combined_score >= 50:
            category = "moderate"
        else:
            category = "low"

        # Update risk of bias
        if combined_score >= 70:
            risk = "low"
        elif combined_score >= 50:
            risk = "some concerns"
        else:
            risk = "high"

        # Calculate quality weight
        quality_weight = (combined_score / 100) ** 2

        return QualityAssessmentResult(
            study_id="",
            overall_quality_score=combined_score,
            quality_category=category,
            risk_of_bias=risk,
            individual_scores=rule_result.individual_scores,
            confidence=confidence,
            quality_weight=quality_weight,
            recommendations=rule_result.recommendations,
            ml_probability=ml_score / 100 if ml_score > 0 else 0,
            cluster_assignment=cluster
        )

    def batch_assess_studies(
        self,
        studies: List[Dict],
        train_ml: bool = False,
        n_clusters: int = 3
    ) -> List[QualityAssessmentResult]:
        """
        Batch assessment of multiple studies.

        :param studies: List of study dictionaries with indicators
        :param train_ml: Whether to train ML model on the data
        :param n_clusters: Number of clusters for unsupervised learning
        :return: List of assessment results
        """
        # Prepare data
        all_indicators = []
        sample_sizes = []
        years = []

        for study in studies:
            ind = study["indicators"]
            all_indicators.append(ind)
            sample_sizes.append(study["sample_size"])
            years.append(study["publication_year"])

        # Fit clustering
        self.cluster_assessor.fit(all_indicators, sample_sizes, years, n_clusters)

        # Optionally train ML
        if train_ml and len(studies) > 50:
            # Use rule-based scores as pseudo-labels
            rule_scores = [
                self.rule_assessor.assess_study(ind).overall_quality_score
                for ind in all_indicators
            ]

            # Convert to categories
            labels = []
            for score in rule_scores:
                if score >= 85:
                    labels.append(3)
                elif score >= 70:
                    labels.append(2)
                elif score >= 50:
                    labels.append(1)
                else:
                    labels.append(0)

            # Training data
            training_data = [
                (ind, n, y, study.get("study_id", ""))
                for ind, n, y, study in zip(all_indicators, sample_sizes, years, studies)
            ]

            self.ml_assessor.train(training_data, np.array(labels))

        # Assess all studies
        results = []
        for study, ind, n, year in zip(studies, all_indicators, sample_sizes, years):
            result = self.assess_study(
                ind, n, year,
                use_ml=train_ml,
                use_clustering=True
            )
            result.study_id = study.get("study_id", "")
            results.append(result)

        return results


class QualityWeightedMetaAnalysis:
    """
    Meta-analysis with quality-based weighting.

    Adjusts study weights based on quality assessment results.
    """

    @staticmethod
    def compute_quality_weights(
        results: List[QualityAssessmentResult]
    ) -> np.ndarray:
        """
        Compute quality-based weights.

        :param results: Quality assessment results
        :return: Array of weights
        """
        # Use quality weights from assessment
        weights = np.array([r.quality_weight for r in results])

        # Normalize to sum to 1
        weights = weights / weights.sum()

        return weights

    @staticmethod
    def compute_quality_adjusted_effect(
        effects: np.ndarray,
        variances: np.ndarray,
        quality_weights: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """
        Compute quality-adjusted pooled effect.

        :param effects: Study effects
        :param variances: Study variances
        :param quality_weights: Quality weights
        :return: (pooled_effect, se, ci_lower, ci_upper)
        """
        # Combined weights: precision x quality
        precision_weights = 1 / variances
        combined_weights = precision_weights * quality_weights

        # Normalize
        combined_weights = combined_weights / combined_weights.sum()

        # Weighted effect
        pooled_effect = np.sum(combined_weights * effects)

        # Standard error
        se = np.sqrt(1 / np.sum(precision_weights))

        # Confidence interval
        z = 1.96
        ci_lower = pooled_effect - z * se
        ci_upper = pooled_effect + z * se

        return pooled_effect, se, ci_lower, ci_upper


def extract_quality_indicators_from_study(
    study_data: Dict
) -> QualityIndicators:
    """
    Extract quality indicators from study data.

    This would typically be called by LLM extraction.
    For now, provides a mapping from study features to indicators.

    :param study_data: Dictionary with study information
    :return: QualityIndicators
    """
    # Default values (moderate quality)
    indicators = QualityIndicators(
        random_sequence_generation=study_data.get("randomization_score", 0.5),
        allocation_concealment=study_data.get("allocation_concealment_score", 0.5),
        blinding_participants=study_data.get("blinding_participants_score", 0.5),
        blinding_personnel=study_data.get("blinding_personnel_score", 0.5),
        blinding_outcome=study_data.get("blinding_outcome_score", 0.5),
        incomplete_outcome_data=study_data.get("incomplete_data_score", 0.7),
        selective_reporting=study_data.get("selective_reporting_score", 0.7),
        other_bias=study_data.get("other_bias_score", 0.7),
        sample_size_adequate=study_data.get("sample_size_adequate", 0.7),
        follow_up_complete=study_data.get("follow_up_complete", 0.7),
        analysis_intention_to_treat=float(study_data.get("itt_analysis", False)),
        baseline_balanced=study_data.get("baseline_balanced", 0.7),
        confounding_controlled=study_data.get("confounding_controlled", 0.7),
        funding_source=study_data.get("funding_source", 0.5),  # 0=independent, 1=industry
        journal_impact_factor=study_data.get("journal_impact", 3.0),
        publication_year=study_data.get("publication_year", 2020)
    )

    return indicators


if __name__ == "__main__":
    print("ML Quality Assessment module loaded")
    print("Features:")
    print("  - Rule-based quality assessment (Cochrane RoB2)")
    print("  - ML-based quality prediction")
    print("  - Unsupervised clustering for quality patterns")
    print("  - Integrated assessment combining all approaches")
    print("  - Quality-weighted meta-analysis")
