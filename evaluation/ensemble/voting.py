"""
Ensemble Methods for LLM Extraction

Combines predictions from multiple LLMs to improve accuracy.

Methods:
- Majority voting
- Confidence-weighted voting
- Stacking (meta-learning)
- Bayesian model averaging
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import stats
import json


@dataclass
class EnsemblePrediction:
    """Result from ensemble prediction."""
    prediction: Dict
    confidence: float
    agreement: float
    individual_votes: List[Dict]
    method: str


class LLMPipeline:
    """
    Manages multiple LLMs for ensemble predictions.
    """

    def __init__(self, models: List[str]):
        """
        Initialize ensemble pipeline.

        Args:
            models: List of model names/identifiers
        """
        self.models = models
        self.model_weights = {m: 1.0 for m in models}
        self.model_performance = {m: {'correct': 0, 'total': 0} for m in models}

    def set_weights(self, weights: Dict[str, float]) -> None:
        """
        Set manual weights for models.

        Args:
            weights: Dictionary mapping model names to weights
        """
        for model, weight in weights.items():
            if model in self.model_weights:
                self.model_weights[model] = weight

    def update_performance(
        self,
        model: str,
        correct: bool
    ) -> None:
        """
        Update performance tracking for a model.

        Args:
            model: Model name
            correct: Whether prediction was correct
        """
        if model in self.model_performance:
            self.model_performance[model]['total'] += 1
            if correct:
                self.model_performance[model]['correct'] += 1

    def get_weights(self) -> Dict[str, float]:
        """
        Get current weights based on performance.

        Returns:
            Dictionary of model weights
        """
        weights = {}
        for model, perf in self.model_performance.items():
            if perf['total'] > 0:
                # Weight by accuracy
                accuracy = perf['correct'] / perf['total']
                weights[model] = accuracy
            else:
                weights[model] = self.model_weights[model]

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights


class MajorityVotingEnsemble:
    """
    Majority voting ensemble for extraction.

    Takes predictions from multiple models and selects the most common answer.
    """

    def __init__(self, models: List[str]):
        """
        Initialize majority voting ensemble.

        Args:
            models: List of model names
        """
        self.models = models
        self.pipeline = LLMPipeline(models)

    def vote(
        self,
        predictions: List[Dict],
        task: str = "extraction"
    ) -> EnsemblePrediction:
        """
        Perform majority voting on predictions.

        Args:
            predictions: List of prediction dictionaries from each model
            task: Type of task ('extraction', 'classification', etc.)

        Returns:
            EnsemblePrediction with majority vote result
        """
        if not predictions:
            raise ValueError("No predictions provided")

        # For numerical extraction, use weighted median
        if task == "extraction":
            return self._vote_extraction(predictions)

        # For classification, use majority vote
        elif task == "classification":
            return self._vote_classification(predictions)

        else:
            return self._vote_general(predictions)

    def _vote_extraction(self, predictions: List[Dict]) -> EnsemblePrediction:
        """Vote on extraction results."""
        # Extract numerical values
        intervention_events = []
        intervention_totals = []
        comparator_events = []
        comparator_totals = []

        for pred in predictions:
            data = pred.get('data', pred)

            if isinstance(data, dict):
                intervention = data.get('intervention', {})
                comparator = data.get('comparator', {})

                intervention_events.append(intervention.get('events'))
                intervention_totals.append(intervention.get('total'))
                comparator_events.append(comparator.get('events'))
                comparator_totals.append(comparator.get('total'))

        # Remove None values
        def filter_and_vote(values):
            valid = [v for v in values if v is not None and v != 'unknown']
            if not valid:
                return None
            # Use median for robustness
            return int(np.median(valid))

        # Get majority vote (median for numerical)
        result = {
            'intervention': {
                'events': filter_and_vote(intervention_events),
                'total': filter_and_vote(intervention_totals)
            },
            'comparator': {
                'events': filter_and_vote(comparator_events),
                'total': filter_and_vote(comparator_totals)
            }
        }

        # Calculate agreement
        agreement = self._calculate_agreement(predictions)

        # Calculate confidence
        confidence = self._calculate_confidence(predictions, result)

        return EnsemblePrediction(
            prediction=result,
            confidence=confidence,
            agreement=agreement,
            individual_votes=predictions,
            method='majority_vote'
        )

    def _vote_classification(self, predictions: List[Dict]) -> EnsemblePrediction:
        """Vote on classification results."""
        # Count votes for each class
        votes = {}
        for pred in predictions:
            label = pred.get('prediction', pred.get('label'))
            if label:
                votes[label] = votes.get(label, 0) + 1

        # Get majority class
        if votes:
            majority_label = max(votes, key=votes.get)
            confidence = votes[majority_label] / len(predictions)
        else:
            majority_label = None
            confidence = 0.0

        # Agreement
        agreement = max(votes.values()) / len(predictions) if votes else 0.0

        return EnsemblePrediction(
            prediction={'label': majority_label},
            confidence=confidence,
            agreement=agreement,
            individual_votes=predictions,
            method='majority_vote'
        )

    def _vote_general(self, predictions: List[Dict]) -> EnsemblePrediction:
        """General voting for other task types."""
        # Use first valid prediction as fallback
        for pred in predictions:
            if pred and not pred.get('error'):
                return EnsemblePrediction(
                    prediction=pred,
                    confidence=1.0 / len(predictions),
                    agreement=1.0 / len(predictions),
                    individual_votes=predictions,
                    method='first_valid'
                )

        return EnsemblePrediction(
            prediction={},
            confidence=0.0,
            agreement=0.0,
            individual_votes=predictions,
            method='fallback'
        )

    def _calculate_agreement(self, predictions: List[Dict]) -> float:
        """Calculate level of agreement among predictions."""
        if len(predictions) <= 1:
            return 1.0

        # For simplicity, check if all predictions are identical
        # More sophisticated: calculate pairwise similarity
        first = predictions[0]
        matches = sum(1 for p in predictions[1:] if p == first)
        return (matches + 1) / len(predictions)

    def _calculate_confidence(
        self,
        predictions: List[Dict],
        result: Dict
    ) -> float:
        """Calculate confidence based on agreement and model confidence."""
        # Base confidence from agreement
        agreement = self._calculate_agreement(predictions)

        # Could also incorporate individual model confidences
        # if models provide confidence scores

        return agreement


class WeightedVotingEnsemble:
    """
    Weighted voting ensemble.

    Models are weighted by performance or confidence.
    """

    def __init__(self, models: List[str]):
        """Initialize weighted voting ensemble."""
        self.models = models
        self.pipeline = LLMPipeline(models)

    def vote(
        self,
        predictions: List[Dict],
        weights: Optional[Dict[str, float]] = None
    ) -> EnsemblePrediction:
        """
        Perform weighted voting.

        Args:
            predictions: List of predictions
            weights: Optional manual weights (uses performance if not provided)

        Returns:
            EnsemblePrediction
        """
        if weights is None:
            weights = self.pipeline.get_weights()

        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        # For numerical extraction
        result_int_events = 0.0
        result_int_total = 0.0
        result_comp_events = 0.0
        result_comp_total = 0.0
        total_weight = 0.0

        for i, pred in enumerate(predictions):
            if i >= len(self.models):
                break

            model = self.models[i]
            weight = weights.get(model, 1.0)

            data = pred.get('data', pred)
            if isinstance(data, dict):
                intervention = data.get('intervention', {})
                comparator = data.get('comparator', {})

                if intervention.get('events') not in [None, 'unknown']:
                    result_int_events += weight * intervention['events']
                if intervention.get('total') not in [None, 'unknown']:
                    result_int_total += weight * intervention['total']
                if comparator.get('events') not in [None, 'unknown']:
                    result_comp_events += weight * comparator['events']
                if comparator.get('total') not in [None, 'unknown']:
                    result_comp_total += weight * comparator['total']

                total_weight += weight

        if total_weight > 0:
            result_int_events /= total_weight
            result_int_total /= total_weight
            result_comp_events /= total_weight
            result_comp_total /= total_weight

        result = {
            'intervention': {
                'events': int(round(result_int_events)) if result_int_events > 0 else None,
                'total': int(round(result_int_total)) if result_int_total > 0 else None
            },
            'comparator': {
                'events': int(round(result_comp_events)) if result_comp_events > 0 else None,
                'total': int(round(result_comp_total)) if result_comp_total > 0 else None
            }
        }

        return EnsemblePrediction(
            prediction=result,
            confidence=0.8,  # Placeholder
            agreement=0.0,
            individual_votes=predictions,
            method='weighted_vote'
        )


class StackingEnsemble:
    """
    Stacking ensemble with meta-learner.

    Trains a meta-model to combine base model predictions.
    """

    def __init__(
        self,
        base_models: List[str],
        meta_model: str = "logistic"
    ):
        """
        Initialize stacking ensemble.

        Args:
            base_models: List of base model names
            meta_model: Type of meta-model ('logistic', 'random_forest', 'linear')
        """
        self.base_models = base_models
        self.meta_model_type = meta_model
        self.meta_model = None
        self.fitted = False

    def train(
        self,
        predictions: List[np.ndarray],
        ground_truth: np.ndarray
    ) -> None:
        """
        Train meta-model on predictions.

        Args:
            predictions: List of prediction arrays from each model
            ground_truth: True labels/values
        """
        # Stack predictions as features
        X = np.column_stack(predictions)

        if self.meta_model_type == "linear":
            # Linear regression
            from sklearn.linear_model import LinearRegression
            self.meta_model = LinearRegression()
            self.meta_model.fit(X, ground_truth)

        elif self.meta_model_type == "logistic":
            # Logistic regression
            from sklearn.linear_model import LogisticRegression
            self.meta_model = LogisticRegression()
            self.meta_model.fit(X, ground_truth)

        elif self.meta_model_type == "random_forest":
            # Random forest
            from sklearn.ensemble import RandomForestClassifier
            self.meta_model = RandomForestClassifier(n_estimators=100)
            self.meta_model.fit(X, ground_truth)

        self.fitted = True

    def predict(
        self,
        predictions: List[np.ndarray]
    ) -> np.ndarray:
        """
        Make prediction with trained meta-model.

        Args:
            predictions: List of prediction arrays

        Returns:
            Ensemble prediction
        """
        if not self.fitted:
            raise RuntimeError("Meta-model not trained. Call train() first.")

        X = np.column_stack(predictions)
        return self.meta_model.predict(X)

    def predict_proba(
        self,
        predictions: List[np.ndarray]
    ) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            predictions: List of prediction arrays

        Returns:
            Prediction probabilities
        """
        if not self.fitted:
            raise RuntimeError("Meta-model not trained. Call train() first.")

        X = np.column_stack(predictions)

        if hasattr(self.meta_model, 'predict_proba'):
            return self.meta_model.predict_proba(X)
        else:
            # For regression, return dummy probabilities
            pred = self.meta_model.predict(X)
            return np.column_stack([1 - pred, pred])


class BayesianModelAveraging:
    """
    Bayesian Model Averaging (BMA).

    Combines models weighted by posterior probability.
    """

    def __init__(
        self,
        models: List[str],
        prior_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize BMA ensemble.

        Args:
            models: List of model names
            prior_weights: Optional prior weights (uniform if not provided)
        """
        self.models = models
        self.prior_weights = prior_weights or {m: 1.0 for m in models}
        self.posterior_weights = None

    def update_posterior(
        self,
        likelihoods: Dict[str, float]
    ) -> None:
        """
        Update posterior weights based on likelihoods.

        Args:
            likelihoods: Dictionary of model likelihoods
        """
        # Calculate posterior: prior * likelihood
        unnormalized = {}
        for model in self.models:
            prior = self.prior_weights.get(model, 1.0)
            likelihood = likelihoods.get(model, 1e-10)
            unnormalized[model] = prior * likelihood

        # Normalize
        total = sum(unnormalized.values())
        if total > 0:
            self.posterior_weights = {
                k: v / total for k, v in unnormalized.items()
            }
        else:
            self.posterior_weights = {m: 1.0 / len(self.models) for m in self.models}

    def predict(
        self,
        predictions: List[Dict]
    ) -> EnsemblePrediction:
        """
        Make BMA prediction.

        Args:
            predictions: List of predictions

        Returns:
            EnsemblePrediction weighted by posterior
        """
        if self.posterior_weights is None:
            # Use uniform weights if no posterior calculated
            weights = {m: 1.0 / len(self.models) for m in self.models}
        else:
            weights = self.posterior_weights

        # Use weighted voting with posterior weights
        weighted_ensemble = WeightedVotingEnsemble(self.models)
        return weighted_ensemble.vote(predictions, weights=weights)


class DisagreementAnalyzer:
    """
    Analyze disagreements between models.

    Identifies cases where models disagree significantly.
    """

    @staticmethod
    def analyze_disagreement(
        predictions: List[Dict],
        ground_truth: Optional[Dict] = None
    ) -> Dict:
        """
        Analyze level and causes of disagreement.

        Args:
            predictions: List of predictions
            ground_truth: Optional true values for comparison

        Returns:
            Dictionary with disagreement analysis
        """
        analysis = {
            'n_models': len(predictions),
            'n_predictions': len(predictions),
            'disagreement_score': 0.0,
            'outlier_models': [],
            'consensus_prediction': None,
            'correct_model': None
        }

        if not predictions:
            return analysis

        # Extract numerical values
        values = []
        for pred in predictions:
            data = pred.get('data', pred)
            if isinstance(data, dict):
                intervention = data.get('intervention', {})
                # Create simple score for comparison
                if intervention.get('events') is not None:
                    values.append(intervention['events'])

        if not values:
            return analysis

        # Calculate disagreement (standard deviation / range)
        values = np.array(values)
        disagreement = np.std(values) / (np.max(values) - np.min(values) + 1e-10)
        analysis['disagreement_score'] = float(disagreement)

        # Find outliers (models far from consensus)
        median = np.median(values)
        mad = np.median(np.abs(values - median))  # Median absolute deviation

        outlier_threshold = 3 * mad
        for i, (pred, val) in enumerate(zip(predictions, values)):
            if abs(val - median) > outlier_threshold:
                analysis['outlier_models'].append({
                    'model': f'model_{i}',
                    'value': val,
                    'deviation': abs(val - median)
                })

        # Consensus prediction (median)
        analysis['consensus_prediction'] = {
            'intervention_events': float(np.median(values))
        }

        # If ground truth provided, find which model was correct
        if ground_truth is not None:
            true_value = ground_truth.get('intervention', {}).get('events')
            if true_value is not None:
                closest_model = np.argmin(np.abs(np.array(values) - true_value))
                analysis['correct_model'] = f'model_{closest_model}'

        return analysis


if __name__ == "__main__":
    print("Ensemble Methods Module loaded")
    print("Features:")
    print("  - Majority voting ensemble")
    print("  - Confidence-weighted voting")
    print("  - Stacking ensemble with meta-learning")
    print("  - Bayesian model averaging")
    print("  - Disagreement analysis")
