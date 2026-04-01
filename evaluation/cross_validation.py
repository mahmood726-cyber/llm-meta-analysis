"""
Cross-Validation and Model Selection for Meta-Analysis

Implements leave-one-out and k-fold cross-validation for meta-analysis models,
along with automated model selection based on predictive performance.

References:
- Hastie et al. (2009) Elements of Statistical Learning
- Viechtbauer (2010) Model selection in meta-analysis
- Jackson et al. (2018) Model selection using information criteria
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from scipy import stats
from scipy.stats import norm
import warnings


@dataclass
class CrossValidationResult:
    """Result from cross-validation"""
    model_name: str
    fold_results: List[Dict]
    mean_mse: float
    mean_mae: float
    mean_coverage: float  # Proportion of CIs containing true effect
    mean_ci_width: float
    std_performance: float
    total_studies: int
    n_folds: int


@dataclass
class ModelSelectionResult:
    """Result from model selection"""
    selected_model: str
    selection_criteria: str
    all_model_scores: Dict[str, float]
    ranking: List[Tuple[str, float]]
    justification: str
    alternative_models: List[str]


@dataclass
class PredictionResult:
    """Result from predicting a left-out study"""
    study_id: str
    true_effect: float
    predicted_effect: float
    prediction_interval: Tuple[float, float]
    standard_error: float
    residual: float
    standardized_residual: float
    coverage: bool  # Is true effect in PI?


class MetaAnalysisCrossValidator:
    """
    Cross-validation for meta-analysis models.

    Evaluates predictive performance by leaving out studies and
    assessing prediction accuracy.
    """

    def __init__(self):
        """Initialize cross-validator"""
        self.results: List[CrossValidationResult] = []
        self.best_model: Optional[str] = None

    def leave_one_out_cv(
        self,
        effects: np.ndarray,
        variances: np.ndarray,
        study_ids: List[str],
        models: Optional[List[str]] = None
    ) -> Dict[str, CrossValidationResult]:
        """
        Perform leave-one-out cross-validation.

        :param effects: Study effects
        :param variances: Study variances
        :param study_ids: Study identifiers
        :param models: List of models to evaluate
        :return: Dictionary of model -> CV result
        """
        if models is None:
            models = ["fixed", "dl", "reml", "het"]

        n_studies = len(effects)
        cv_results = {}

        for model_name in models:
            predictions = []
            residuals = []
            coverages = []
            ci_widths = []

            for i in range(n_studies):
                # Leave out study i
                mask = np.ones(n_studies, dtype=bool)
                mask[i] = False

                effects_train = effects[mask]
                variances_train = variances[mask]
                effect_test = effects[i]
                variance_test = variances[i]

                # Fit model on training data
                model_fit = self._fit_model(
                    effects_train, variances_train, model_name
                )

                # Predict left-out study
                pred_result = self._predict_study(
                    model_fit, effect_test, variance_test, model_name
                )

                predictions.append(pred_result.predicted_effect)
                residuals.append(pred_result.residual)
                coverages.append(pred_result.coverage)
                ci_widths.append(
                    pred_result.prediction_interval[1] -
                    pred_result.prediction_interval[0]
                )

            # Compute performance metrics
            mse = np.mean(np.array(residuals)**2)
            mae = np.mean(np.abs(residuals))
            coverage = np.mean(coverages)
            mean_ci_width = np.mean(ci_widths)
            std_perf = np.std(np.array(residuals)**2)

            cv_results[model_name] = CrossValidationResult(
                model_name=model_name,
                fold_results=[{
                    "study_id": study_ids[i],
                    "predicted": predictions[i],
                    "residual": residuals[i]
                } for i in range(n_studies)],
                mean_mse=mse,
                mean_mae=mae,
                mean_coverage=coverage,
                mean_ci_width=mean_ci_width,
                std_performance=std_perf,
                total_studies=n_studies,
                n_folds=n_studies
            )

        self.results = list(cv_results.values())
        return cv_results

    def k_fold_cv(
        self,
        effects: np.ndarray,
        variances: np.ndarray,
        study_ids: List[str],
        k: int = 5,
        models: Optional[List[str]] = None,
        stratified: bool = False
    ) -> Dict[str, CrossValidationResult]:
        """
        Perform k-fold cross-validation.

        :param effects: Study effects
        :param variances: Study variances
        :param study_ids: Study identifiers
        :param k: Number of folds
        :param models: List of models to evaluate
        :param stratified: Whether to stratify by precision
        :return: Dictionary of model -> CV result
        """
        if models is None:
            models = ["fixed", "dl", "reml"]

        n_studies = len(effects)

        # Create folds
        if stratified:
            # Stratify by precision (variance)
            indices = np.argsort(variances)
            folds = []
            for i in range(k):
                fold_indices = indices[i::k]
                folds.append(fold_indices)
        else:
            # Random folds
            indices = np.random.permutation(n_studies)
            fold_size = n_studies // k
            folds = []
            for i in range(k):
                start_idx = i * fold_size
                end_idx = start_idx + fold_size if i < k - 1 else n_studies
                folds.append(indices[start_idx:end_idx])

        cv_results = {}

        for model_name in models:
            all_predictions = []
            all_residuals = []
            all_coverages = []
            all_ci_widths = []

            for fold in folds:
                # Test set
                test_mask = np.zeros(n_studies, dtype=bool)
                test_mask[fold] = True
                train_mask = ~test_mask

                effects_train = effects[train_mask]
                variances_train = variances[train_mask]
                effects_test = effects[test_mask]
                variances_test = variances[test_mask]

                # Fit model
                model_fit = self._fit_model(
                    effects_train, variances_train, model_name
                )

                # Predict test studies
                for effect_test, var_test in zip(effects_test, variances_test):
                    pred_result = self._predict_study(
                        model_fit, effect_test, var_test, model_name
                    )
                    all_predictions.append(pred_result.predicted_effect)
                    all_residuals.append(pred_result.residual)
                    all_coverages.append(pred_result.coverage)
                    all_ci_widths.append(
                        pred_result.prediction_interval[1] -
                        pred_result.prediction_interval[0]
                    )

            # Performance metrics
            mse = np.mean(np.array(all_residuals)**2)
            mae = np.mean(np.abs(all_residuals))
            coverage = np.mean(all_coverages)
            mean_ci_width = np.mean(all_ci_widths)
            std_perf = np.std(np.array(all_residuals)**2)

            cv_results[model_name] = CrossValidationResult(
                model_name=model_name,
                fold_results=[],
                mean_mse=mse,
                mean_mae=mae,
                mean_coverage=coverage,
                mean_ci_width=mean_ci_width,
                std_performance=std_perf,
                total_studies=n_studies,
                n_folds=k
            )

        return cv_results

    def _fit_model(
        self,
        effects: np.ndarray,
        variances: np.ndarray,
        model_type: str
    ) -> Dict:
        """Fit meta-analysis model"""
        weights = 1 / variances
        sum_w = np.sum(weights)

        if model_type == "fixed":
            # Fixed effect model
            pooled = np.sum(weights * effects) / sum_w
            tau2 = 0

        elif model_type == "dl":
            # DerSimonian-Laird
            weighted_mean = np.sum(weights * effects) / sum_w
            q = np.sum(weights * (effects - weighted_mean)**2)
            df = len(effects) - 1
            sum_w2 = np.sum(weights**2)
            tau2 = max(0, (q - df) / (sum_w - sum_w2 / sum_w))
            pooled = weighted_mean

        elif model_type == "reml":
            # REML (simplified)
            tau2 = self._estimate_tau_reml(effects, variances)
            weights_re = 1 / (variances + tau2)
            sum_w_re = np.sum(weights_re)
            pooled = np.sum(weights_re * effects) / sum_w_re

        elif model_type == "het":
            # Hedges (similar to DL)
            weighted_mean = np.sum(weights * effects) / sum_w
            q = np.sum(weights * (effects - weighted_mean)**2)
            df = len(effects) - 1
            c = sum_w - sum_w**2 / sum_w
            tau2 = max(0, (q - df) / c)
            pooled = weighted_mean

        else:
            raise ValueError(f"Unknown model: {model_type}")

        return {
            "pooled_effect": pooled,
            "tau_squared": tau2,
            "weights": weights,
            "model_type": model_type
        }

    def _estimate_tau_reml(
        self,
        effects: np.ndarray,
        variances: np.ndarray,
        max_iter: int = 100,
        tolerance: float = 1e-6
    ) -> float:
        """Estimate tau² using REML"""
        tau2 = 0.1  # Initial value

        for _ in range(max_iter):
            weights = 1 / (variances + tau2)
            sum_w = np.sum(weights)
            weighted_mean = np.sum(weights * effects) / sum_w

            # Update tau²
            residual_var = np.sum(weights * (effects - weighted_mean)**2) / len(effects)
            tau2_new = residual_var - np.mean(variances)

            tau2_new = max(0, tau2_new)

            if abs(tau2_new - tau2) < tolerance:
                break

            tau2 = tau2_new

        return tau2

    def _predict_study(
        self,
        model_fit: Dict,
        true_effect: float,
        true_variance: float,
        model_type: str
    ) -> PredictionResult:
        """Predict a left-out study"""
        tau2 = model_fit["tau_squared"]

        # Prediction includes between-study variance
        prediction_variance = true_variance + tau2
        prediction_se = np.sqrt(prediction_variance)

        # 95% prediction interval
        z = 1.96
        pi_lower = model_fit["pooled_effect"] - z * prediction_se
        pi_upper = model_fit["pooled_effect"] + z * prediction_se

        residual = true_effect - model_fit["pooled_effect"]
        standardized_residual = residual / prediction_se
        coverage = pi_lower <= true_effect <= pi_upper

        return PredictionResult(
            study_id="",
            true_effect=true_effect,
            predicted_effect=model_fit["pooled_effect"],
            prediction_interval=(pi_lower, pi_upper),
            standard_error=prediction_se,
            residual=residual,
            standardized_residual=standardized_residual,
            coverage=coverage
        )


class MetaAnalysisModelSelector:
    """
    Automated model selection for meta-analysis.

    Selects the best model based on information criteria,
    predictive performance, or likelihood-based methods.
    """

    def __init__(self):
        """Initialize model selector"""
        self.criteria_weights = {
            "aic": 0.3,
            "bic": 0.3,
            "cv_mse": 0.2,
            "likelihood": 0.2
        }

    def select_model(
        self,
        effects: np.ndarray,
        variances: np.ndarray,
        criteria: str = "auto",
        models: Optional[List[str]] = None
    ) -> ModelSelectionResult:
        """
        Select best model for meta-analysis.

        :param effects: Study effects
        :param variances: Study variances
        :param criteria: Selection criteria ('aic', 'bic', 'cv', 'auto')
        :param models: List of models to consider
        :return: ModelSelectionResult
        """
        if models is None:
            models = ["fixed", "dl", "reml", "het"]

        scores = {}

        if criteria == "auto" or criteria == "aic":
            aic_scores = self._compute_aic(effects, variances, models)
            for model, score in aic_scores.items():
                scores.setdefault(model, {})["aic"] = score

        if criteria == "auto" or criteria == "bic":
            bic_scores = self._compute_bic(effects, variances, models)
            for model, score in bic_scores.items():
                scores.setdefault(model, {})["bic"] = score

        if criteria == "auto" or criteria == "cv":
            cv = MetaAnalysisCrossValidator()
            cv_results = cv.leave_one_out_cv(effects, variances, ["Study_" + str(i) for i in range(len(effects))], models)
            for model, result in cv_results.items():
                scores.setdefault(model, {})["cv_mse"] = result.mean_mse

        # Combine scores (lower is better for AIC, BIC, MSE)
        combined_scores = {}
        for model in models:
            if model not in scores:
                continue

            # Normalize scores
            score_values = []
            for criterion in ["aic", "bic", "cv_mse"]:
                if criterion in scores[model]:
                    score_values.append(scores[model][criterion])

            if score_values:
                combined_scores[model] = np.mean(score_values)

        # Rank models
        ranking = sorted(combined_scores.items(), key=lambda x: x[1])

        if not ranking:
            selected = "dl"  # Default
            justification = "Default model (DerSimonian-Laird) used - no scoring criteria available"
        else:
            selected = ranking[0][0]
            justification = self._generate_justification(
                selected, ranking, scores, criteria
            )

        # Alternative models
        alternatives = [m for m, _ in ranking[1:4]] if len(ranking) > 1 else []

        return ModelSelectionResult(
            selected_model=selected,
            selection_criteria=criteria,
            all_model_scores=combined_scores,
            ranking=ranking,
            justification=justification,
            alternative_models=alternatives
        )

    def _compute_aic(
        self,
        effects: np.ndarray,
        variances: np.ndarray,
        models: List[str]
    ) -> Dict[str, float]:
        """Compute AIC for each model"""
        aic_scores = {}
        n = len(effects)

        for model_type in models:
            cv = MetaAnalysisCrossValidator()
            fit = cv._fit_model(effects, variances, model_type)

            # Log-likelihood
            weights = 1 / (variances + fit["tau_squared"])
            mu = fit["pooled_effect"]

            # Log-likelihood assuming normality
            log_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi * (variances + fit["tau_squared"])) +
                (effects - mu)**2 / (variances + fit["tau_squared"])
            )

            # AIC = 2k - 2*loglik
            # k = 1 (mean) + 1 (tau2 if random effects)
            k = 2 if fit["tau_squared"] > 0 else 1
            aic = 2 * k - 2 * log_likelihood

            aic_scores[model_type] = aic

        return aic_scores

    def _compute_bic(
        self,
        effects: np.ndarray,
        variances: np.ndarray,
        models: List[str]
    ) -> Dict[str, float]:
        """Compute BIC for each model"""
        bic_scores = {}
        n = len(effects)

        for model_type in models:
            cv = MetaAnalysisCrossValidator()
            fit = cv._fit_model(effects, variances, model_type)

            # Log-likelihood
            weights = 1 / (variances + fit["tau_squared"])
            mu = fit["pooled_effect"]

            log_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi * (variances + fit["tau_squared"])) +
                (effects - mu)**2 / (variances + fit["tau_squared"])
            )

            # BIC = k*log(n) - 2*loglik
            k = 2 if fit["tau_squared"] > 0 else 1
            bic = k * np.log(n) - 2 * log_likelihood

            bic_scores[model_type] = bic

        return bic_scores

    def _generate_justification(
        self,
        selected: str,
        ranking: List[Tuple[str, float]],
        scores: Dict,
        criteria: str
    ) -> str:
        """Generate justification for model selection"""
        justification = f"The {selected.upper()} model was selected"

        if criteria == "auto":
            justification += " based on综合考虑 AIC, BIC, and cross-validation performance"
        elif criteria == "aic":
            justification += " as it had the lowest AIC, indicating best fit with model complexity penalty"
        elif criteria == "bic":
            justification += " as it had the lowest BIC, favoring parsimonious models"
        elif criteria == "cv":
            justification += " based on cross-validation predictive performance"

        # Add comparison
        if len(ranking) > 1:
            diff = ranking[1][1] - ranking[0][1]
            justification += f". It outperformed the {ranking[1][0]} model by {diff:.2f} points"

        return justification


class EnsembleMetaAnalyzer:
    """
    Ensemble methods for meta-analysis.

    Combines multiple models for improved prediction.
    """

    def __init__(self):
        """Initialize ensemble analyzer"""
        self.base_models = ["fixed", "dl", "reml"]
        self.weights: Optional[np.ndarray] = None

    def fit_ensemble(
        self,
        effects: np.ndarray,
        variances: np.ndarray,
        method: str = "average"
    ) -> Dict:
        """
        Fit ensemble model.

        :param effects: Study effects
        :param variances: Study variances
        :param method: Ensemble method ('average', 'weighted', 'stacked')
        :return: Ensemble fit results
        """
        cv = MetaAnalysisCrossValidator()
        model_fits = {}

        # Fit base models
        for model in self.base_models:
            model_fits[model] = cv._fit_model(effects, variances, model)

        # Compute weights
        if method == "average":
            weights = np.ones(len(self.base_models)) / len(self.base_models)

        elif method == "weighted":
            # Weight by inverse of prediction error
            cv_results = cv.leave_one_out_cv(
                effects, variances,
                [str(i) for i in range(len(effects))],
                self.base_models
            )
            mses = np.array([cv_results[m].mean_mse for m in self.base_models])
            inv_mses = 1 / (mses + 1e-10)
            weights = inv_mses / np.sum(inv_mses)

        elif method == "stacked":
            # Stacked generalization (simplified)
            # Use cross-validation to get stacking weights
            cv_results = cv.leave_one_out_cv(
                effects, variances,
                [str(i) for i in range(len(effects))],
                self.base_models
            )

            # Solve for optimal weights using least squares
            predictions = np.vstack([
                np.array([f["predicted"] for f in cv_results[m].fold_results])
                for m in self.base_models
            ]).T

            true_effects = effects

            # Non-negative least squares
            from scipy.optimize import nnls
            weights, _ = nnls(predictions, true_effects)

        self.weights = weights

        # Ensemble prediction
        ensemble_effect = np.sum([
            weights[i] * model_fits[m]["pooled_effect"]
            for i, m in enumerate(self.base_models)
        ])

        # Ensemble variance
        ensemble_var = np.sum([
            weights[i]**2 * model_fits[m].get("tau_squared", 0.01)
            for i, m in enumerate(self.base_models)
        ])

        return {
            "ensemble_effect": ensemble_effect,
            "ensemble_variance": ensemble_var,
            "weights": dict(zip(self.base_models, weights)),
            "base_models": model_fits,
            "method": method
        }


def perform_model_selection(
    effects: np.ndarray,
    variances: np.ndarray,
    criteria: str = "auto"
) -> ModelSelectionResult:
    """
    Convenience function for model selection.

    :param effects: Study effects
    :param variances: Study variances
    :param criteria: Selection criteria
    :return: ModelSelectionResult
    """
    selector = MetaAnalysisModelSelector()
    return selector.select_model(effects, variances, criteria)


def perform_cross_validation(
    effects: np.ndarray,
    variances: np.ndarray,
    method: str = "loo"
) -> Dict[str, CrossValidationResult]:
    """
    Convenience function for cross-validation.

    :param effects: Study effects
    :param variances: Study variances
    :param method: CV method ('loo' or 'kfold')
    :return: CV results
    """
    cv = MetaAnalysisCrossValidator()

    if method == "loo":
        return cv.leave_one_out_cv(
            effects, variances,
            [str(i) for i in range(len(effects))]
        )
    else:
        return cv.k_fold_cv(
            effects, variances,
            [str(i) for i in range(len(effects))],
            k=5
        )


if __name__ == "__main__":
    print("Cross-Validation and Model Selection module loaded")
    print("Features:")
    print("  - Leave-one-out cross-validation")
    print("  - K-fold cross-validation")
    print("  - Model selection (AIC, BIC, CV)")
    print("  - Ensemble methods")
    print("  - Prediction intervals")
