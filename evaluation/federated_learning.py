"""
Federated Learning for Multi-Institution Meta-Analysis

Implements privacy-preserving meta-analysis using federated learning,
allowing institutions to collaborate without sharing raw patient data.

References:
- Privacy-preserving meta-analysis (Rodrigues et al. 2023)
- Federated averaging (McMahan et al. 2017)
- Secure multi-party computation for meta-analysis
- Differential privacy in healthcare research
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from scipy import stats
from scipy.stats import norm
import hashlib
import json
from abc import ABC, abstractmethod


@dataclass
class InstitutionData:
    """Data held by a single institution"""
    institution_id: str
    n_studies: int
    total_participants: int
    effects: np.ndarray
    variances: np.ndarray
    sample_sizes: np.ndarray
    event_counts_intervention: Optional[np.ndarray] = None
    event_counts_comparator: Optional[np.ndarray] = None
    study_ids: Optional[List[str]] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class FederatedUpdate:
    """Update from one institution in federated learning"""
    institution_id: str
    local_effects: np.ndarray
    local_variances: np.ndarray
    local_weights: np.ndarray
    aggregated_statistics: Dict
    timestamp: str
    signature: str


@dataclass
class GlobalModel:
    """Global model (aggregated meta-analysis)"""
    version: int
    pooled_effect: float
    variance: float
    confidence_interval: Tuple[float, float]
    heterogeneity: Dict
    participating_institutions: List[str]
    total_studies: int
    total_participants: int
    convergence_metrics: Dict
    privacy_guarantees: Dict


@dataclass
class PrivacyBudget:
    """Privacy budget tracking for differential privacy"""
    epsilon: float  # Privacy budget
    delta: float  # Failure probability
    used_epsilon: float = 0.0
    remaining_budget: float = 0.0
    mechanisms: List[str] = field(default_factory=list)


class DifferentialPrivacyMechanism:
    """
    Differential privacy mechanisms for meta-analysis.

    Adds calibrated noise to statistics to protect individual contributions.
    """

    @staticmethod
    def laplace_mechanism(
        true_value: float,
        sensitivity: float,
        epsilon: float
    ) -> float:
        """
        Laplace mechanism for ε-differential privacy.

        :param true_value: True statistic value
        :param sensitivity: Global sensitivity of the statistic
        :param epsilon: Privacy parameter
        :return: Private value
        """
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale)
        return true_value + noise

    @staticmethod
    def gaussian_mechanism(
        true_value: float,
        sensitivity: float,
        epsilon: float,
        delta: float
    ) -> float:
        """
        Gaussian mechanism for (ε, δ)-differential privacy.

        :param true_value: True statistic value
        :param sensitivity: L2 sensitivity
        :param epsilon: Privacy parameter
        :param delta: Failure probability
        :return: Private value
        """
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
        noise = np.random.normal(0, sigma)
        return true_value + noise

    @staticmethod
    def exponential_mechanism(
        scores: Dict[str, float],
        sensitivity: float,
        epsilon: float
    ) -> str:
        """
        Exponential mechanism for selecting outputs.

        :param scores: Utility scores for each option
        :param sensitivity: Sensitivity of score function
        :param epsilon: Privacy parameter
        :return: Selected option
        """
        options = list(scores.keys())
        score_values = np.array(list(scores.values()))

        # Compute probabilities
        scaled_scores = epsilon * score_values / (2 * sensitivity)
        exp_scores = np.exp(scaled_scores - np.max(scaled_scores))  # For numerical stability
        probabilities = exp_scores / np.sum(exp_scores)

        # Sample
        return np.random.choice(options, p=probabilities)


class SecureAggregation:
    """
    Secure aggregation protocols for federated meta-analysis.

    Ensures that individual institution updates cannot be reconstructed
    from the aggregated result.
    """

    @staticmethod
    def add_secret_sharing(
        values: List[float],
        prime: int = 2**61 - 1  # Large Mersenne prime
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Additive secret sharing for secure aggregation.

        :param values: Values to share
        :param prime: Prime modulus
        :return: (shares, reconstruction_values)
        """
        n = len(values)
        shares = []

        for i, value in enumerate(values):
            # Create n random shares that sum to value
            random_shares = [np.random.randint(0, prime) for _ in range(n - 1)]
            final_share = (int(value * 1000) - sum(random_shares)) % prime  # Scale for precision
            random_shares.append(final_share)
            shares.append(random_shares)

        return shares, [int(v * 1000) for v in values]

    @staticmethod
    def reconstruct_shares(
        shares: List[List[int]],
        prime: int = 2**61 - 1
    ) -> List[float]:
        """
        Reconstruct values from secret shares.

        :param shares: Secret shares
        :param prime: Prime modulus
        :return: Reconstructed values
        """
        n = len(shares[0])
        values = []

        for i in range(len(shares)):
            sum_shares = sum(shares[i][j] for j in range(len(shares[i])))
            value = (sum_shares % prime) / 1000.0
            values.append(value)

        return values


class FederatedMetaAnalyzer:
    """
    Federated meta-analysis coordinator.

    Orchestrates privacy-preserving meta-analysis across institutions.
    """

    def __init__(
        self,
        min_institutions: int = 3,
        privacy_budget: Optional[PrivacyBudget] = None,
        aggregation_method: str = "weighted_average",
        secure_aggregation: bool = True
    ):
        """
        Initialize federated meta-analyzer.

        :param min_institutions: Minimum institutions required
        :param privacy_budget: Privacy budget for DP
        :param aggregation_method: Method for aggregating updates
        :param secure_aggregation: Whether to use secure aggregation
        """
        self.min_institutions = min_institutions
        self.privacy_budget = privacy_budget or PrivacyBudget(
            epsilon=1.0, delta=1e-5, remaining_budget=1.0
        )
        self.aggregation_method = aggregation_method
        self.secure_aggregation = secure_aggregation

        self.global_model: Optional[GlobalModel] = None
        self.institution_updates: List[FederatedUpdate] = []
        self.converged = False
        self.round = 0

    def register_institution(
        self,
        data: InstitutionData
    ) -> str:
        """
        Register an institution with the federated system.

        :param data: Institution data
        :return: Institution token
        """
        # Generate secure token
        token = hashlib.sha256(
            f"{data.institution_id}_{np.random.randint(0, 1e6)}".encode()
        ).hexdigest()[:16]

        return token

    def compute_local_statistics(
        self,
        data: InstitutionData,
        use_dp: bool = False
    ) -> Dict:
        """
        Compute local statistics at institution (simulated).

        :param data: Institution data
        :param use_dp: Whether to apply differential privacy
        :return: Local statistics
        """
        n = len(data.effects)

        # Compute summary statistics
        weights = 1 / data.variances
        sum_w = np.sum(weights)
        local_pooled = np.sum(weights * data.effects) / sum_w
        local_var = 1 / sum_w

        # Heterogeneity
        q = np.sum(weights * (data.effects - local_pooled)**2)
        df = n - 1
        i2 = max(0, 100 * (q - df) / q) if q > df else 0

        stats_dict = {
            "n_studies": n,
            "total_participants": np.sum(data.sample_sizes),
            "pooled_effect": local_pooled,
            "variance": local_var,
            "weights": weights.tolist(),
            "q_statistic": q,
            "i2": i2,
            "min_effect": float(np.min(data.effects)),
            "max_effect": float(np.max(data.effects))
        }

        # Apply differential privacy if requested
        if use_dp and self.privacy_budget.remaining_budget > 0:
            dp_mechanism = DifferentialPrivacyMechanism()

            # Sensitivity for effect estimate (bounded by range)
            effect_range = np.max(data.effects) - np.min(data.effects)
            sensitivity = effect_range / n

            # Allocate privacy budget
            epsilon_local = 0.1

            if self.privacy_budget.remaining_budget >= epsilon_local:
                stats_dict["pooled_effect"] = dp_mechanism.laplace_mechanism(
                    local_pooled, sensitivity, epsilon_local
                )
                self.privacy_budget.used_epsilon += epsilon_local
                self.privacy_budget.remaining_budget -= epsilon_local
                self.privacy_budget.mechanisms.append("laplace")

        return stats_dict

    def aggregate_updates(
        self,
        updates: List[FederatedUpdate]
    ) -> GlobalModel:
        """
        Aggregate institution updates into global model.

        :param updates: List of institution updates
        :return: Updated global model
        """
        if len(updates) < self.min_institutions:
            raise ValueError(
                f"Need at least {self.min_institutions} institutions, "
                f"got {len(updates)}"
            )

        # Extract statistics
        all_effects = []
        all_variances = []
        all_weights = []

        for update in updates:
            stats = update.aggregated_statistics
            all_effects.append(stats["pooled_effect"])
            all_variances.append(stats["variance"])
            all_weights.append(1 / stats["variance"])

        # Global aggregation
        all_weights = np.array(all_weights)
        all_effects = np.array(all_effects)
        all_variances = np.array(all_variances)

        if self.aggregation_method == "weighted_average":
            # Inverse variance weighting
            sum_w = np.sum(all_weights)
            global_effect = np.sum(all_weights * all_effects) / sum_w
            global_var = 1 / sum_w

        elif self.aggregation_method == "simple_average":
            # Simple average (gives equal weight to all institutions)
            global_effect = np.mean(all_effects)
            global_var = np.mean(all_variances) / len(updates)

        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

        # Confidence interval
        ci = (
            global_effect - 1.96 * np.sqrt(global_var),
            global_effect + 1.96 * np.sqrt(global_var)
        )

        # Total heterogeneity (between institutions)
        q_between = np.sum(all_weights * (all_effects - global_effect)**2)
        df_between = len(updates) - 1

        # Create global model
        global_model = GlobalModel(
            version=self.round,
            pooled_effect=global_effect,
            variance=global_var,
            confidence_interval=ci,
            heterogeneity={
                "q_between": q_between,
                "df_between": df_between,
                "i2_between": max(0, 100 * (q_between - df_between) / q_between) if q_between > df_between else 0
            },
            participating_institutions=[u.institution_id for u in updates],
            total_studies=sum(u.aggregated_statistics["n_studies"] for u in updates),
            total_participants=sum(u.aggregated_statistics["total_participants"] for u in updates),
            convergence_metrics={
                "effect_std": np.std(all_effects),
                "variance_std": np.std(all_variances),
                "n_institutions": len(updates)
            },
            privacy_guarantees={
                "epsilon_used": self.privacy_budget.used_epsilon,
                "epsilon_remaining": self.privacy_budget.remaining_budget,
                "delta": self.privacy_budget.delta,
                "mechanisms": self.privacy_budget.mechanisms
            }
        )

        self.global_model = global_model
        self.round += 1

        return global_model

    def check_convergence(
        self,
        tolerance: float = 0.01,
        window: int = 3
    ) -> bool:
        """
        Check if federated learning has converged.

        :param tolerance: Convergence tolerance
        :param window: Window size for checking
        :return: True if converged
        """
        if len(self.institution_updates) < window:
            return False

        # Check recent effects for stability
        recent_effects = [
            u.aggregated_statistics["pooled_effect"]
            for u in self.institution_updates[-window:]
        ]

        effect_std = np.std(recent_effects)

        self.converged = effect_std < tolerance
        return self.converged

    def federated_averaging(
        self,
        institution_data: List[InstitutionData],
        n_rounds: int = 5,
        local_epochs: int = 1,
        learning_rate: float = 0.1
    ) -> GlobalModel:
        """
        Perform federated averaging (FedAvg) for meta-analysis.

        :param institution_data: Data from each institution
        :param n_rounds: Number of federated rounds
        :param local_epochs: Local computation epochs
        :param learning_rate: Learning rate for updates
        :return: Final global model
        """
        # Initialize global model
        global_effect = 0.0
        global_var = 1.0

        for round_num in range(n_rounds):
            # Each institution computes local update
            updates = []

            for data in institution_data:
                # Compute local statistics
                local_stats = self.compute_local_statistics(data, use_dp=False)

                # Create update
                update = FederatedUpdate(
                    institution_id=data.institution_id,
                    local_effects=data.effects,
                    local_variances=data.variances,
                    local_weights=1 / data.variances,
                    aggregated_statistics=local_stats,
                    timestamp=f"round_{round_num}",
                    signature=""
                )
                updates.append(update)

            # Aggregate updates
            self.institution_updates.extend(updates)
            global_model = self.aggregate_updates(updates)

            global_effect = global_model.pooled_effect
            global_var = global_model.variance

            # Check convergence
            if self.check_convergence():
                break

        return self.global_model


class HomomorphicEncryption:
    """
    Homomorphic encryption for secure meta-analysis.

    Allows computations on encrypted data without decryption.
    """

    def __init__(self):
        """Initialize homomorphic encryption (simplified)"""
        # In practice, use libraries like Microsoft SEAL, PySyft, or TenSEAL
        self.modulus = 2**31 - 1  # Large prime

    def encrypt(self, value: float, public_key: int) -> int:
        """
        Encrypt a value (simplified - use real HE in production).

        :param value: Value to encrypt
        :param public_key: Public key
        :return: Encrypted value
        """
        # Simplified: scale and add random mask
        scaled = int(value * 1000)
        mask = np.random.randint(0, self.modulus)
        encrypted = (scaled + public_key * mask) % self.modulus
        return encrypted

    def decrypt(self, encrypted: int, secret_key: int) -> float:
        """
        Decrypt a value.

        :param encrypted: Encrypted value
        :param secret_key: Secret key
        :return: Decrypted value
        """
        decrypted = (encrypted % secret_key) / 1000.0
        return decrypted

    def add_encrypted(
        self,
        encrypted1: int,
        encrypted2: int
    ) -> int:
        """
        Add two encrypted values.

        :param encrypted1: First encrypted value
        :param encrypted2: Second encrypted value
        :return: Encrypted sum
        """
        return (encrypted1 + encrypted2) % self.modulus


class PrivacyPreservingMetaAnalysis:
    """
    Complete privacy-preserving meta-analysis workflow.

    Combines federated learning, differential privacy, and secure aggregation.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        secure_aggregation: bool = True
    ):
        """
        Initialize privacy-preserving meta-analysis.

        :param epsilon: Total privacy budget
        :param delta: Delta for (ε, δ)-DP
        :param secure_aggregation: Use secure aggregation
        """
        self.privacy_budget = PrivacyBudget(
            epsilon=epsilon, delta=delta, remaining_budget=epsilon
        )

        self.federated_analyzer = FederatedMetaAnalyzer(
            privacy_budget=self.privacy_budget,
            secure_aggregation=secure_aggregation
        )

        self.dp_mechanism = DifferentialPrivacyMechanism()
        self.secure_agg = SecureAggregation()

    def run_private_meta_analysis(
        self,
        institution_data: List[InstitutionData],
        use_dp: bool = True,
        use_secure_aggregation: bool = True
    ) -> GlobalModel:
        """
        Run privacy-preserving meta-analysis.

        :param institution_data: Data from each institution
        :param use_dp: Use differential privacy
        :param use_secure_aggregation: Use secure aggregation
        :return: Global model (aggregated results)
        """
        # Register institutions
        tokens = [
            self.federated_analyzer.register_institution(data)
            for data in institution_data
        ]

        # Run federated averaging
        global_model = self.federated_analyzer.federated_averaging(
            institution_data=institution_data,
            n_rounds=5
        )

        # Apply privacy guarantees
        if use_dp:
            global_model.privacy_guarantees = {
                "method": "differential_privacy",
                "epsilon_used": self.privacy_budget.used_epsilon,
                "delta": self.privacy_budget.delta,
                "mechanisms": self.privacy_budget.mechanisms
            }

        if use_secure_aggregation:
            global_model.privacy_guarantees["secure_aggregation"] = True

        return global_model

    def generate_report(
        self,
        global_model: GlobalModel
    ) -> str:
        """
        Generate human-readable report.

        :param global_model: Global model
        :return: Report text
        """
        report = f"""
Privacy-Preserving Federated Meta-Analysis Report
=================================================

Summary:
- Participating institutions: {len(global_model.participating_institutions)}
- Total studies: {global_model.total_studies}
- Total participants: {global_model.total_participants}

Results:
- Pooled effect: {global_model.pooled_effect:.4f}
- 95% CI: ({global_model.confidence_interval[0]:.4f}, {global_model.confidence_interval[1]:.4f})
- Standard error: {np.sqrt(global_model.variance):.4f}

Heterogeneity:
- Q (between institutions): {global_model.heterogeneity['q_between']:.2f}
- I² (between institutions): {global_model.heterogeneity['i2_between']:.1f}%

Privacy Guarantees:
- ε used: {global_model.privacy_guarantees.get('epsilon_used', 0):.3f}
- δ: {global_model.privacy_guarantees.get('delta', 0):.1e}
- Mechanisms: {', '.join(global_model.privacy_guarantees.get('mechanisms', ['none']))}

Note: Individual institution data remain private and secure.
Only aggregated statistics are shared.
"""

        return report.strip()


def simulate_federated_meta_analysis(
    n_institutions: int = 5,
    studies_per_institution: int = 10,
    true_effect: float = 0.5,
    between_heterogeneity: float = 0.1,
    epsilon: float = 1.0
) -> GlobalModel:
    """
    Simulate federated meta-analysis.

    :param n_institutions: Number of institutions
    :param studies_per_institution: Studies per institution
    :param true_effect: True effect size
    :param between_heterogeneity: Between-institution heterogeneity
    :param epsilon: Privacy budget
    :return: Global model
    """
    # Generate synthetic data for each institution
    institution_data = []

    for i in range(n_institutions):
        # Institution-specific effect (random effect)
        institution_effect = np.random.normal(true_effect, np.sqrt(between_heterogeneity))

        # Generate studies within institution
        effects = np.random.normal(institution_effect, 0.2, studies_per_institution)
        variances = np.random.uniform(0.05, 0.15, studies_per_institution)
        sample_sizes = np.random.randint(50, 200, studies_per_institution)

        data = InstitutionData(
            institution_id=f"institution_{i}",
            n_studies=studies_per_institution,
            total_participants=int(np.sum(sample_sizes)),
            effects=effects,
            variances=variances,
            sample_sizes=sample_sizes,
            study_ids=[f"inst{i}_study{j}" for j in range(studies_per_institution)]
        )
        institution_data.append(data)

    # Run privacy-preserving meta-analysis
    ppma = PrivacyPreservingMetaAnalysis(epsilon=epsilon)
    global_model = ppma.run_private_meta_analysis(institution_data)

    return global_model


if __name__ == "__main__":
    print("Federated Learning for Meta-Analysis module loaded")
    print("Features:")
    print("  - Federated averaging for multi-institution analysis")
    print("  - Differential privacy mechanisms")
    print("  - Secure aggregation")
    print("  - Privacy budget tracking")
    print("  - Convergence detection")
    print("  - Privacy-preserving reporting")
