"""
Advanced Bayesian Meta-Analysis

Implements hierarchical Bayesian models, robust models, and
Bayesian network meta-analysis.

References:
- Gelman et al. (2013). Bayesian Data Analysis.
- Higgins et al. (2009). Bayesian models for meta-analysis.
- Dias et al. (2013). Network meta-analysis using Bayesian methods.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False

    print("Note: pymc not available. Install with: pip install pymc arviz")


@dataclass
class BayesianMAResult:
    """Results from Bayesian meta-analysis."""
    posterior_mean: float
    posterior_sd: float
    ci_lower: float
    ci_upper: float
    hdi_95: Tuple[float, float]
    r_hat: float
    ess_bulk: float
    ess_tail: float
    posterior_samples: np.ndarray
    n_studies: int
    n_chains: int
    n_draws: int


class HierarchicalBayesianMA:
    """
    Hierarchical Bayesian meta-analysis.

    Models between-study heterogeneity with a hierarchical prior.
    """

    @staticmethod
    def fit(
        effects: np.ndarray,
        variances: np.ndarray,
        n_chains: int = 4,
        n_draws: int = 2000,
        n_tune: int = 1000,
        target_accept: float = 0.9,
        seed: Optional[int] = None
    ) -> BayesianMAResult:
        """
        Fit hierarchical Bayesian random-effects model.

        Model:
        y_i ~ N(theta_i, sigma_i^2)  # Study-specific effect
        theta_i ~ N(mu, tau^2)        # Study effects from population

        Priors:
        mu ~ N(0, 10^2)               # Overall effect
        tau ~ Half-Cauchy(1)           # Between-study SD

        Args:
            effects: Study effect estimates
            variances: Study variances
            n_chains: Number of MCMC chains
            n_draws: Number of posterior samples per chain
            n_tune: Number of tuning steps
            target_accept: Target acceptance rate
            seed: Random seed

        Returns:
            BayesianMAResult with posterior samples
        """
        if not PYMC_AVAILABLE:
            raise ImportError("pymc is required for Bayesian meta-analysis")

        n = len(effects)

        with pm.Model() as model:
            # Data
            y = pm.MutableData("y", effects)
            sigma = pm.MutableData("sigma", np.sqrt(variances))

            # Priors
            mu = pm.Normal("mu", mu=0, sigma=10)
            tau = pm.HalfCauchy("tau", beta=1)

            # Study-specific effects
            theta = pm.Normal("theta", mu=mu, sigma=tau, shape=n)

            # Likelihood
            pm.Normal("likelihood", mu=theta, sigma=sigma, observed=effects)

            # Sample
            trace = pm.sample(
                draws=n_draws,
                tune=n_tune,
                chains=n_chains,
                target_accept=target_accept,
                random_seed=seed,
                return_inferencedata=False
            )

        # Convert to ArviZ
        idata = az.from_pymc(trace)

        # Extract posterior of mu
        mu_posterior = trace.posterior["mu"].values
        mu_posterior_flat = mu_posterior.flatten()

        # Compute statistics
        posterior_mean = float(np.mean(mu_posterior_flat))
        posterior_sd = float(np.std(mu_posterior_flat))

        # 95% credible interval
        ci_lower = float(np.percentile(mu_posterior_flat, 2.5))
        ci_upper = float(np.percentile(mu_posterior_flat, 97.5))

        # 95% highest density interval
        hdi = az.hdi(idata, var_names=["mu"], hdi_prob=0.95)
        hdi_95 = (float(hdi["mu"].sel(hdi="lower").values),
                  float(hdi["mu"].sel(hdi="higher").values))

        # Diagnostics
        r_hat = float(az.rhat(idata, var_names=["mu"]).values)
        ess_bulk = float(az.ess(idata, var_names=["mu"], method="bulk").values)
        ess_tail = float(az.ess(idata, var_names=["mu"], method="tail").values)

        return BayesianMAResult(
            posterior_mean=posterior_mean,
            posterior_sd=posterior_sd,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            hdi_95=hdi_95,
            r_hat=r_hat,
            ess_bulk=ess_bulk,
            ess_tail=ess_tail,
            posterior_samples=mu_posterior_flat,
            n_studies=n,
            n_chains=n_chains,
            n_draws=n_draws
        )

    @staticmethod
    def predictive_check(
        effects: np.ndarray,
        variances: np.ndarray,
        result: BayesianMAResult
    ) -> Dict:
        """
        Perform posterior predictive check.

        Args:
            effects: Observed effects
            variances: Study variances
            result: BayesianMAResult from fit()

        Returns:
            Dictionary with PPC statistics
        """
        # Generate posterior predictive distribution
        # y_rep ~ N(theta_i, sigma_i^2)

        n = len(effects)
        posterior_samples = result.posterior_samples

        # For each posterior sample, generate replicated data
        y_rep_list = []

        for mu in posterior_samples:
            # Draw tau from its posterior (simplified - use sample SD)
            tau = np.std(posterior_samples)  # Simplified

            # Draw theta_i from population
            theta = np.random.normal(mu, tau, size=n)

            # Draw y_rep from likelihood
            y_rep = np.random.normal(theta, np.sqrt(variances))
            y_rep_list.append(y_rep)

        y_rep = np.array(y_rep_list)

        # Compare observed to replicated
        # Compute test statistics
        observed_mean = np.mean(effects)
        replicated_means = np.mean(y_rep, axis=1)

        # P-value: proportion where replicated >= observed
        p_value = np.mean(replicated_means >= observed_mean)

        return {
            'observed_mean': observed_mean,
            'replicated_mean': np.mean(replicated_means),
            'p_value': p_value,
            'y_rep': y_rep
        }


class RobustBayesianMA:
    """
    Robust Bayesian meta-analysis using t-distribution.

    Uses Student-t likelihood to handle outliers.
    """

    @staticmethod
    def fit(
        effects: np.ndarray,
        variances: np.ndarray,
        df: int = 4,  # Degrees of freedom for t-distribution
        n_chains: int = 4,
        n_draws: int = 2000,
        seed: Optional[int] = None
    ) -> BayesianMAResult:
        """
        Fit robust Bayesian model with t-likelihood.

        Model:
        y_i ~ StudentT_nu(theta_i, sigma_i^2)  # Robust likelihood
        theta_i ~ N(mu, tau^2)

        Args:
            effects: Study effect estimates
            variances: Study variances
            df: Degrees of freedom for t-distribution (lower = more robust)
            n_chains: Number of MCMC chains
            n_draws: Number of posterior samples
            seed: Random seed

        Returns:
            BayesianMAResult
        """
        if not PYMC_AVAILABLE:
            raise ImportError("pymc is required for Bayesian meta-analysis")

        n = len(effects)

        with pm.Model() as model:
            # Data
            y = pm.MutableData("y", effects)
            sigma = pm.MutableData("sigma", np.sqrt(variances))

            # Priors
            mu = pm.Normal("mu", mu=0, sigma=10)
            tau = pm.HalfCauchy("tau", beta=1)

            # Study-specific effects
            theta = pm.Normal("theta", mu=mu, sigma=tau, shape=n)

            # Robust likelihood (Student-t)
            # PyMC parameterization: nu, mu, sigma
            pm.StudentT("likelihood", nu=df, mu=theta, sigma=sigma, observed=y)

            # Sample
            trace = pm.sample(
                draws=n_draws,
                tune=1000,
                chains=n_chains,
                target_accept=0.9,
                random_seed=seed,
                return_inferencedata=False
            )

        # Convert to ArviZ and extract results
        idata = az.from_pymc(trace)
        mu_posterior = trace.posterior["mu"].values.flatten()

        return BayesianMAResult(
            posterior_mean=float(np.mean(mu_posterior)),
            posterior_sd=float(np.std(mu_posterior)),
            ci_lower=float(np.percentile(mu_posterior, 2.5)),
            ci_upper=float(np.percentile(mu_posterior, 97.5)),
            hdi_95=(
                float(np.percentile(mu_posterior, 2.5)),
                float(np.percentile(mu_posterior, 97.5))
            ),
            r_hat=float(az.rhat(idata, var_names=["mu"]).values),
            ess_bulk=float(az.ess(idata, var_names=["mu"], method="bulk").values),
            ess_tail=float(az.ess(idata, var_names=["mu"], method="tail").values),
            posterior_samples=mu_posterior,
            n_studies=n,
            n_chains=n_chains,
            n_draws=n_draws
        )


class BayesianNetworkMA:
    """
    Bayesian network meta-analysis.

    Extends NMA with Bayesian inference for all treatment effects.
    """

    @staticmethod
    def fit(
        studies: List[Dict],
        treatments: List[str],
        reference: str,
        n_chains: int = 4,
        n_draws: int = 5000,
        seed: Optional[int] = None
    ) -> Dict:
        """
        Fit Bayesian network meta-analysis model.

        Model:
        d_ij ~ N(delta_ij, sigma_ij^2)  # Direct comparison effect
        delta_ij = theta_i - theta_j      # Relativized effects

        Priors:
        theta_1 = 0 (reference)
        theta_i ~ N(0, tau^2) for i > 1

        Args:
            studies: List of studies with comparison data
            treatments: List of treatment names
            reference: Reference treatment name
            n_chains: Number of MCMC chains
            n_draws: Number of posterior samples
            seed: Random seed

        Returns:
            Dictionary with NMA results
        """
        if not PYMC_AVAILABLE:
            raise ImportError("pymc is required for Bayesian NMA")

        # Prepare data
        n_treatments = len(treatments)
        reference_idx = treatments.index(reference)

        # Map treatment names to indices
        comparisons = []
        for study in studies:
            tx = study.get('treatments', [])
            if len(tx) == 2:
                i = treatments.index(tx[0])
                j = treatments.index(tx[1])
                comparisons.append({
                    'study_id': study.get('study_id'),
                    'i': i,
                    'j': j,
                    'effect': study.get('effect'),
                    'variance': study.get('variance')
                })

        m = len(comparisons)

        with pm.Model() as model:
            # Data
            idx_i = np.array([c['i'] for c in comparisons])
            idx_j = np.array([c['j'] for c in comparisons])
            y = np.array([c['effect'] for c in comparisons])
            sigma = np.array([np.sqrt(c['variance']) for c in comparisons])

            # Priors
            tau = pm.HalfCauchy("tau", beta=1)

            # Treatment effects (relativized to reference)
            theta = pm.Normal("theta", mu=0, sigma=tau, shape=n_treatments)

            # Constraint: theta[reference] = 0
            # We model theta[1..n] but only identify n-1 contrasts
            # Set reference to 0
            fixes = pm.Potential("theta_ref", theta[reference_idx], 0)

            # Expected effect for each comparison
            mu_delta = theta[idx_i] - theta[idx_j]

            # Likelihood
            pm.Normal("likelihood", mu=mu_delta, sigma=sigma, observed=y)

            # Sample
            trace = pm.sample(
                draws=n_draws,
                tune=2000,
                chains=n_chains,
                target_accept=0.9,
                random_seed=seed,
                return_inferencedata=False
            )

        # Extract results
        idata = az.from_pymc(trace)

        # Treatment effects relative to reference
        treatment_effects = {}
        for i, treatment in enumerate(treatments):
            if i != reference_idx:
                theta_post = trace.posterior["theta"][:, :, i].values.flatten()
                treatment_effects[treatment] = {
                    'mean': float(np.mean(theta_post)),
                    'sd': float(np.std(theta_post)),
                    'ci_lower': float(np.percentile(theta_post, 2.5)),
                    'ci_upper': float(np.percentile(theta_post, 97.5)),
                    'r_hat': float(az.rhat(idata, var_names=["theta"], coords={"theta_dim": i}).values)
                }

        # Ranking (SUCRA)
        ranks = BayesianNetworkMA._calculate_sucra(trace, treatments, reference_idx)

        return {
            'treatment_effects': treatment_effects,
            'ranking': ranks,
            'reference': reference,
            'n_studies': len(studies),
            'n_chains': n_chains,
            'n_draws': n_draws
        }

    @staticmethod
    def _calculate_sucra(
        trace,
        treatments: List[str],
        reference_idx: int
    ) -> Dict[str, float]:
        """
        Calculate SUCRA (Surface Under the Cumulative Ranking Curve).

        Args:
            trace: MCMC trace
            treatments: List of treatment names
            reference_idx: Index of reference treatment

        Returns:
            Dictionary mapping treatments to SUCRA scores
        """
        n_treatments = len(treatments)
        n_samples = trace.posterior["theta"].shape[0] * trace.posterior["theta"].shape[1]

        sucra_scores = {t: 0.0 for t in treatments}

        for sample_idx in range(n_samples):
            # Get one sample from each chain
            theta_samples = []
            for chain_idx in range(trace.posterior["theta"].shape[1]):
                sample = trace.posterior["theta"][sample_idx, chain_idx, :].values
                theta_samples.append(sample)

            for sample in theta_samples:
                # Rank treatments (lower is better for negative outcomes)
                # Use theta as effect measure
                ranks = np.argsort(sample)
                for rank, treatment_idx in enumerate(ranks):
                    treatment = treatments[treatment_idx]
                    sucra_scores[treatment] += (n_treatments - 1 - rank)

        # Normalize
        for treatment in sucra_scores:
            sucra_scores[treatment] /= n_samples * (n_treatments - 1)

        return sucra_scores


class PosteriorPredictiveChecks:
    """
    Posterior predictive checks for Bayesian model validation.
    """

    @staticmethod
    def check_heterogeneity(
        effects: np.ndarray,
        variances: np.ndarray,
        result: BayesianMAResult
    ) -> Dict:
        """
        Check heterogeneity using posterior predictive checks.

        Args:
            effects: Observed effects
            variances: Study variances
            result: BayesianMAResult

        Returns:
            Dictionary with PPC results
        """
        # Compute Q statistic
        mu = result.posterior_mean
        tau = np.std(result.posterior_samples)

        n = len(effects)
        weights = 1 / variances
        sum_w = np.sum(weights)

        # Q statistic
        q = np.sum(weights * (effects - mu) ** 2)

        # Generate posterior distribution of Q
        q_rep = []
        for _ in range(1000):
            # Draw mu from posterior
            mu_draw = np.random.choice(result.posterior_samples)

            # Draw tau from posterior
            tau_draw = np.random.choice(result.posterior_samples)
            tau_draw = np.abs(tau_draw)  # Ensure positive

            # Simulate data
            theta = np.random.normal(mu_draw, tau_draw, size=n)
            y_sim = np.random.normal(theta, np.sqrt(variances))

            # Compute Q for simulated data
            q_sim = np.sum(weights * (y_sim - mu_draw) ** 2)
            q_rep.append(q_sim)

        q_rep = np.array(q_rep)

        # P-value
        p_value = np.mean(q_rep >= q)

        return {
            'q_observed': q,
            'q_mean_pred': np.mean(q_rep),
            'p_value': p_value,
            'fit': 'adequate' if p_value > 0.05 else 'poor'
        }

    @staticmethod
    def forest_plot(
        effects: np.ndarray,
        variances: np.ndarray,
        result: BayesianMAResult,
        study_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Generate forest plot data from Bayesian results.

        Args:
            effects: Observed effects
            variances: Study variances
            result: BayesianMAResult
            study_names: Optional list of study names

        Returns:
            Dictionary with forest plot data
        """
        if study_names is None:
            study_names = [f"Study {i+1}" for i in range(len(effects))]

        n = len(effects)

        # Credible intervals for each study
        study_cis = []
        for i in range(n):
            se = np.sqrt(variances[i])
            ci_lower = effects[i] - 1.96 * se
            ci_upper = effects[i] + 1.96 * se
            study_cis.append((ci_lower, ci_upper))

        return {
            'study_names': study_names,
            'effects': effects.tolist(),
            'cis': study_cis,
            'pooled_effect': result.posterior_mean,
            'pooled_ci': (result.ci_lower, result.ci_upper),
            'pooled_ci_95': result.hdi_95,
            'hdi_95': result.hdi_95
        }


if __name__ == "__main__":
    print("Advanced Bayesian Meta-Analysis Module loaded")
    print("Features:")
    print("  - Hierarchical Bayesian random-effects model")
    print("  - Robust Bayesian model (Student-t likelihood)")
    print("  - Bayesian network meta-analysis")
    print("  - Posterior predictive checks")
    print("  - MCMC diagnostics (R-hat, ESS)")
