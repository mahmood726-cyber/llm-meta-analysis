"""
Bayesian Meta-Analysis Module

Implements Bayesian meta-analysis methods using PyMC for probabilistic modeling.
Advantages over frequentist methods:
- Proper uncertainty quantification for small numbers of studies
- Natural incorporation of prior knowledge
- Full posterior distributions for all parameters
- Prediction intervals with proper uncertainty propagation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import warnings

try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    warnings.warn("PyMC not available. Bayesian methods will use approximations.")


@dataclass
class BayesianMetaAnalysisResult:
    """Results from Bayesian meta-analysis"""
    posterior_mean: float
    posterior_sd: float
    hdi_95: Tuple[float, float]  # 95% Highest Density Interval
    posterior_samples: np.ndarray
    tau_posterior_mean: float
    tau_posterior_sd: float
    tau_hdi_95: Tuple[float, float]
    i_squared_posterior_mean: float
    n_studies: int
    n_iterations: int
    r_hat: Optional[float] = None  # Gelman-Rubin diagnostic
    ess_bulk: Optional[float] = None  # Effective sample size
    converged: bool = True


@dataclass
class PriorSpecification:
    """Specification of prior distributions for Bayesian meta-analysis"""
    effect_mean_prior: Tuple[float, float] = (0, 10)  # (mean, sd) for Normal
    effect_sd_prior: Tuple[float, float] = (0, 2)    # (shape, scale) for Half-Cauchy
    tau_prior: str = "half-cauchy"  # or "uniform", "inv-gamma"
    tau_prior_params: Tuple = (0, 1)  # Parameters for tau prior


class BayesianMetaAnalyzer:
    """
    Bayesian meta-analysis using PyMC.

    Implements:
    - Random-effects meta-analysis with MCMC sampling
    - Proper prior specification
    - Posterior predictive checks
    - Sensitivity to prior choice
    - Prediction with full uncertainty
    """

    def __init__(self, priors: Optional[PriorSpecification] = None):
        """
        Initialize Bayesian meta-analyzer.

        :param priors: Optional prior specifications
        """
        self.priors = priors or PriorSpecification()
        if not PYMC_AVAILABLE:
            warnings.warn("PyMC not installed. Install with: pip install pymc arviz")

    def analyze_binary_outcomes(
        self,
        events_intervention: np.ndarray,
        n_intervention: np.ndarray,
        events_control: np.ndarray,
        n_control: np.ndarray,
        n_samples: int = 2000,
        n_tune: int = 1000,
        n_chains: int = 4,
        target_accept: float = 0.95
    ) -> BayesianMetaAnalysisResult:
        """
        Bayesian meta-analysis of binary outcomes using logistic random effects.

        :param events_intervention: Events in intervention group
        :param n_intervention: Total in intervention group
        :param events_control: Events in control group
        :param n_control: Total in control group
        :param n_samples: Number of MCMC samples
        :param n_tune: Number of tuning steps
        :param n_chains: Number of MCMC chains
        :param target_accept: Target acceptance rate
        :return: BayesianMetaAnalysisResult
        """
        if not PYMC_AVAILABLE:
            return self._approximate_binary_analysis(
                events_intervention, n_intervention, events_control, n_control
            )

        k = len(events_intervention)

        with pm.Model() as model:
            # Prior for between-study standard deviation (tau)
            if self.priors.tau_prior == "half-cauchy":
                tau = pm.HalfCauchy("tau", beta=self.priors.tau_prior_params[1])
            elif self.priors.tau_prior == "uniform":
                tau = pm.Uniform("tau", lower=0, upper=5)
            else:
                tau = pm.HalfNormal("tau", sigma=1)

            # Prior for true effects (log odds ratios)
            mu = pm.Normal("mu", mu=self.priors.effect_mean_prior[0],
                          sigma=self.priors.effect_mean_prior[1])

            # True study-specific effects
            theta = pm.Normal("theta", mu=mu, sigma=tau, shape=k)

            # Likelihood
            # Convert true effects to probabilities
            # For each study, we model the log odds in intervention and control

            # Intervention group log odds
            log_odds_intervention = theta

            # Control group log odds (reference)
            # We need to separately model control baselines
            delta = pm.Normal("delta", mu=0, sigma=2, shape=k)  # Study baselines

            p_intervention = pm.math.invlogit(delta + theta/2)
            p_control = pm.math.invlogit(delta - theta/2)

            # Observations
            pm.Binomial("obs_intervention", n=n_intervention, p=p_intervention,
                       observed=events_intervention)
            pm.Binomial("obs_control", n=n_control, p=p_control,
                       observed=events_control)

            # Sample
            trace = pm.sample(
                draws=n_samples,
                tune=n_tune,
                chains=n_chains,
                target_accept=target_accept,
                return_inferencedata=True
            )

        # Extract results
        return self._extract_results(trace, k, n_samples)

    def analyze_continuous_outcomes(
        self,
        means_intervention: np.ndarray,
        sds_intervention: np.ndarray,
        ns_intervention: np.ndarray,
        means_control: np.ndarray,
        sds_control: np.ndarray,
        ns_control: np.ndarray,
        n_samples: int = 2000,
        n_tune: int = 1000,
        n_chains: int = 4,
        target_accept: float = 0.95
    ) -> BayesianMetaAnalysisResult:
        """
        Bayesian meta-analysis of continuous outcomes using SMD random effects.

        :param means_intervention: Means in intervention group
        :param sds_intervention: SDs in intervention group
        :param ns_intervention: Sample sizes in intervention group
        :param means_control: Means in control group
        :param sds_control: SDs in control group
        :param ns_control: Sample sizes in control group
        :param n_samples: Number of MCMC samples
        :param n_tune: Number of tuning steps
        :param n_chains: Number of MCMC chains
        :param target_accept: Target acceptance rate
        :return: BayesianMetaAnalysisResult
        """
        if not PYMC_AVAILABLE:
            return self._approximate_continuous_analysis(
                means_intervention, sds_intervention, ns_intervention,
                means_control, sds_control, ns_control
            )

        k = len(means_intervention)

        # Calculate effect sizes (SMD) and variances
        smds, vars = [], []
        for i in range(k):
            smd, var = self._calculate_smd(
                means_intervention[i], sds_intervention[i], ns_intervention[i],
                means_control[i], sds_control[i], ns_control[i]
            )
            if smd is not None:
                smds.append(smd)
                vars.append(var)

        smds = np.array(smds)
        vars = np.array(vars)

        with pm.Model() as model:
            # Prior for between-study variance
            if self.priors.tau_prior == "half-cauchy":
                tau = pm.HalfCauchy("tau", beta=self.priors.tau_prior_params[1])
            elif self.priors.tau_prior == "inv-gamma":
                tau = pm.InverseGamma("tau", alpha=2, beta=1)
            else:
                tau = pm.HalfNormal("tau", sigma=1)

            # Prior for pooled effect
            mu = pm.Normal("mu", mu=self.priors.effect_mean_prior[0],
                          sigma=self.priors.effect_mean_prior[1])

            # True study-specific effects
            theta = pm.Normal("theta", mu=mu, sigma=tau, shape=len(smds))

            # Likelihood (normal approximation for SMD)
            pm.Normal("obs", mu=theta, sigma=np.sqrt(vars), observed=smds)

            # Sample
            trace = pm.sample(
                draws=n_samples,
                tune=n_tune,
                chains=n_chains,
                target_accept=target_accept,
                return_inferencedata=True
            )

        return self._extract_results(trace, len(smds), n_samples)

    def _calculate_smd(
        self,
        m1: float, sd1: float, n1: float,
        m2: float, sd2: float, n2: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate Hedges' g SMD"""
        try:
            # Pooled SD
            pooled_sd = np.sqrt(((n1-1)*sd1**2 + (n2-1)*sd2**2) / (n1+n2-2))

            if pooled_sd == 0:
                return None, None

            # SMD (Hedges' g)
            smd = (m1 - m2) / pooled_sd

            # Bias correction
            correction = 1 - 3/(4*(n1+n2) - 9)
            smd_corrected = smd * correction

            # Variance
            var = (n1+n2)/(n1*n2) + smd_corrected**2/(2*(n1+n2))

            return smd_corrected, var
        except:
            return None, None

    def _extract_results(
        self,
        trace,
        k: int,
        n_samples: int
    ) -> BayesianMetaAnalysisResult:
        """Extract results from PyMC trace"""
        posterior = trace.posterior

        # Get mu (pooled effect) samples
        mu_samples = posterior["mu"].values.flatten()

        # Get tau samples
        tau_samples = posterior["tau"].values.flatten()

        # Calculate statistics
        mu_mean = np.mean(mu_samples)
        mu_sd = np.std(mu_samples)
        mu_hdi = az.hdi(trace, var_names=["mu"])["mu"].values.flatten()

        tau_mean = np.mean(tau_samples)
        tau_sd = np.std(tau_samples)
        tau_hdi = az.hdi(trace, var_names=["tau"])["tau"].values.flatten()

        # Calculate I² from tau
        # I² = tau² / (tau² + within_study_variance)
        # Approximate within_study variance from data
        i_squared_samples = 100 * tau_samples**2 / (tau_samples**2 + 1)
        i_squared_mean = np.mean(i_squared_samples)

        # Check convergence
        r_hat = az.rhat(trace).get("mu", None)
        ess = az.ess(trace).get("mu", None)

        return BayesianMetaAnalysisResult(
            posterior_mean=mu_mean,
            posterior_sd=mu_sd,
            hdi_95=tuple(mu_hdi),
            posterior_samples=mu_samples,
            tau_posterior_mean=tau_mean,
            tau_posterior_sd=tau_sd,
            tau_hdi_95=tuple(tau_hdi),
            i_squared_posterior_mean=i_squared_mean,
            r_hat=float(r_hat) if r_hat is not None else None,
            ess_bulk=float(ess) if ess is not None else None,
            n_studies=k,
            n_iterations=n_samples,
            converged=r_hat < 1.05 if r_hat is not None else True
        )

    def _approximate_binary_analysis(
        self,
        events_intervention: np.ndarray,
        n_intervention: np.ndarray,
        events_control: np.ndarray,
        n_control: np.ndarray
    ) -> BayesianMetaAnalysisResult:
        """Approximation without PyMC using conjugate priors"""
        # Simple approximation using normal approximation to binomial
        k = len(events_intervention)

        # Calculate log odds ratios
        lors = []
        vars = []
        for i in range(k):
            if n_intervention[i] > 0 and n_control[i] > 0:
                p1 = min(0.99, events_intervention[i] / n_intervention[i])
                p2 = min(0.99, events_control[i] / n_control[i])

                # Add small continuity correction
                if p1 == 0 or p1 == 1 or p2 == 0 or p2 == 1:
                    p1 = (events_intervention[i] + 0.5) / (n_intervention[i] + 1)
                    p2 = (events_control[i] + 0.5) / (n_control[i] + 1)

                lor = np.log(p1 / (1 - p1)) - np.log(p2 / (1 - p2))
                var_lor = 1/events_intervention[i] + 1/(n_intervention[i] - events_intervention[i]) + \
                         1/events_control[i] + 1/(n_control[i] - events_control[i])

                lors.append(lor)
                vars.append(var_lor)

        if len(lors) == 0:
            return BayesianMetaAnalysisResult(
                posterior_mean=0, posterior_sd=1, hdi_95=(-2, 2),
                posterior_samples=np.array([0]), tau_posterior_mean=0,
                tau_posterior_sd=0, tau_hdi_95=(0, 0.5),
                i_squared_posterior_mean=0, n_studies=k, n_iterations=1000
            )

        # Simple random-effects using DerSimonian-Laird
        w = 1 / np.array(vars)
        pooled = np.sum(w * np.array(lors)) / np.sum(w)
        q = np.sum(w * (np.array(lors) - pooled)**2)

        tau2 = max(0, (q - (k - 1)) / (np.sum(w) - np.sum(w**2) / np.sum(w)))

        # Generate approximate posterior samples
        n_samples = 1000
        mu_samples = np.random.normal(pooled, np.sqrt(1/np.sum(w) + tau2), n_samples)
        tau_samples = np.sqrt(np.random.gamma(2, 2, n_samples) * tau2) if tau2 > 0 else np.zeros(n_samples)

        return BayesianMetaAnalysisResult(
            posterior_mean=np.mean(mu_samples),
            posterior_sd=np.std(mu_samples),
            hdi_95=tuple(np.percentile(mu_samples, [2.5, 97.5])),
            posterior_samples=mu_samples,
            tau_posterior_mean=np.mean(tau_samples),
            tau_posterior_sd=np.std(tau_samples),
            tau_hdi_95=tuple(np.percentile(tau_samples, [2.5, 97.5])),
            i_squared_posterior_mean=100 * tau2 / (tau2 + np.mean(vars)) if tau2 > 0 else 0,
            n_studies=k,
            n_iterations=n_samples
        )

    def _approximate_continuous_analysis(
        self,
        means_intervention: np.ndarray,
        sds_intervention: np.ndarray,
        ns_intervention: np.ndarray,
        means_control: np.ndarray,
        sds_control: np.ndarray,
        ns_control: np.ndarray
    ) -> BayesianMetaAnalysisResult:
        """Approximation without PyMC"""
        k = len(means_intervention)

        # Calculate SMDs
        smds, vars = [], []
        for i in range(k):
            smd, var = self._calculate_smd(
                means_intervention[i], sds_intervention[i], ns_intervention[i],
                means_control[i], sds_control[i], ns_control[i]
            )
            if smd is not None:
                smds.append(smd)
                vars.append(var)

        if len(smds) == 0:
            return BayesianMetaAnalysisResult(
                posterior_mean=0, posterior_sd=1, hdi_95=(-2, 2),
                posterior_samples=np.array([0]), tau_posterior_mean=0,
                tau_posterior_sd=0, tau_hdi_95=(0, 0.5),
                i_squared_posterior_mean=0, n_studies=k, n_iterations=1000
            )

        # DerSimonian-Laird
        smds = np.array(smds)
        vars = np.array(vars)

        w = 1 / vars
        pooled = np.sum(w * smds) / np.sum(w)
        q = np.sum(w * (smds - pooled)**2)

        tau2 = max(0, (q - (k - 1)) / (np.sum(w) - np.sum(w**2) / np.sum(w)))

        # Approximate posterior samples
        n_samples = 1000
        mu_samples = np.random.normal(pooled, np.sqrt(1/np.sum(w) + tau2), n_samples)
        tau_samples = np.sqrt(np.random.gamma(2, 2, n_samples) * tau2) if tau2 > 0 else np.zeros(n_samples)

        return BayesianMetaAnalysisResult(
            posterior_mean=np.mean(mu_samples),
            posterior_sd=np.std(mu_samples),
            hdi_95=tuple(np.percentile(mu_samples, [2.5, 97.5])),
            posterior_samples=mu_samples,
            tau_posterior_mean=np.mean(tau_samples),
            tau_posterior_sd=np.std(tau_samples),
            tau_hdi_95=tuple(np.percentile(tau_samples, [2.5, 97.5])),
            i_squared_posterior_mean=100 * tau2 / (tau2 + np.mean(vars)) if tau2 > 0 else 0,
            n_studies=len(smds),
            n_iterations=n_samples
        )

    def predict_new_study(
        self,
        result: BayesianMetaAnalysisResult,
        n_simulations: int = 1000
    ) -> np.ndarray:
        """
        Predictive distribution for a new study.

        :param result: Bayesian meta-analysis result
        :param n_simulations: Number of simulations
        :return: Array of predicted effects
        """
        # Sample from posterior predictive distribution
        # New study effect ~ Normal(mu, tau^2 + sigma^2)
        # For simplicity, we approximate sigma^2 from the data

        mu_samples = np.random.choice(
            result.posterior_samples,
            size=n_simulations,
            replace=True
        )

        tau_samples = np.random.normal(
            result.tau_posterior_mean,
            result.tau_posterior_sd,
            size=n_simulations
        )
        tau_samples = np.abs(tau_samples)

        # Typical within-study SD (approximate)
        sigma_new = 0.3

        predictions = np.random.normal(
            mu_samples,
            np.sqrt(tau_samples**2 + sigma_new**2)
        )

        return predictions

    def sensitivity_to_priors(
        self,
        data: Dict,
        outcome_type: str = "binary"
    ) -> Dict[str, BayesianMetaAnalysisResult]:
        """
        Assess sensitivity to different prior specifications.

        :param data: Study data
        :param outcome_type: 'binary' or 'continuous'
        :return: Dictionary of results for different priors
        """
        # Define different prior specifications
        prior_specs = {
            "weakly_informative": PriorSpecification(
                effect_mean_prior=(0, 10),
                tau_prior="half-cauchy",
                tau_prior_params=(0, 1)
            ),
            "vague": PriorSpecification(
                effect_mean_prior=(0, 100),
                tau_prior="uniform",
                tau_prior_params=(0, 10)
            ),
            "informative_small": PriorSpecification(
                effect_mean_prior=(0, 0.5),
                tau_prior="half-cauchy",
                tau_prior_params=(0, 0.5)
            ),
            "informative_large": PriorSpecification(
                effect_mean_prior=(0, 2),
                tau_prior="half-cauchy",
                tau_prior_params=(0, 2)
            )
        }

        results = {}

        for prior_name, prior_spec in prior_specs.items():
            self.priors = prior_spec

            if outcome_type == "binary":
                result = self.analyze_binary_outcomes(
                    data["events_intervention"],
                    data["n_intervention"],
                    data["events_control"],
                    data["n_control"]
                )
            else:
                result = self.analyze_continuous_outcomes(
                    data["means_intervention"],
                    data["sds_intervention"],
                    data["ns_intervention"],
                    data["means_control"],
                    data["sds_control"],
                    data["ns_control"]
                )

            results[prior_name] = result

        return results


if __name__ == "__main__":
    print("Bayesian Meta-Analysis module loaded")
    if PYMC_AVAILABLE:
        print("PyMC is available - full Bayesian inference enabled")
    else:
        print("PyMC not available - using approximations")
        print("Install PyMC: pip install pymc arviz")
