"""
Individual Participant Data (IPD) Meta-Analysis Module

Implements IPD meta-analysis which uses raw participant-level data
rather than aggregated study-level data.

Benefits:
- Time-to-event analysis
- Adjustment for participant covariates
- Subgroup analysis at individual level
- More powerful than aggregate data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from scipy import stats
from lifelines import CoxPHFitter, KaplanMeierFitter
import warnings


@dataclass
class IPDStudy:
    """Individual participant data from a single study"""
    study_id: str
    data: pd.DataFrame
    participant_id_col: str
    treatment_col: str
    outcome_col: str
    time_col: Optional[str] = None  # For time-to-event
    event_col: Optional[str] = None   # For time-to-event
    covariates: Optional[List[str]] = None


@dataclass
class IPDResult:
    """Results from IPD meta-analysis"""
    method: str
    n_studies: int
    n_participants: int
    pooled_effect: float
    pooled_se: float
    pooled_ci: Tuple[float, float]
    p_value: float
    heterogeneity: Dict[str, float]
    covariate_effects: Optional[Dict[str, Dict[str, float]]] = None
    survival_curves: Optional[Dict[str, pd.DataFrame]] = None


class IPDMetaAnalyzer:
    """
    Individual participant data meta-analysis.

    Implements:
    - One-stage IPD meta-analysis (pooled analysis)
    - Two-stage IPD meta-analysis
    - Time-to-event analysis (Cox regression)
    - Subgroup detection
    - Covariate adjustment
    """

    def __init__(self):
        self.studies: List[IPDStudy] = []
        self.pooled_data: Optional[pd.DataFrame] = None

    def add_study(
        self,
        study_id: str,
        data: pd.DataFrame,
        participant_id: str,
        treatment: str,
        outcome: str,
        time_col: Optional[str] = None,
        event_col: Optional[str] = None,
        covariates: Optional[List[str]] = None
    ) -> None:
        """
        Add an IPD study.

        :param study_id: Study identifier
        :param data: DataFrame with participant-level data
        :param participant_id: Column name for participant ID
        :param treatment: Column name for treatment assignment
        :param outcome: Column name for outcome
        :param time_col: Column name for time-to-event
        :param event_col: Column name for event indicator
        :param covariates: List of covariate column names
        """
        # Validate required columns
        required_cols = [participant_id, treatment, outcome]
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data")

        if time_col and time_col not in data.columns:
            warnings.warn(f"Time column '{time_col}' not found, using binary outcome")
            time_col = None

        if event_col and event_col not in data.columns:
            warnings.warn(f"Event column '{event_col}' not found")
            event_col = None

        study = IPDStudy(
            study_id=study_id,
            data=data.copy(),
            participant_id_col=participant_id,
            treatment_col=treatment,
            outcome_col=outcome,
            time_col=time_col,
            event_col=event_col,
            covariates=covariates
        )

        self.studies.append(study)

    def pool_data(self) -> pd.DataFrame:
        """
        Pool all participant data into a single dataset.

        :return: Combined DataFrame with study_id added
        """
        if not self.studies:
            raise ValueError("No studies to pool")

        pooled_dfs = []

        for study in self.studies:
            df = study.data.copy()
            df['_study_id'] = study.study_id
            df['_participant_id'] = df[study.participant_id_col]
            df['_treatment'] = df[study.treatment_col]
            df['_outcome'] = df[study.outcome_col]

            if study.time_col:
                df['_time'] = df[study.time_col]
            if study.event_col:
                df['_event'] = df[study.event_col]

            pooled_dfs.append(df)

        self.pooled_data = pd.concat(pooled_dfs, ignore_index=True)
        return self.pooled_data

    def one_stage_analysis(
        self,
        model_type: str = "logistic",
        adjust_for_study: bool = True,
        covariates: Optional[List[str]] = None
    ) -> IPDResult:
        """
        One-stage IPD meta-analysis using regression on pooled data.

        :param model_type: Type of model ('logistic', 'cox', 'linear')
        :param adjust_for_study: Include study as random effect
        :param covariates: Covariates to adjust for
        :return: IPDResult
        """
        if self.pooled_data is None:
            self.pool_data()

        data = self.pooled_data
        n_studies = data['_study_id'].nunique()
        n_participants = len(data)

        if model_type == "cox":
            return self._cox_analysis(data, adjust_for_study, covariates)
        elif model_type == "logistic":
            return self._logistic_analysis(data, adjust_for_study, covariates)
        elif model_type == "linear":
            return self._linear_analysis(data, adjust_for_study, covariates)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _cox_analysis(
        self,
        data: pd.DataFrame,
        adjust_for_study: bool,
        covariates: Optional[List[str]]
    ) -> IPDResult:
        """Cox proportional hazards analysis"""
        # Check for required columns
        if '_time' not in data.columns or '_event' not in data.columns:
            raise ValueError("Time-to-event data not available")

        # Prepare formula
        treatment_effect = []
        study_effects = {}

        # Fit Cox model with study stratification
        fitter = CoxPHFitter()

        # Build formula string
        covariate_list = ['_treatment']
        if covariates:
            covariate_list.extend(covariates)

        formula = ' + '.join(covariate_list)

        # Stratify by study if requested
        if adjust_for_study:
            strata = data['_study_id']
        else:
            strata = None

        try:
            fitter.fit(data, duration_col='_time', event_col='_event',
                      formula=formula, strata=strata)

            # Extract results
            summary = fitter.summary
            treatment_idx = list(summary.index).index('_treatment')

            hr = summary['coef'].iloc[treatment_idx]
            hr_se = summary['se(coef)'].iloc[treatment_idx]
            hr_ci_lower = summary['coef lower 95%'].iloc[treatment_idx]
            hr_ci_upper = summary['coef upper 95%'].iloc[treatment_idx]
            p_value = summary['p'].iloc[treatment_idx]

            # Log HR
            log_hr = np.log(hr)
            log_hr_se = hr_se / hr

            # Covariate effects
            cov_effects = None
            if covariates:
                cov_effects = {}
                for cov in covariates:
                    if cov in summary.index:
                        cov_effects[cov] = {
                            'hr': np.exp(summary['coef'].loc[cov]),
                            'ci_lower': np.exp(summary['coef lower 95%'].loc[cov]),
                            'ci_upper': np.exp(summary['coef upper 95%'].loc[cov]),
                            'p_value': summary['p'].loc[cov]
                        }

            # Survival curves by study
            survival_curves = {}
            kmf = KaplanMeierFitter()
            for study_id in data['_study_id'].unique():
                study_data = data[data['_study_id'] == study_id]

                for treatment in study_data['_treatment'].unique():
                    treatment_data = study_data[study_data['_treatment'] == treatment]

                    kmf.fit(treatment_data['_time'], 1 - treatment_data['_event'])

                    if study_id not in survival_curves:
                        survival_curves[study_id] = {}
                    survival_curves[study_id][treatment] = kmf.survival_function_

            return IPDResult(
                method="one_stage_cox",
                n_studies=n_studies,
                n_participants=len(data),
                pooled_effect=log_hr,  # Log hazard ratio
                pooled_se=log_hr_se,
                pooled_ci=(log_hr_ci_lower, log_hr_ci_upper),
                p_value=p_value,
                heterogeneity={},  # Would need two-stage for this
                covariate_effects=cov_effects,
                survival_curves=survival_curves
            )

        except Exception as e:
            raise RuntimeError(f"Cox regression failed: {e}")

    def _logistic_analysis(
        self,
        data: pd.DataFrame,
        adjust_for_study: bool,
        covariates: Optional[List[str]]
    ) -> IPDResult:
        """Logistic regression analysis"""
        import statsmodels.api as sm
        import statsmodels.formula.api as smf

        # Build formula
        formula = "_outcome ~ _treatment"
        if covariates:
            formula += " + " + " + ".join(covariates)

        if adjust_for_study:
            formula += " + C(_study_id)"

        try:
            # Fit logistic regression
            model = smf.logit(formula=formula, data=data).fit(disp=0)

            # Extract treatment effect
            treatment_coef = model.params.get('_treatment')
            treatment_se = model.bse.get('_treatment')
            conf_int = model.conf_int().loc['_treatment']

            p_value = model.pvalues.get('_treatment')

            # Convert to log odds ratio
            lor = treatment_coef
            lor_se = treatment_se

            # Covariate effects
            cov_effects = None
            if covariates:
                cov_effects = {}
                for cov in covariates:
                    if cov in model.params:
                        cov_effects[cov] = {
                            'or': np.exp(model.params[cov]),
                            'ci_lower': np.exp(conf_int.loc[cov, 0]),
                            'ci_upper': np.exp(conf_int.loc[cov, 1]),
                            'p_value': model.pvalues[cov]
                        }

            return IPDResult(
                method="one_stage_logistic",
                n_studies=data['_study_id'].nunique(),
                n_participants=len(data),
                pooled_effect=lor,
                pooled_se=lor_se,
                pooled_ci=(conf_int[0], conf_int[1]),
                p_value=p_value,
                heterogeneity={},
                covariate_effects=cov_effects
            )

        except Exception as e:
            raise RuntimeError(f"Logistic regression failed: {e}")

    def _linear_analysis(
        self,
        data: pd.DataFrame,
        adjust_for_study: bool,
        covariates: Optional[List[str]]
    ) -> IPDResult:
        """Linear regression analysis for continuous outcomes"""
        import statsmodels.api as sm
        import statsmodels.formula.api as smf

        formula = "_outcome ~ _treatment"
        if covariates:
            formula += " + " + " + ".join(covariates)

        if adjust_for_study:
            formula += " + C(_study_id)"

        try:
            model = smf.ols(formula=formula, data=data).fit()

            treatment_coef = model.params.get('_treatment')
            treatment_se = model.bse.get('_treatment')
            conf_int = model.conf_int().loc['_treatment']
            p_value = model.pvalues.get('_treatment')

            cov_effects = None
            if covariates:
                cov_effects = {}
                for cov in covariates:
                    if cov in model.params:
                        cov_effects[cov] = {
                            'coef': model.params[cov],
                            'ci_lower': conf_int.loc[cov, 0],
                            'ci_upper': conf_int.loc[cov, 1],
                            'p_value': model.pvalues[cov]
                        }

            return IPDResult(
                method="one_stage_linear",
                n_studies=data['_study_id'].nunique(),
                n_participants=len(data),
                pooled_effect=treatment_coef,
                pooled_se=treatment_se,
                pooled_ci=(conf_int[0], conf_int[1]),
                p_value=p_value,
                heterogeneity={},
                covariate_effects=cov_effects
            )

        except Exception as e:
            raise RuntimeError(f"Linear regression failed: {e}")

    def two_stage_analysis(
        self,
        outcome_type: str = "binary"
    ) -> IPDResult:
        """
        Two-stage IPD meta-analysis.

        Stage 1: Estimate treatment effect within each study
        Stage 2: Pool study estimates using meta-analysis

        :param outcome_type: Type of outcome ('binary' or 'continuous')
        :return: IPDResult
        """
        from statsmodels.stats.meta_analysis import effectsize_2proportions, effectsize_smd

        study_effects = []
        study_variances = []
        study_ns = []

        for study in self.studies:
            data = study.data
            treatment_col = study.treatment_col
            outcome_col = study.outcome_col

            # Calculate study-specific effect
            if outcome_type == "binary":
                # Get 2x2 table
                treatment_data = data[data[treatment_col] == 1]
                control_data = data[data[treatment_col] == 0]

                events_t = treatment_data[outcome_col].sum()
                total_t = len(treatment_data)
                events_c = control_data[outcome_col].sum()
                total_c = len(control_data)

                effect, var_effect = effectsize_2proportions(
                    np.array([events_t]),
                    np.array([total_t]),
                    np.array([events_c]),
                    np.array([total_c]),
                    statistic='odds-ratio',
                    zero_correction=0.5
                )
                effect = effect[0]
                var_effect = var_effect[0]

                study_ns.append(total_t + total_c)

            else:  # continuous
                treatment_mean = data[data[treatment_col] == 1][outcome_col].mean()
                control_mean = data[data[treatment_col] == 0][outcome_col].mean()
                treatment_sd = data[data[treatment_col] == 1][outcome_col].std()
                control_sd = data[data[treatment_col] == 0][outcome_col].std()
                n_t = len(data[data[treatment_col] == 1])
                n_c = len(data[data[treatment_col] == 0])

                effect, var_effect = effectsize_smd(
                    np.array([treatment_mean]),
                    np.array([treatment_sd]),
                    np.array([n_t]),
                    np.array([control_mean]),
                    np.array([control_sd]),
                    np.array([n_c])
                )
                effect = effect[0]
                var_effect = var_effect[0]

                study_ns.append(n_t + n_c)

            study_effects.append(effect)
            study_variances.append(var_effect)

        # Stage 2: Pool using meta-analysis
        effects = np.array(study_effects)
        variances = np.array(study_variances)

        # DerSimonian-Laird random-effects
        weights = 1 / variances
        weighted_mean = np.sum(weights * effects) / np.sum(weights)
        q = np.sum(weights * (effects - weighted_mean)**2)
        df = len(effects) - 1

        if q > df:
            sum_w = np.sum(weights)
            sum_w2 = np.sum(weights**2)
            tau2 = (q - df) / (sum_w - sum_w2 / sum_w)
        else:
            tau2 = 0

        # Re-calculate weights with tau2
        re_weights = 1 / (variances + tau2)
        pooled_effect = np.sum(re_weights * effects) / np.sum(re_weights)
        se = np.sqrt(1 / np.sum(re_weights))

        # CI
        ci_lower = pooled_effect - 1.96 * se
        ci_upper = pooled_effect + 1.96 * se

        # P-value
        z = pooled_effect / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        # Heterogeneity
        i_squared = max(0, 100 * (q - df) / q) if q > 0 else 0

        return IPDResult(
            method="two_stage",
            n_studies=len(self.studies),
            n_participants=sum(study_ns),
            pooled_effect=pooled_effect,
            pooled_se=se,
            pooled_ci=(ci_lower, ci_upper),
            p_value=p_value,
            heterogeneity={
                'q': q,
                'df': df,
                'i_squared': i_squared,
                'tau_squared': tau2
            }
        )

    def subgroup_analysis_ipd(
        self,
        subgroup_var: str,
        min_n_per_subgroup: int = 50
    ) -> Dict[str, IPDResult]:
        """
        Perform subgroup analysis using individual participant data.

        :param subgroup_var: Variable to subgroup by
        :param min_n_per_subgroup: Minimum sample size per subgroup
        :return: Dictionary mapping subgroup names to results
        """
        if self.pooled_data is None:
            self.pool_data()

        data = self.pooled_data

        # Identify subgroups
        subgroups = data[subgroup_var].unique()

        results = {}

        for subgroup in subgroups:
            subgroup_data = data[data[subgroup_var] == subgroup]

            if len(subgroup_data) < min_n_per_subgroup:
                continue

            # Create temporary analyzer for this subgroup
            temp_analyzer = IPDMetaAnalyzer()

            # Add as a single "study"
            temp_analyzer.add_study(
                study_id=f"subgroup_{subgroup}",
                data=subgroup_data,
                participant_id='_participant_id',
                treatment='_treatment',
                outcome='_outcome',
                covariates=None
            )

            # Analyze
            try:
                result = temp_analyzer.one_stage_analysis(
                    model_type="logistic",
                    adjust_for_study=False
                )
                results[str(subgroup)] = result
            except Exception as e:
                warnings.warn(f"Subgroup {subgroup} analysis failed: {e}")

        return results

    def test_interaction(
        self,
        subgroup_var: str,
        covariates: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Test for interaction between treatment and subgroup variable.

        :param subgroup_var: Variable to test interaction with
        :param covariates: Additional covariates to adjust for
        :return: Dictionary with test results
        """
        if self.pooled_data is None:
            self.pool_data()

        data = self.pooled_data

        # Build formula with interaction
        formula = f"_outcome ~ _treatment * C({subgroup_var})"
        if covariates:
            formula += " + " + " + ".join(covariates)

        # Fit logistic regression with interaction
        import statsmodels.formula.api as smf

        try:
            model = smf.logit(formula=formula, data=data).fit(disp=0)

            # Extract interaction term
            interaction_term = f"_treatment:C({subgroup_var})[T.{data[subgroup_var].dtype.type}]"

            # Find the actual interaction term in the model
            interaction_coef = None
            interaction_p = None

            for param in model.params.index:
                if 'treatment' in param and subgroup_var in param:
                    interaction_coef = model.params[param]
                    interaction_p = model.pvalues[param]
                    break

            if interaction_coef is None:
                # Try alternate naming
                for param in model.params.index:
                    if ':' in param or '*' in param:
                        interaction_coef = model.params[param]
                        interaction_p = model.pvalues[param]
                        break

            if interaction_p is None:
                return {
                    'interaction_coefficient': None,
                    'p_value': None,
                    'significant': False,
                    'error': 'Could not find interaction term'
                }

            return {
                'interaction_coefficient': float(interaction_coef),
                'p_value': float(interaction_p),
                'significant': interaction_p < 0.05,
                'interpretation': self._interpret_interaction(interaction_p)
            }

        except Exception as e:
            return {
                'interaction_coefficient': None,
                'p_value': None,
                'significant': False,
                'error': str(e)
            }

    def _interpret_interaction(self, p_value: float) -> str:
        """Interpret interaction p-value"""
        if p_value < 0.001:
            return "Strong evidence of treatment effect modification"
        elif p_value < 0.01:
            return "Very strong evidence of interaction"
        elif p_value < 0.05:
            return "Significant interaction (p < 0.05)"
        elif p_value < 0.10:
            return "Suggestive evidence of interaction"
        else:
            return "No significant interaction"

    def create_participant_level_forest_plot(
        self,
        time_horizons: Optional[List[float]] = None
    ) -> None:
        """
        Create forest plot showing participant-level data.

        :param time_horizons: Time points for survival curves
        """
        if self.pooled_data is None:
            self.pool_data()

        import matplotlib.pyplot as plt

        data = self.pooled_data

        # Calculate study-level effects from IPD
        study_effects = {}
        for study_id in data['_study_id'].unique():
            study_data = data[data['_study_id'] == study_id]

            # Simple treatment effect
            treated = study_data[study_data['_treatment'] == 1]['_outcome']
            control = study_data[study_data['_treatment'] == 0]['_outcome']

            if len(treated) > 0 and len(control) > 0:
                effect = treated.mean() - control.mean()
                n = len(treated) + len(control)
                study_effects[study_id] = {'effect': effect, 'n': n}

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))

        studies = list(study_effects.keys())
        effects = [study_effects[s]['effect'] for s in studies]
        ns = [study_effects[s]['n'] for s in studies]

        y_pos = np.arange(len(studies))
        ax.scatter(effects, y_pos, s=[n/50 for n in ns], alpha=0.6)

        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(studies)
        ax.set_xlabel('Treatment Effect')
        ax.set_title('Participant-Level Forest Plot')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("IPD Meta-Analysis module loaded")
    print("Features:")
    print("  - One-stage pooled analysis")
    print("  - Two-stage analysis")
    print("  - Cox regression for time-to-event")
    print("  - Logistic regression for binary outcomes")
    print("  - Subgroup analysis at individual level")
    print("  - Interaction testing")
    print("  - Participant-level forest plots")
