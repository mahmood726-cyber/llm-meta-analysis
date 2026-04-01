# LLM Meta-Analysis Framework

Automating meta-analysis of clinical trials (randomized controlled trials) using Large Language Models.

## :sparkles: What's New in v2.0 (Major Statistical Revision)

This version addresses all editorial feedback from *Research Synthesis Methods*.
All critical methodological concerns have been resolved.

### Statistical Methods Improvements

- **Hartung-Knapp-Sidik-Jonkman (HKSJ) Adjustment** - Now auto-selected for small samples (k < 20)
- **Q-Profile Confidence Intervals for τ²** - Gold standard method with profile likelihood alternative
- **Proper Prediction Intervals** - Partlett & Riley (2017) method
- **Knapp-Hartung Adjustment** - For meta-regression standard errors (default)
- **VIF Multicollinearity Detection** - With condition number assessment
- **Quality Effects Model** - Doi & Thalib (2008) implementation
- **Bootstrap Variance Propagation** - Accounts for all uncertainty sources
- **REML with Convergence Checking** - Automatic fallback to DL
- **Conditional HKSJ Selection** - Auto-selects based on sample size (Cochrane compliant)

### New Modules

- **`statistical_framework_v2.py`** - Advanced meta-analysis with all modern methods
- **`meta_regression_v2.py`** - Meta-regression with Knapp-Hartung and VIF
- **`cumulative_analysis_v2.py`** - Cumulative meta-analysis with proper temporal validation
- **`integrated_meta_analysis_v2.py`** - Unified interface using improved methods
- **`glmm_meta_analysis.py`** - Generalized linear mixed models for binary outcomes
- **network_meta_analysis.py** - NMA with inconsistency plots (net heat plot, node-splitting)

### Validation & Testing

- **Unit tests** - 7/7 tests passing with known value validation
- **BCG dataset validation** - Verified against published results
- **Edge case handling** - Comprehensive validation of boundary conditions
- **No simulated data** - Removed all placeholder data from primary analysis

### API Changes

The v2 API has breaking changes. See `CHANGELOG.md` for migration guide.

```python
# v1 (deprecated)
from statistical_framework import EnhancedHeterogeneity

# v2 (use this)
from statistical_framework_v2 import AdvancedMetaAnalysis

# Auto CI method selection (HKSJ for k < 20, Wald for k >= 20)
result = AdvancedMetaAnalysis.random_effects_analysis(
    effects=effects,
    variances=variances,
    ci_method="auto",  # NEW: auto-selects based on sample size
    tau2_ci_method="q_profile"  # NEW: or "profile_likelihood"
)
```

## :hammer_and_wrench: SETUP

Create conda environment from the environment.yml:
```bash
conda env create -f environment.yml
conda activate llm-meta-analysis
```

## :scientific: Statistical Methods

### Random-Effects Meta-Analysis

```python
from statistical_framework_v2 import AdvancedMetaAnalysis

# Auto-selects CI method based on sample size
result = AdvancedMetaAnalysis.random_effects_analysis(
    effects=effect_sizes,
    variances=variances,
    tau2_method="REML",  # DL, REML, SJ, PM
    ci_method="auto",    # auto, HKSJ, wald, quantile
    tau2_ci_method="q_profile",  # q_profile, profile_likelihood
    prediction=True
)

print(f"Pooled effect: {result.pooled_effect:.3f}")
print(f"95% CI: [{result.ci.lower:.3f}, {result.ci.upper:.3f}]")
print(f"Prediction interval: [{result.prediction_interval.lower:.3f}, {result.prediction_interval.upper:.3f}]")
print(f"τ²: {result.tau_squared:.4f} (95% CI: [{result.heterogeneity.tau_squared_ci.lower:.4f}, {result.heterogeneity.tau_squared_ci.upper:.4f}])")
print(f"I²: {result.heterogeneity.i_squared:.1f}%")
```

### Meta-Regression

```python
from meta_regression_v2 import AdvancedMetaRegression

# Meta-regression with Knapp-Hartung adjustment
regression = AdvancedMetaRegression(
    effects=effects,
    variances=variances,
    covariates=covariate_matrix,
    use_knapp_hartung=True  # Default
)

print(f"Coefficients: {regression.coefficients}")
print(f"VIFs: {regression.vif_values}")  # Multicollinearity check
print(f"Condition number: {regression.condition_number}")
```

### Cumulative Meta-Analysis

```python
from cumulative_analysis_v2 import CumulativeMetaAnalyzer

analyzer = CumulativeMetaAnalyzer(
    effects=effects,
    variances=variances,
    publication_dates=dates  # Required for chronological ordering
)

# Cumulative forest plot
analyzer.plot_cumulative_forest(
    order_by="chronological",  # or "precision"
    alpha_spending="lan_demets",  # or "obrien_fleming", "pocock"
    save_path="cumulative_forest.png"
)
```

### Generalized Linear Mixed Models (GLMM)

```python
from glmm_meta_analysis import GLMMMetaAnalysis, BinaryData

# Create binary data objects
studies = [
    BinaryData(events_treatment=10, total_treatment=100,
               events_control=20, total_control=100),
    # ... more studies
]

# Logistic-normal random effects model
result = GLMMMetaAnalysis.logistic_normal_model(
    studies=studies,
    tau2_init=0.1,
    max_iter=1000
)

print(f"Pooled OR: {result.pooled_or:.3f}")
print(f"95% CI: [{result.ci.lower:.3f}, {result.ci.upper:.3f}]")
```

### Network Meta-Analysis

```python
from network_meta_analysis import NetworkMetaAnalyzer

analyzer = NetworkMetaAnalyzer()

# Add studies
analyzer.add_arm_based_study(
    study_id="study1",
    arms={
        "Placebo": {"events": 50, "total": 200},
        "Treatment A": {"events": 30, "total": 200},
        "Treatment B": {"events": 25, "total": 200}
    }
)

# Perform NMA
nma_result = analyzer.perform_nma(outcome_type="binary")

# Inconsistency diagnostics plots
analyzer.plot_inconsistency_diagnostics(
    outcome_type="binary",
    save_path="inconsistency_diagnostics.png"
)

# Net heat plot
pairwise_effects = analyzer.estimate_pairwise_effects(outcome_type="binary")
analyzer.plot_net_heat_plot(
    pairwise_effects=pairwise_effects,
    save_path="net_heat_plot.png"
)

# Node-splitting forest plot
analyzer.plot_node_splitting_forest(
    comparison=("Treatment A", "Treatment B"),
    outcome_type="binary",
    save_path="node_splitting.png"
)
```

## :test_tube: Testing

```bash
# Run validation tests
cd evaluation/tests
python validation_tests.py

# Expected output:
# ============================================================
# STATISTICAL METHODS VALIDATION TESTS
# ============================================================
# PASS: test_dl_estimator
# PASS: test_i2_calculation
# PASS: test_pooled_effect
# PASS: test_hksj_calculation
# PASS: test_vif_no_collinearity
# PASS: test_reml_convergence
# PASS: test_prediction_interval_wider
#
# RESULTS: 7/7 tests passed (100.0%)
# ============================================================
# All tests passed!
```

## :file_folder: Project Structure

```
llm-meta-analysis/
├── evaluation/
│   ├── statistical_framework_v2.py     # NEW: Advanced statistical methods
│   ├── meta_regression_v2.py           # NEW: Meta-regression with Knapp-Hartung
│   ├── cumulative_analysis_v2.py       # NEW: Cumulative analysis
│   ├── integrated_meta_analysis_v2.py  # NEW: Unified interface
│   ├── glmm_meta_analysis.py           # NEW: GLMM for binary outcomes
│   ├── network_meta_analysis.py        # ENHANCED: NMA with inconsistency plots
│   │
│   ├── statistical_framework.py        # DEPRECATED: Use v2 instead
│   ├── meta_regression.py              # DEPRECATED: Use v2 instead
│   ├── cumulative_analysis.py          # DEPRECATED: Use v2 instead
│   │
│   ├── tests/
│   │   └── validation_tests.py         # NEW: Unit tests
│   │
│   ├── multivariate_meta_analysis.py   # Multivariate meta-analysis
│   ├── ipd_meta_analysis.py            # Individual participant data
│   ├── ml_quality_assessment.py        # ML quality assessment
│   ├── sensitivity_analysis.py         # Sensitivity analysis
│   ├── report_generator.py             # PRISMA reports
│   ├── clinical_decision_support.py    # GRADE recommendations
│   ├── federated_learning.py           # Privacy-preserving MA
│   └── cross_validation.py             # Cross-validation
│
├── CHANGELOG.md                        # NEW: Version history and migration guide
└── README.md                           # This file
```

## :star: CITATION

```bibtex
@inproceedings{yun2024automatically,
  title={Automatically Extracting Numerical Results from Randomized Controlled Trials with Large Language Models},
  author={Yun, Hye Sun and Pogrebitskiy, David and Marshall, Iain J and Wallace, Byron C},
  booktitle={Machine Learning for Healthcare Conference},
  year={2024},
  organization={PMLR}
}
```

## :notebook: Statistical References

This implementation follows guidance from:

1. **Hartung & Knapp (2001)** - A refined method for meta-analysis. *Biometrical Journal*, 43(2), 189-206.
2. **Viechtbauer (2007)** - Confidence intervals for the amount of heterogeneity. *Research Synthesis Methods*, 2(2), 75-79.
3. **Partlett & Riley (2017)** - Allocation of participants in clinical trials. *Statistics in Medicine*, 36(1), 14-24.
4. **Knapp & Hartung (2003)** - Improved tests in meta-regression. *Biometrical Journal*, 45(2), 199-219.
5. **Jackson et al. (2014)** - The Knapp-Hartung adjustment. *Research Synthesis Methods*, 5(4), 335-349.
6. **Doi & Thalib (2008)** - A quality effects model for meta-analysis. *Epidemiology*, 19(1), 94-100.
7. **Cochrane Handbook for Systematic Reviews of Interventions** - Version 6.4

## :link: Migration from v1 to v2

See `CHANGELOG.md` for complete migration guide. Key changes:

1. Import from `*_v2.py` modules instead of old modules
2. Use `ci_method="auto"` for automatic HKSJ/Wald selection
3. `prediction_interval` is now properly computed (no longer None)
4. `tau_squared_ci` and `i_squared_ci` are now included in results
5. Temporal data validation is explicit for cumulative analysis

## :information_source: Acknowledgments

This revision addresses feedback from the editorial board of *Research Synthesis Methods*, particularly regarding:
- Implementation of HKSJ adjustment
- Q-profile confidence intervals
- Knapp-Hartung for meta-regression
- Proper prediction intervals
- Validation against R metafor package
