// ============================================
// Meta-Analysis WASM Module
// Fast statistical computations in WebAssembly
// ============================================

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};

// Study data structure
#[wasm_bindgen]
pub struct Study {
    name: String,
    effect: f64,
    ci_lower: f64,
    ci_upper: f64,
    variance: f64,
}

#[wasm_bindgen]
impl Study {
    #[wasm_bindgen(constructor)]
    pub fn new(name: String, effect: f64, ci_lower: f64, ci_upper: f64, variance: f64) -> Study {
        Study {
            name,
            effect,
            ci_lower,
            ci_upper,
            variance,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.name.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn effect(&self) -> f64 {
        self.effect
    }

    #[wasm_bindgen(getter)]
    pub fn ci_lower(&self) -> f64 {
        self.ci_lower
    }

    #[wasm_bindgen(getter)]
    pub fn ci_upper(&self) -> f64 {
        self.ci_upper
    }

    #[wasm_bindgen(getter)]
    pub fn variance(&self) -> f64 {
        self.variance
    }
}

// Meta-analysis result
#[wasm_bindgen]
pub struct MetaAnalysisResult {
    pooled_effect: f64,
    ci_lower: f64,
    ci_upper: f64,
    p_value: f64,
    i_squared: f64,
    tau_squared: f64,
    q_statistic: f64,
    df: f64,
}

#[wasm_bindgen]
impl MetaAnalysisResult {
    #[wasm_bindgen(getter)]
    pub fn pooled_effect(&self) -> f64 {
        self.pooled_effect
    }

    #[wasm_bindgen(getter)]
    pub fn ci_lower(&self) -> f64 {
        self.ci_lower
    }

    #[wasm_bindgen(getter)]
    pub fn ci_upper(&self) -> f64 {
        self.ci_upper
    }

    #[wasm_bindgen(getter)]
    pub fn p_value(&self) -> f64 {
        self.p_value
    }

    #[wasm_bindgen(getter)]
    pub fn i_squared(&self) -> f64 {
        self.i_squared
    }

    #[wasm_bindgen(getter)]
    pub fn tau_squared(&self) -> f64 {
        self.tau_squared
    }

    #[wasm_bindgen(getter)]
    pub fn q_statistic(&self) -> f64 {
        self.q_statistic
    }

    #[wasm_bindgen(getter)]
    pub fn df(&self) -> f64 {
        self.df
    }

    pub fn to_json(&self) -> String {
        format!(
            r#"{{"pooled_effect":{},"ci_lower":{},"ci_upper":{},"p_value":{},"i_squared":{},"tau_squared":{},"q_statistic":{},"df":{}}}"#,
            self.pooled_effect, self.ci_lower, self.ci_upper,
            self.p_value, self.i_squared, self.tau_squared,
            self.q_statistic, self.df
        )
    }
}

// Fixed effect meta-analysis (inverse variance method)
#[wasm_bindgen]
pub fn fixed_effect_ma(studies: &[Study]) -> MetaAnalysisResult {
    let n = studies.len();

    if n == 0 {
        return MetaAnalysisResult {
            pooled_effect: 0.0,
            ci_lower: 0.0,
            ci_upper: 0.0,
            p_value: 1.0,
            i_squared: 0.0,
            tau_squared: 0.0,
            q_statistic: 0.0,
            df: 0.0,
        };
    }

    // Calculate weights (inverse variance)
    let mut sum_weight = 0.0;
    let mut sum_weighted_effect = 0.0;

    for study in studies {
        let weight = 1.0 / study.variance;
        sum_weight += weight;
        sum_weighted_effect += weight * study.effect;
    }

    // Pooled effect
    let pooled_effect = sum_weighted_effect / sum_weight;

    // Standard error
    let se = (1.0 / sum_weight).sqrt();

    // 95% CI
    let ci_lower = pooled_effect - 1.96 * se;
    let ci_upper = pooled_effect + 1.96 * se;

    // P-value (z-test)
    let z = pooled_effect / se;
    let p_value = normal_cdf(-z.abs()) * 2.0;

    // Heterogeneity (Q statistic)
    let mut q_statistic = 0.0;
    for study in studies {
        let weight = 1.0 / study.variance;
        q_statistic += weight * (study.effect - pooled_effect).powi(2);
    }

    let df = (n as f64) - 1.0;
    let i_squared = if q_statistic > df {
        ((q_statistic - df) / q_statistic) * 100.0
    } else {
        0.0
    };

    MetaAnalysisResult {
        pooled_effect,
        ci_lower,
        ci_upper,
        p_value,
        i_squared,
        tau_squared: 0.0,
        q_statistic,
        df,
    }
}

// Random effects meta-analysis (DerSimonian-Laird)
#[wasm_bindgen]
pub fn random_effects_ma(studies: &[Study]) -> MetaAnalysisResult {
    let n = studies.len();

    if n == 0 {
        return MetaAnalysisResult {
            pooled_effect: 0.0,
            ci_lower: 0.0,
            ci_upper: 0.0,
            p_value: 1.0,
            i_squared: 0.0,
            tau_squared: 0.0,
            q_statistic: 0.0,
            df: 0.0,
        };
    }

    // Fixed effect weights
    let mut sum_weight = 0.0;
    let mut sum_weighted_effect = 0.0;

    for study in studies {
        let weight = 1.0 / study.variance;
        sum_weight += weight;
        sum_weighted_effect += weight * study.effect;
    }

    let pooled_fe = sum_weighted_effect / sum_weight;

    // Q statistic
    let mut q_statistic = 0.0;
    for study in studies {
        let weight = 1.0 / study.variance;
        q_statistic += weight * (study.effect - pooled_fe).powi(2);
    }

    let df = (n as f64) - 1.0;

    // Tau squared (DerSimonian-Laird)
    let tau_squared = if q_statistic > df {
        ((q_statistic - df) / (sum_weight - (sum_weight * sum_weight) / sum_weight)).max(0.0)
    } else {
        0.0
    };

    // Random effects weights
    let mut sum_weight_re = 0.0;
    let mut sum_weighted_effect_re = 0.0;

    for study in studies {
        let weight = 1.0 / (study.variance + tau_squared);
        sum_weight_re += weight;
        sum_weighted_effect_re += weight * study.effect;
    }

    let pooled_effect = sum_weighted_effect_re / sum_weight_re;
    let se = (1.0 / sum_weight_re).sqrt();

    // 95% CI
    let ci_lower = pooled_effect - 1.96 * se;
    let ci_upper = pooled_effect + 1.96 * se;

    // P-value
    let z = pooled_effect / se;
    let p_value = normal_cdf(-z.abs()) * 2.0;

    // I squared
    let i_squared = if q_statistic > df {
        ((q_statistic - df) / q_statistic) * 100.0
    } else {
        0.0
    };

    MetaAnalysisResult {
        pooled_effect,
        ci_lower,
        ci_upper,
        p_value,
        i_squared,
        tau_squared,
        q_statistic,
        df,
    }
}

// Hartung-Knapp-Sidik-Jonkman adjustment
#[wasm_bindgen]
pub fn hksj_adjustment(studies: &[Study]) -> MetaAnalysisResult {
    let n = studies.len();

    if n < 2 {
        return random_effects_ma(studies);
    }

    // First get random effects result
    let re_result = random_effects_ma(studies);

    // Calculate adjusted variance
    let mut sum_weight = 0.0;
    for study in studies {
        let weight = 1.0 / (study.variance + re_result.tau_squared);
        sum_weight += weight;
    }

    let tau_sq = re_result.tau_squared;

    let mut sum_weighted_squared_diff = 0.0;
    for study in studies {
        let weight = 1.0 / (study.variance + tau_sq);
        sum_weighted_squared_diff += weight * (study.effect - re_result.pooled_effect).powi(2);
    }

    // Adjusted variance (use t-distribution)
    let adjusted_variance = sum_weighted_squared_diff / ((n as f64) - 1.0);
    let adjusted_se = adjusted_variance.sqrt();

    // T-distribution critical value (approximate for df = n-1)
    let t_crit = t_critical((n as f64) - 1.0, 0.975);

    let ci_lower = re_result.pooled_effect - t_crit * adjusted_se;
    let ci_upper = re_result.pooled_effect + t_crit * adjusted_se;

    // P-value using t-distribution
    let t_stat = re_result.pooled_effect / adjusted_se;
    let p_value = 2.0 * (1.0 - t_cdf(t_stat.abs(), (n as f64) - 1.0));

    MetaAnalysisResult {
        pooled_effect: re_result.pooled_effect,
        ci_lower,
        ci_upper,
        p_value,
        i_squared: re_result.i_squared,
        tau_squared: re_result.tau_squared,
        q_statistic: re_result.q_statistic,
        df: (n as f64) - 1.0,
    }
}

// Helper functions
fn normal_cdf(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    const SIGN: f64 = 1.0;
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x / 2.0).exp();

    0.5 * (1.0 + sign * y)
}

fn t_critical(df: f64, alpha: f64) -> f64 {
    // Approximation for t-distribution critical value
    if df > 100.0 {
        return 1.96; // Approximate normal
    }

    // Simple approximation (for common df values)
    let lookup = [
        (1.0, 12.706), (2.0, 4.303), (3.0, 3.182), (4.0, 2.776),
        (5.0, 2.571), (6.0, 2.447), (7.0, 2.365), (8.0, 2.306),
        (9.0, 2.262), (10.0, 2.228), (20.0, 2.086), (30.0, 2.042),
        (40.0, 2.021), (50.0, 2.009), (60.0, 2.000), (100.0, 1.984)
    ];

    for (d, val) in lookup.iter() {
        if df <= *d {
            return *val;
        }
    }

    1.96
}

fn t_cdf(t: f64, df: f64) -> f64 {
    // Approximation for t-distribution CDF
    if df > 100.0 {
        return normal_cdf(t);
    }

    // Simple approximation using beta function
    let x = (t + (t * t + df).sqrt()) / (2.0 * (t * t + df).sqrt());
    incomplete_beta(x, 0.5 * df, 0.5)
}

fn incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    // Simple approximation
    if a == 0.5 && b == 0.5 {
        (2.0 / std::f64::consts::PI) * (x.asin() + (1.0 - x * x).sqrt() * x)
    } else {
        x.powf(a) // Very rough approximation
    }
}

// Forest plot data generation
#[wasm_bindgen]
pub fn generate_forest_data(studies: &[Study], pooled: &MetaAnalysisResult) -> String {
    let mut result = String::from("[");

    for (i, study) in studies.iter().enumerate() {
        if i > 0 {
            result.push_str(",");
        }
        result.push_str(&format!(
            r#"{{"name":"{}","effect":{},"ci_lower":{},"ci_upper":{}}}"#,
            study.name, study.effect, study.ci_lower, study.ci_upper
        ));
    }

    result.push_str(&format!(
        r#",{{"name":"Pooled","effect":{},"ci_lower":{},"ci_upper":{}}}"#,
        pooled.pooled_effect, pooled.ci_lower, pooled.ci_upper
    ));

    result.push_str("]");
    result
}

// Funnel plot data generation
#[wasm_bindgen]
pub fn generate_funnel_data(studies: &[Study], pooled_effect: f64) -> String {
    let mut result = String::from("[");

    for (i, study) in studies.iter().enumerate() {
        if i > 0 {
            result.push_str(",");
        }
        let se = (study.ci_upper - study.ci_lower) / (2.0 * 1.96);
        result.push_str(&format!(
            r#"{{"name":"{}","effect":{},"se":{}}}"#,
            study.name, study.effect, se
        ));
    }

    result.push_str("]");
    result
}

// Version info
#[wasm_bindgen]
pub fn version() -> String {
    "0.1.0".to_string()
}

#[wasm_bindgen]
pub fn build_info() -> String {
    r#"{"name":"meta-analysis-wasm","version":"0.1.0","features":["fixed-effect","random-effects","hksj"]}"#.to_string()
}
