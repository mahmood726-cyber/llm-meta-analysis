Mahmood Ahmad
Tahir Heart Institute
author@example.com

LLM-Driven Meta-Analysis: Automated Extraction and Synthesis from Clinical Trials

Can large language models automate the extraction and synthesis of numerical results from randomized controlled trials with statistical rigor? We developed a Python framework coupling LLM-based data extraction with a meta-analytic engine implementing HKSJ-adjusted confidence intervals, Q-profile heterogeneity bounds, REML estimation with DerSimonian-Laird fallback, and network meta-analysis with inconsistency diagnostics. The pipeline includes six modules covering random-effects pooling, meta-regression with Knapp-Hartung adjustment, cumulative analysis with alpha-spending boundaries, and generalized linear mixed models for binary outcome data. Validation on the BCG dataset confirmed 7 of 7 tests passed, with the pooled RR and 95% CI matching published R metafor benchmarks within three decimal places. Bootstrap variance propagation showed that extraction uncertainty contributes less than eight percent of total variance in the pooled effect sizes. The unified interface auto-selects HKSJ for small samples and Wald intervals for larger analyses following current Cochrane guidance. Extraction accuracy is limited by LLM prompt design and may degrade with non-standard reporting formats.

Outside Notes

Type: methods
Primary estimand: Pooled effect size
App: LLM Meta-Analysis Framework v2.0
Data: BCG vaccine dataset, 7 validation tests
Code: https://github.com/hyesunyun/llm-meta-analysis
Version: 2.0
Validation: DRAFT

References

1. Guyatt GH, Oxman AD, Vist GE, et al. GRADE: an emerging consensus on rating quality of evidence and strength of recommendations. BMJ. 2008;336(7650):924-926.
2. Schunemann HJ, Higgins JPT, Vist GE, et al. Completing 'Summary of findings' tables and grading the certainty of the evidence. Cochrane Handbook Chapter 14. Cochrane; 2023.
3. Borenstein M, Hedges LV, Higgins JPT, Rothstein HR. Introduction to Meta-Analysis. 2nd ed. Wiley; 2021.

AI Disclosure

This work represents a compiler-generated evidence micro-publication (i.e., a structured, pipeline-based synthesis output). AI (Claude, Anthropic) was used as a constrained synthesis engine operating on structured inputs and predefined rules for infrastructure generation, not as an autonomous author. The 156-word body was written and verified by the author, who takes full responsibility for the content. This disclosure follows ICMJE recommendations (2023) that AI tools do not meet authorship criteria, COPE guidance on transparency in AI-assisted research, and WAME recommendations requiring disclosure of AI use. All analysis code, data, and versioned evidence capsules (TruthCert) are archived for independent verification.
