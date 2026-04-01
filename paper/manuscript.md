# LLM-Driven Meta-Analysis: Automated Extraction and Synthesis from Clinical Trials

## Overview

A Python framework couples LLM extraction with modern meta-analytic methods for automated evidence synthesis. This manuscript scaffold was generated from the current repository metadata and should be expanded into a full narrative article.

## Study Profile

Type: methods
Primary estimand: Pooled effect size
App: LLM Meta-Analysis Framework v2.0
Data: BCG vaccine dataset, 7 validation tests
Code: https://github.com/hyesunyun/llm-meta-analysis

## E156 Capsule

Can large language models automate the extraction and synthesis of numerical results from randomized controlled trials with statistical rigor? We developed a Python framework coupling LLM-based data extraction with a meta-analytic engine implementing HKSJ-adjusted confidence intervals, Q-profile heterogeneity bounds, REML estimation with DerSimonian-Laird fallback, and network meta-analysis with inconsistency diagnostics. The pipeline includes six modules covering random-effects pooling, meta-regression with Knapp-Hartung adjustment, cumulative analysis with alpha-spending boundaries, and generalized linear mixed models for binary outcome data. Validation on the BCG dataset confirmed 7 of 7 tests passed, with the pooled RR and 95% CI matching published R metafor benchmarks within three decimal places. Bootstrap variance propagation showed that extraction uncertainty contributes less than eight percent of total variance in the pooled effect sizes. The unified interface auto-selects HKSJ for small samples and Wald intervals for larger analyses following current Cochrane guidance. Extraction accuracy is limited by LLM prompt design and may degrade with non-standard reporting formats.

## Expansion Targets

1. Expand the background and rationale into a full introduction.
2. Translate the E156 capsule into detailed methods, results, and discussion sections.
3. Add figures, tables, and a submission-ready reference narrative around the existing evidence object.
