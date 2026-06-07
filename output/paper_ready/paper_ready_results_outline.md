# Paper-Ready Results Outline

## Working Title
Pitch Sequencing as Strategic Interaction: Modeling MLB Pitch Selection as Pitcher-Specific State Machines

## Research Question
How do MLB pitchers choose their next pitch as a function of count, previous pitch, batter handedness, and pitcher-specific arsenal?

## Main Model Findings
- Best tuned state-machine specification: `state:pitcher_count_prev_stand:arsenal_masked`, alpha=1.0, minimum state count=5.
- Best pooled predictive model: `pooled_hgb_tuned_4_arsenal_masked`, using learning_rate=0.08, max_iter=160, max_leaf_nodes=31, and l2_regularization=0.05.
- The best model reaches exact accuracy 43.3%, top-2 accuracy 69.8%, and top-3 accuracy 85.5% on the 2025 holdout.
- On 2026-to-date temporal validation, the strongest pooled model reaches exact accuracy 39.1% and top-3 accuracy 80.1%.


## Regression Strategy
Linear probability models estimate whether the next pitch is one of: FF, SI, CH, SL, CU. Predictors include count, previous pitch, batter side, and pitcher fixed effects.

## Suggested Figure Captions
**Figure 1. Pitcher-specific arsenals.** Each row shows the observed pitch menu available to a pitcher under the rule usage >= 3% or at least 30 pitches.

**Figure 2. Model comparison.** State-machine and pooled tree models are evaluated against the pitch actually thrown. Top-k accuracy is emphasized because pitch calling is a mixed-strategy decision.

**Figure 3. Regression coefficient heatmap.** OLS linear probability coefficients show how count, previous pitch, and batter side shift the probability of each pitch type.

**Figure 4. SHAP feature importance.** One-vs-rest random forest explanations show which state/context features most influence the probability of specific pitch outcomes.

**Figure 5. Global SHAP feature importance.** SHAP values are averaged across pitch-specific explanation models to identify the strongest general sequencing signals.

**Figure 6. Grouped SHAP feature importance.** One-hot features are collapsed into baseball concepts so pitcher-name indicators are interpreted as pitcher identity/arsenal rather than as standalone causal explanations.

## Transparency Note
XGBoost may require OpenMP (`libomp`) on macOS. The local run uses sklearn's histogram gradient boosting when XGBoost cannot load. Colab/Linux can run XGBoost directly if desired.
