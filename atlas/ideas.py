"""Task-local testable hypotheses, with no externally learned search policy."""
from __future__ import annotations


def idea_registry(spec):
    rows=[
        ('native_categories',bool(spec.categorical_features),'Native categorical splits may outperform ordinal representations','one booster fit','class labels and entity IDs excluded from features',['adult_income','beyond_healthcare_insurance_expenses']),
        ('class_weight',spec.classification,'Sqrt weighting may improve minority recall','one fit plus common controls','weights derived from fold training labels',['dry_beans','beyond_cardiotocography']),
        ('probability_adjustment',spec.classification and not spec.metric.needs_proba,'Bounded class offsets may improve primary F1','bounded prediction-only search','freeze offsets before control evaluation',['dry_beans','beyond_ecoli_proteins']),
        ('text_representation',bool(spec.text_candidates),'Character n-grams may recover spelling and morphology missed by words','one sparse linear fit','vocabulary fitted only on training fold',['sms_spam']),
        ('target_transform',not spec.classification,'Log target may stabilize positive heavy tails','one fit plus controls','nonnegative train target only; score on original scale',['california_housing','concrete_strength']),
        ('feature_interaction',len(spec.numeric_features)>=2,'A semantic ratio or product may explain residual structure','one controlled feature fit','no target/ID or validation-fitted statistics',['beyond_airfoil_self_noise','concrete_strength']),
        ('group_statistics',bool(spec.categorical_features and spec.numeric_features),'Fold-fitted group statistics may explain conditional scale','one controlled feature fit','unseen groups use training-only global fallback',['adult_income']),
        ('ensemble_candidate',True,'Complementary legal out-of-fold errors may reduce variance','measured fit costs for at most three components','primary weights frozen before common controls',['dry_beans','concrete_strength']),
        ('seed_bagging',True,'Averaging two independently seeded fits may reduce variance','approximately two measured fits','identical legal fold; no pseudo-labels',['concrete_strength','dry_beans'])]
    return [{'kind':kind,'eligible':eligible,'gain_hypothesis':gain,'cost':cost,'leakage_risk':risk,
             'test_cases':cases,'state':'candidate_requires_task_local_measurement'}
            for kind,eligible,gain,cost,risk,cases in rows]
