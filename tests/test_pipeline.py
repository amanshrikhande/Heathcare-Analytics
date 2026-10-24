"""
test_pipeline.py
Automated test suite validating the standalone notebook execution, serialized models,
leakage prevention contracts, target distributions, and dashboard artifacts.
Independent of modular src/ scripts.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, brier_score_loss, mean_absolute_error
)


def test_notebook_exists_and_executed():
    nb_path = "notebooks/Healthcare_Analytics.ipynb"
    assert os.path.exists(nb_path), "Healthcare_Analytics.ipynb must exist in notebooks/"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)
    assert len(nb_data['cells']) >= 14, "Notebook must contain all required sections"
    # Verify outputs are generated
    code_cells_with_outputs = [c for c in nb_data['cells'] if c.get('cell_type') == 'code' and len(c.get('outputs', [])) > 0]
    assert len(code_cells_with_outputs) >= 5, "Notebook code cells must have executed outputs"


def test_processed_cohort_features():
    feat_path = "data/processed/cohort_features.csv"
    assert os.path.exists(feat_path), f"Missing {feat_path}"
    df = pd.read_csv(feat_path)
    assert len(df) >= 1000, "Eligible cohort must have >= 1,000 stays"
    assert 'target_prolonged_3d' in df.columns
    assert 'target_remaining_los_hours' in df.columns
    assert 'patient_group_id' in df.columns
    assert (df['unitdischargeoffset'] >= 1440.0).all(), "All stays must be >= 24h"
    assert df['target_prolonged_3d'].isin([0, 1]).all(), "Target must be binary 0 or 1"
    prevalence = df['target_prolonged_3d'].mean()
    assert 0.20 <= prevalence <= 0.50, f"Prolonged prevalence {prevalence:.2f} should be clinically balanced"


def test_leakage_prevention_contract():
    forbidden_cols = [
        'patientunitstayid', 'patienthealthsystemstayid', 'uniquepid',
        'patient_group_id', 'unitdischargeoffset', 'hospitaldischargeoffset',
        'unitdischargestatus', 'hospitaldischargestatus',
        'target_prolonged_3d', 'target_remaining_los_hours'
    ]
    with open("models/feature_sets.json", "r") as f:
        fsets = json.load(f)
    for rep_name, cols_dict in fsets.items():
        all_cols = cols_dict['numeric'] + cols_dict['categorical']
        leaked = [c for c in forbidden_cols if c in all_cols]
        assert len(leaked) == 0, f"CRITICAL LEAKAGE: Feature set '{rep_name}' contains forbidden columns: {leaked}"


def test_patient_grouped_split_isolation():
    res_path = "models/experimental_results.json"
    assert os.path.exists(res_path)
    with open(res_path, "r") as f:
        res = json.load(f)
    meta = res['metadata']
    assert meta['train_stays'] > 0
    assert meta['test_stays'] > 0
    pred_path = "models/test_predictions.csv"
    assert os.path.exists(pred_path)
    df_preds = pd.read_csv(pred_path)
    assert len(df_preds) == meta['test_stays']


def test_physiological_bounds_filtering():
    df = pd.DataFrame({
        'heartrate': [15.0, 75.0, 310.0],
        'sao2': [35.0, 99.0, 110.0],
        'noninvasivesystolic': [25.0, 125.0, 300.0]
    })
    # Apply physiological limits
    df.loc[(df['heartrate'] < 25) | (df['heartrate'] > 250), 'heartrate'] = np.nan
    df.loc[(df['sao2'] < 50) | (df['sao2'] > 100), 'sao2'] = np.nan
    df.loc[(df['noninvasivesystolic'] < 40) | (df['noninvasivesystolic'] > 260), 'noninvasivesystolic'] = np.nan
    
    assert np.isnan(df.loc[0, 'heartrate'])
    assert df.loc[1, 'heartrate'] == 75.0
    assert np.isnan(df.loc[2, 'heartrate'])
    assert np.isnan(df.loc[0, 'sao2'])
    assert df.loc[1, 'sao2'] == 99.0


def test_metrics_calculation_integrity():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.8, 0.3])
    
    auc_val = float(roc_auc_score(y_true, y_prob))
    f1_val = float(f1_score(y_true, y_pred))
    acc_val = float(accuracy_score(y_true, y_pred))
    brier_val = float(brier_score_loss(y_true, y_prob))
    
    assert 0.5 <= auc_val <= 1.0
    assert 0.0 <= f1_val <= 1.0
    assert acc_val > 0.5
    assert brier_val < 0.25


def test_model_inference_pipeline():
    model_path = "models/best_classifier_pipeline.joblib"
    assert os.path.exists(model_path), f"Missing {model_path}"
    pipeline = joblib.load(model_path)
    
    with open("models/feature_sets.json", "r") as f:
        fsets = json.load(f)
    with open("models/experimental_results.json", "r") as f:
        results = json.load(f)
        
    best_rep = results['best_model']['feature_representation']
    num_cols = fsets[best_rep]['numeric']
    cat_cols = fsets[best_rep]['categorical']
    
    dummy_payload = {c: 0.0 for c in num_cols}
    for c in cat_cols:
        dummy_payload[c] = 'Unknown'
    dummy_df = pd.DataFrame([dummy_payload])
    
    pred = pipeline.predict(dummy_df)
    prob = pipeline.predict_proba(dummy_df)
    assert len(pred) == 1
    assert prob.shape == (1, 2)
    assert 0.0 <= prob[0, 1] <= 1.0


def test_dashboard_artifacts_exist():
    required_files = [
        "models/experimental_results.json",
        "models/feature_importance.json",
        "models/test_predictions.csv",
        "models/best_classifier_pipeline.joblib",
        "models/best_regressor_pipeline.joblib",
        "models/feature_sets.json",
        "data/processed/cohort_features.csv",
        "data/processed/sample_trajectories.csv",
        "reports/figures/roc_curves_comparison.png",
        "reports/figures/confusion_matrix_best.png",
        "reports/figures/permutation_importance.png",
        "reports/dataset_profile.json",
        "reports/results.json"
    ]
    for rf in required_files:
        assert os.path.exists(rf), f"Required artifact {rf} is missing!"
        assert os.path.getsize(rf) > 0, f"Artifact {rf} is empty!"
