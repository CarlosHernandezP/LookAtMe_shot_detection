"""
Train Random Forest classifier for shot detection using normalized pose sequences.

Replicates training pipeline compatible with LookAtMeProtoApp's shot_predictor.py
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, balanced_accuracy_score,
    precision_score, recall_score, f1_score, accuracy_score
)
import joblib
import xgboost as xgb
from shot_detector.ball_features import get_ball_feature_names
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from shot_detector.shot_mapper import (
    map_shot_to_class, 
    DEFAULT_SHOT_MAPPING,
    extract_shot_type_from_filename,
    get_all_shot_types_from_pose_files,
    validate_mapping
)
from shot_detector.temporal_features import extract_temporal_features, get_feature_names
from shot_detector.data_augmentation import augment_dataset


# Expected feature columns (27 features)
BODY_KEYPOINT_NAMES = [
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle'
]

EXPECTED_FEATURE_COLS = []
for kp_name in BODY_KEYPOINT_NAMES:
    EXPECTED_FEATURE_COLS.extend([f'{kp_name}_x_body_rel', f'{kp_name}_y_body_rel'])
EXPECTED_FEATURE_COLS.extend(['hip_y_abs', 'hip_x_abs', 'shoulder_center_y_abs'])

# Add ball feature columns (8 features)
BALL_FEATURE_COLS = get_ball_feature_names()
EXPECTED_FEATURE_COLS.extend(BALL_FEATURE_COLS)  # Now 27 + 10 = 37 total features

SEQUENCE_LENGTH = 30  # 30 frames per sequence


def load_pose_csv(pose_csv_path: str) -> Optional[np.ndarray]:
    """
    Load pose features from CSV file.
    
    Parameters
    ----------
    pose_csv_path : str
        Path to pose CSV file
        
    Returns
    -------
    np.ndarray or None
        Array of shape (num_frames, 35) with pose + ball features, or None if invalid
        (27 pose features + 8 ball features)
    """
    try:
        df = pd.read_csv(pose_csv_path)
        
        # Check for required columns
        if 'frame_num' not in df.columns:
            return None
        
        # Check for normalized features
        missing_cols = [col for col in EXPECTED_FEATURE_COLS if col not in df.columns]
        if missing_cols:
            # Try old format (raw keypoints) - skip these
            return None
        
        # Extract features in exact order
        # Check which columns exist (ball features may be missing in old files)
        available_cols = [col for col in EXPECTED_FEATURE_COLS if col in df.columns]
        missing_cols = [col for col in EXPECTED_FEATURE_COLS if col not in df.columns]
        
        if missing_cols:
            # Some features missing - create DataFrame with NaN for missing columns
            features_df = df[available_cols].copy()
            for col in missing_cols:
                features_df[col] = np.nan
            # Reorder to match EXPECTED_FEATURE_COLS
            features_df = features_df[EXPECTED_FEATURE_COLS]
        else:
            features_df = df[EXPECTED_FEATURE_COLS].copy()
        
        features = features_df.values
        
        # Handle NaN values for pose features only (forward/backward fill)
        # Ball features: keep NaN (XGBoost can handle NaN)
        pose_features = features[:, :27]  # First 27 are pose features
        ball_features = features[:, 27:] if features.shape[1] > 27 else np.full((features.shape[0], len(BALL_FEATURE_COLS)), np.nan)
        
        # Fill NaN for pose features only
        pose_df = pd.DataFrame(pose_features)
        pose_df = pose_df.ffill().bfill().fillna(0)  # Forward fill, backward fill, then zero
        pose_features = pose_df.values
        
        # Combine: pose features (filled) + ball features (NaN preserved)
        if features.shape[1] > 27:
            features = np.hstack([pose_features, ball_features])
        else:
            # No ball features in this file - add NaN columns
            features = np.hstack([pose_features, np.full((features.shape[0], len(BALL_FEATURE_COLS)), np.nan)])
        
        # Check for too many NaN values (>50%)
        nan_ratio = np.isnan(features).sum() / features.size
        if nan_ratio > 0.5:
            return None
        
        # Ensure we have at least some frames
        if len(features) < 10:
            return None
        
        return features
        
    except Exception as e:
        print(f"Warning: Could not load {pose_csv_path}: {e}")
        return None


def validate_sequence(sequence: np.ndarray) -> bool:
    """
    Validate that a sequence has correct shape and is not mostly NaN.
    
    Parameters
    ----------
    sequence : np.ndarray
        Array of shape (T, 27)
        
    Returns
    -------
    bool
        True if valid, False otherwise
    """
    if sequence is None:
        return False
    
    if len(sequence.shape) != 2:
        return False
    
    T, D = sequence.shape
    
    if D != 37 and D != 27:
        return False
    
    # Check for too many NaN values
    nan_ratio = np.isnan(sequence).sum() / sequence.size
    if nan_ratio > 0.5:
        return False
    
    return True


def load_training_data(data_dir: str, shot_mapping: Optional[Dict] = None, 
                      use_augmentation: bool = False, 
                      augmentation_config: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all pose sequences and labels from data directory.
    
    NOTE: Augmentation should be applied AFTER train/test split, only to training data.
    This function loads raw data without augmentation.
    
    Parameters
    ----------
    data_dir : str
        Directory containing pose CSV files
    shot_mapping : dict, optional
        Shot type mapping dictionary. If None, uses DEFAULT_SHOT_MAPPING.
    use_augmentation : bool
        DEPRECATED: Augmentation should be applied after train/test split.
        This parameter is kept for backward compatibility but ignored.
    augmentation_config : dict, optional
        DEPRECATED: Augmentation should be applied after train/test split.
        This parameter is kept for backward compatibility but ignored.
        
    Returns
    -------
    tuple
        (sequences, labels) where:
        - sequences: (N, 30, 27) array
        - labels: (N,) array of class names
    """
    if shot_mapping is None:
        shot_mapping = DEFAULT_SHOT_MAPPING
    
    sequences_list = []
    labels_list = []
    
    # Find all pose CSV files
    pattern = os.path.join(data_dir, '*_pose.csv')
    pose_files = glob.glob(pattern)
    
    print(f"Found {len(pose_files)} pose CSV files")
    
    if len(pose_files) == 0:
        raise ValueError(f"No pose CSV files found in {data_dir}. Check the data directory path.")
    
    skipped = 0
    skip_reasons = {'no_shot_type': 0, 'unmapped': 0, 'load_failed': 0, 'invalid_sequence': 0}
    
    for pose_file in tqdm(pose_files, desc="Loading sequences"):
        # Extract shot type from filename
        shot_type = extract_shot_type_from_filename(pose_file)
        if shot_type is None:
            skipped += 1
            skip_reasons['no_shot_type'] += 1
            if skip_reasons['no_shot_type'] <= 3:  # Show first few
                print(f"  Warning: Could not extract shot type from {os.path.basename(pose_file)}")
            continue
        
        # Map to class
        class_name = map_shot_to_class(shot_type, shot_mapping)
        if class_name is None:
            skipped += 1
            skip_reasons['unmapped'] += 1
            if skip_reasons['unmapped'] <= 5:  # Show first few unmapped
                print(f"  Unmapped shot type: {shot_type} (from {os.path.basename(pose_file)})")
            continue
        
        # Load pose features
        features = load_pose_csv(pose_file)
        if features is None:
            skipped += 1
            skip_reasons['load_failed'] += 1
            continue
        
        # Ensure exactly 30 frames (pad or truncate)
        if len(features) < SEQUENCE_LENGTH:
            # Pad with last frame
            padding = np.tile(features[-1:], (SEQUENCE_LENGTH - len(features), 1))
            features = np.vstack([features, padding])
        elif len(features) > SEQUENCE_LENGTH:
            # Take first 30 frames
            features = features[:SEQUENCE_LENGTH]
        
        # Validate sequence
        if not validate_sequence(features):
            skipped += 1
            skip_reasons['invalid_sequence'] += 1
            continue
        
        # Only use sequences with ball data (35 features = 27 pose + 8 ball)
        # Check if we have ball features and they're not all NaN
        if features.shape[1] < 37:
            skipped += 1
            skip_reasons['no_ball_data'] = skip_reasons.get('no_ball_data', 0) + 1
            continue
        
        # Check if ball features have any non-NaN values in ANY frame
        ball_features = features[:, 27:37]
        if np.all(np.isnan(ball_features)):
            skipped += 1
            skip_reasons['no_ball_data'] = skip_reasons.get('no_ball_data', 0) + 1
            continue
        
        sequences_list.append(features)
        labels_list.append(class_name)
    
    if skipped > 0:
        print(f"\nSkipped {skipped} files:")
        print(f"  - No shot type extracted: {skip_reasons.get('no_shot_type', 0)}")
        print(f"  - No ball data: {skip_reasons.get('no_ball_data', 0)}")
        print(f"  - Unmapped shot type: {skip_reasons['unmapped']}")
        print(f"  - Load failed: {skip_reasons['load_failed']}")
        print(f"  - Invalid sequence: {skip_reasons['invalid_sequence']}")
    
    if len(sequences_list) == 0:
        raise ValueError("No valid sequences found! Check data directory and shot mapping.")
    
    sequences = np.array(sequences_list)
    labels = np.array(labels_list)
    
    print(f"Loaded {len(sequences)} sequences")
    print(f"Class distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  {cls}: {cnt}")
    
    # NOTE: Data augmentation should be applied AFTER train/test split, only to training data
    # This is handled in train_random_forest() function
    
    return sequences, labels


def train_xgboost_classifier(
    data_dir: str,
    output_dir: str = "model_weights",
    shot_mapping: Optional[Dict] = None,
    use_augmentation: bool = True,
    augmentation_config: Optional[Dict] = None,
    n_estimators: int = 100,
    max_depth: Optional[int] = 6,
    random_state: int = 42,
    cv_folds: int = 5,
    scale_pos_weight: Optional[float] = None,  # For class imbalance (auto-calculated if None)
    test_size: float = 0.2,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8
) -> Tuple[xgb.XGBClassifier, LabelEncoder]:
    """
    Train XGBoost classifier for shot detection.
    XGBoost can handle NaN values natively, so missing ball data is preserved as NaN.
    
    Parameters
    ----------
    data_dir : str
        Directory containing pose CSV files
    output_dir : str
        Directory to save model and label encoder
    shot_mapping : dict, optional
        Shot type mapping dictionary
    use_augmentation : bool
        Whether to apply data augmentation
    augmentation_config : dict, optional
        Augmentation configuration
    n_estimators : int
        Number of boosting rounds (default: 100)
    max_depth : int, optional
        Maximum depth of trees (default: 6)
    random_state : int
        Random seed for reproducibility
    cv_folds : int
        Number of cross-validation folds (default: 5)
    scale_pos_weight : float, optional
        Controls the balance of positive and negative weights. 
        If None, will be calculated automatically from class distribution (default: None)
    test_size : float
        Proportion of dataset to include in the test split (default: 0.2 = 20%)
    learning_rate : float
        Learning rate (default: 0.1)
    subsample : float
        Subsample ratio of training instances (default: 0.8)
    colsample_bytree : float
        Subsample ratio of columns when constructing each tree (default: 0.8)
        
    Returns
    -------
    tuple
        (trained_model, label_encoder)
    """
    # Load raw data (no augmentation yet)
    sequences, labels = load_training_data(
        data_dir, 
        shot_mapping=shot_mapping,
        use_augmentation=False,  # Augmentation applied after split
        augmentation_config=None
    )
    
    # Encode labels first (needed for stratified split)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    
    print(f"\nClasses: {label_encoder.classes_}")
    
    # Split into train/test BEFORE augmentation (split on sequences, not features)
    # Using stratified split to maintain class distribution in both sets
    print(f"\nSplitting data into train/test ({1-test_size:.0%}/{test_size:.0%}) with stratified sampling...")
    train_indices, test_indices = train_test_split(
        np.arange(len(sequences)),
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded  # Stratify on labels to maintain class distribution
    )
    
    train_sequences = sequences[train_indices]
    train_labels = labels[train_indices]
    test_sequences = sequences[test_indices]
    test_labels = labels[test_indices]
    
    print(f"Training set: {len(train_sequences)} sequences")
    print(f"Test set: {len(test_sequences)} sequences")
    print("✓ Stratified split ensures proportional class distribution in both sets")
    
    # Apply data augmentation ONLY to training data
    if use_augmentation:
        print("\nApplying data augmentation to TRAINING data only...")
        # Apply augmentation to training sequences
        train_sequences_aug, train_labels_aug = augment_dataset(
            train_sequences, train_labels, augmentation_config
        )
        
        # Extract features from augmented training sequences
        X_train = extract_temporal_features(train_sequences_aug)
        y_train = label_encoder.transform(train_labels_aug)
        
        print(f"After augmentation: {len(X_train)} training samples")
    else:
        # Extract features from original training sequences (no augmentation)
        X_train = extract_temporal_features(train_sequences)
        y_train = label_encoder.transform(train_labels)
        print("\nNo data augmentation applied")
    
    # Extract features from test sequences (NO augmentation)
    X_test = extract_temporal_features(test_sequences)
    y_test = label_encoder.transform(test_labels)
    
    # Show class distribution
    print(f"\nTraining set class distribution:")
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    for cls_idx, cls_name in enumerate(label_encoder.classes_):
        count = counts_train[unique_train == cls_idx][0] if len(counts_train[unique_train == cls_idx]) > 0 else 0
        print(f"  {cls_name}: {count} samples")
    
    print(f"\nTest set class distribution:")
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    for cls_idx, cls_name in enumerate(label_encoder.classes_):
        count = counts_test[unique_test == cls_idx][0] if len(counts_test[unique_test == cls_idx]) > 0 else 0
        print(f"  {cls_name}: {count} samples")
    
    # Calculate scale_pos_weight for class imbalance (if not provided)
    if scale_pos_weight is None:
        # Calculate from training data class distribution
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        if len(unique_classes) == 2:
            # Binary classification: scale_pos_weight = negative_count / positive_count
            negative_count = class_counts[0]
            positive_count = class_counts[1]
            scale_pos_weight = negative_count / positive_count if positive_count > 0 else 1.0
            print(f"\nCalculated scale_pos_weight: {scale_pos_weight:.4f} (for class imbalance)")
        else:
            # Multi-class: use sum of all other classes / current class (average)
            total_samples = len(y_train)
            scale_pos_weight = 1.0  # Default for multi-class
            print(f"\nMulti-class classification, using default scale_pos_weight: {scale_pos_weight}")
    else:
        print(f"\nUsing provided scale_pos_weight: {scale_pos_weight:.4f}")
    
    # Train XGBoost on training data
    print("\nTraining XGBoost classifier on training set...")
    print(f"Note: XGBoost can handle NaN values natively (missing ball data preserved as NaN)")
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        tree_method='hist',  # Faster training
        enable_categorical=False,
        n_jobs=-1,
        verbosity=1
    )
    
    # XGBoost can handle NaN natively - no need to impute
    xgb_model.fit(X_train, y_train)
    
    # Cross-validation on TRAINING data only (for model selection/hyperparameter tuning)
    print("\nPerforming cross-validation on TRAINING data...")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring='accuracy')
    cv_balanced_scores = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring='balanced_accuracy')
    
    print(f"\nCross-validation results on training data ({cv_folds}-fold):")
    print(f"  Mean accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"  Mean balanced accuracy: {cv_balanced_scores.mean():.4f} (+/- {cv_balanced_scores.std() * 2:.4f})")
    print(f"  Per-fold accuracy: {cv_scores}")
    print(f"  Per-fold balanced accuracy: {cv_balanced_scores}")
    
    # Evaluate on TEST set (unseen data)
    print("\n" + "="*80)
    print("EVALUATING ON TEST SET (unseen data)")
    print("="*80)
    y_test_pred = xgb_model.predict(X_test)
    y_test_pred_proba = xgb_model.predict_proba(X_test)
    
    # Also evaluate on training set for comparison
    y_train_pred = xgb_model.predict(X_train)
    y_train_pred_proba = xgb_model.predict_proba(X_train)
    
    # Calculate comprehensive metrics on TEST set
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_balanced_acc = balanced_accuracy_score(y_test, y_test_pred)
    
    # Per-class metrics on TEST set
    test_precision_per_class = precision_score(y_test, y_test_pred, average=None, zero_division=0)
    test_recall_per_class = recall_score(y_test, y_test_pred, average=None, zero_division=0)
    test_f1_per_class = f1_score(y_test, y_test_pred, average=None, zero_division=0)
    
    # Macro and micro averages on TEST set
    test_precision_macro = precision_score(y_test, y_test_pred, average='macro', zero_division=0)
    test_recall_macro = recall_score(y_test, y_test_pred, average='macro', zero_division=0)
    test_f1_macro = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
    
    test_precision_micro = precision_score(y_test, y_test_pred, average='micro', zero_division=0)
    test_recall_micro = recall_score(y_test, y_test_pred, average='micro', zero_division=0)
    test_f1_micro = f1_score(y_test, y_test_pred, average='micro', zero_division=0)
    
    # Confusion matrix on TEST set
    cm_test = confusion_matrix(y_test, y_test_pred, labels=range(len(label_encoder.classes_)))
    
    # Also calculate metrics on TRAINING set for comparison
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_balanced_acc = balanced_accuracy_score(y_train, y_train_pred)
    train_precision_macro = precision_score(y_train, y_train_pred, average='macro', zero_division=0)
    train_recall_macro = recall_score(y_train, y_train_pred, average='macro', zero_division=0)
    train_f1_macro = f1_score(y_train, y_train_pred, average='macro', zero_division=0)
    
    # Use TEST metrics as primary metrics
    accuracy = test_accuracy
    balanced_acc = test_balanced_acc
    precision_per_class = test_precision_per_class
    recall_per_class = test_recall_per_class
    f1_per_class = test_f1_per_class
    precision_macro = test_precision_macro
    recall_macro = test_recall_macro
    f1_macro = test_f1_macro
    precision_micro = test_precision_micro
    recall_micro = test_recall_micro
    f1_micro = test_f1_micro
    cm = cm_test
    y_pred = y_test_pred
    y_pred_proba = y_test_pred_proba
    
    # Print to console
    print("\n" + "="*80)
    print("TEST SET RESULTS (unseen data)")
    print("="*80)
    
    print(f"\nOverall Metrics (TEST SET):")
    print(f"  Accuracy: {test_accuracy:.4f}")
    print(f"  Balanced Accuracy: {test_balanced_acc:.4f}")
    print(f"  Precision (macro): {test_precision_macro:.4f}")
    print(f"  Recall (macro): {test_recall_macro:.4f}")
    print(f"  F1-score (macro): {test_f1_macro:.4f}")
    
    print(f"\nTraining Set Metrics (for comparison):")
    print(f"  Accuracy: {train_accuracy:.4f}")
    print(f"  Balanced Accuracy: {train_balanced_acc:.4f}")
    print(f"  Precision (macro): {train_precision_macro:.4f}")
    print(f"  Recall (macro): {train_recall_macro:.4f}")
    print(f"  F1-score (macro): {train_f1_macro:.4f}")
    
    print(f"\nPer-Class Metrics (TEST SET):")
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 65)
    for i, cls in enumerate(label_encoder.classes_):
        support = np.sum(y_test == i)  # Use y_test for test set support
        print(f"{cls:<15} {precision_per_class[i]:<12.4f} {recall_per_class[i]:<12.4f} "
              f"{f1_per_class[i]:<12.4f} {support:<10}")
    
    print("\nConfusion Matrix (TEST SET):")
    print(" " * 15, end="")
    for cls in label_encoder.classes_:
        print(f"{cls:>12}", end="")
    print()
    for i, cls in enumerate(label_encoder.classes_):
        print(f"{cls:>15}", end="")
        for j in range(len(label_encoder.classes_)):
            print(f"{cm[i, j]:>12}", end="")
        print()
    
    # Classification report (on TEST set)
    print("\nDetailed Classification Report (TEST SET):")
    print(classification_report(y_test, y_test_pred, target_names=label_encoder.classes_))
    
    # Feature importance
    feature_importance = xgb_model.feature_importances_
    feature_names = get_feature_names(27)
    top_indices = np.argsort(feature_importance)[-20:][::-1]
    print("\nTop 20 Most Important Features:")
    for idx in top_indices:
        feat_name = feature_names[idx] if idx < len(feature_names) else f'feature_{idx}'
        print(f"  Feature {idx} ({feat_name}): {feature_importance[idx]:.6f}")
    
    # Misclassification analysis (on TEST set)
    print("\n" + "="*80)
    print("MISCLASSIFICATION ANALYSIS (TEST SET)")
    print("="*80)
    misclassifications = analyze_misclassifications(y_test, y_test_pred, y_test_pred_proba, label_encoder.classes_)
    print_misclassification_summary(misclassifications, label_encoder.classes_)
    
    # Save model and label encoder
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_path = output_path / "xgboost_model_cv.pkl"
    encoder_path = output_path / "label_encoder_rf_cv.pkl"
    
    print(f"\nSaving model to {model_path}")
    joblib.dump(xgb_model, model_path)
    
    print(f"Saving label encoder to {encoder_path}")
    joblib.dump(label_encoder, encoder_path)
    
    # Save comprehensive training report
    results_dir = output_path / "training_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_training_report(
        results_dir, timestamp, accuracy, balanced_acc,
        precision_per_class, recall_per_class, f1_per_class,
        precision_macro, recall_macro, f1_macro,
        precision_micro, recall_micro, f1_micro,
        cm, label_encoder.classes_, cv_scores, cv_balanced_scores,
        feature_importance, misclassifications,
        len(X_train), len(X_test), use_augmentation
    )
    
    print(f"\nTraining report saved to {results_dir}/")
    print("\nTraining complete!")
    
    return xgb_model, label_encoder


def analyze_misclassifications(y_true: np.ndarray, y_pred: np.ndarray, 
                               y_pred_proba: np.ndarray, class_names: np.ndarray) -> Dict:
    """
    Analyze misclassifications to identify patterns.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_pred_proba : np.ndarray
        Prediction probabilities
    class_names : np.ndarray
        Class names
        
    Returns
    -------
    dict
        Dictionary with misclassification analysis
    """
    misclassifications = {
        'confusion_pairs': {},  # (true_class, pred_class) -> count
        'per_class_errors': {cls: {'total': 0, 'confused_as': {}} for cls in class_names},
        'high_confidence_errors': [],  # Errors with high confidence
        'low_confidence_correct': []  # Correct predictions with low confidence
    }
    
    # Find misclassified samples
    misclassified_mask = y_true != y_pred
    
    for i in range(len(y_true)):
        true_class = class_names[y_true[i]]
        pred_class = class_names[y_pred[i]]
        true_class_idx = y_true[i]
        pred_class_idx = y_pred[i]
        
        if true_class != pred_class:
            # Confusion pair
            pair = (true_class, pred_class)
            misclassifications['confusion_pairs'][pair] = misclassifications['confusion_pairs'].get(pair, 0) + 1
            
            # Per-class errors
            misclassifications['per_class_errors'][true_class]['total'] += 1
            if pred_class not in misclassifications['per_class_errors'][true_class]['confused_as']:
                misclassifications['per_class_errors'][true_class]['confused_as'][pred_class] = 0
            misclassifications['per_class_errors'][true_class]['confused_as'][pred_class] += 1
            
            # High confidence errors
            confidence = y_pred_proba[i][pred_class_idx]
            if confidence > 0.7:
                misclassifications['high_confidence_errors'].append({
                    'true_class': true_class,
                    'pred_class': pred_class,
                    'confidence': float(confidence),
                    'sample_idx': int(i)
                })
        else:
            # Low confidence correct predictions
            confidence = y_pred_proba[i][pred_class_idx]
            if confidence < 0.5:
                misclassifications['low_confidence_correct'].append({
                    'class': true_class,
                    'confidence': float(confidence),
                    'sample_idx': int(i)
                })
    
    return misclassifications


def print_misclassification_summary(misclassifications: Dict, class_names: np.ndarray):
    """
    Print misclassification analysis summary.
    
    Parameters
    ----------
    misclassifications : dict
        Output from analyze_misclassifications()
    class_names : np.ndarray
        Class names
    """
    print(f"\nMost Common Confusion Pairs:")
    sorted_pairs = sorted(misclassifications['confusion_pairs'].items(), 
                         key=lambda x: x[1], reverse=True)
    for (true_cls, pred_cls), count in sorted_pairs[:10]:
        print(f"  {true_cls} -> {pred_cls}: {count} times")
    
    print(f"\nPer-Class Error Analysis:")
    for cls in class_names:
        errors = misclassifications['per_class_errors'][cls]
        if errors['total'] > 0:
            print(f"  {cls}: {errors['total']} errors")
            sorted_confusions = sorted(errors['confused_as'].items(), 
                                     key=lambda x: x[1], reverse=True)
            for confused_as, count in sorted_confusions:
                print(f"    - Confused as {confused_as}: {count} times")
    
    if misclassifications['high_confidence_errors']:
        print(f"\nHigh Confidence Errors (>0.7): {len(misclassifications['high_confidence_errors'])}")
        for err in misclassifications['high_confidence_errors'][:10]:
            print(f"  {err['true_class']} -> {err['pred_class']} (confidence: {err['confidence']:.3f})")
    
    if misclassifications['low_confidence_correct']:
        print(f"\nLow Confidence Correct Predictions (<0.5): {len(misclassifications['low_confidence_correct'])}")
        for corr in misclassifications['low_confidence_correct'][:10]:
            print(f"  {corr['class']} (confidence: {corr['confidence']:.3f})")


def plot_confusion_matrix(cm: np.ndarray, class_names: np.ndarray, save_path: Path):
    """
    Plot confusion matrix as a heatmap and save to file.
    
    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix
    class_names : np.ndarray
        Class names
    save_path : Path
        Path to save the image
    """
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix to percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
    
    # Create heatmap
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Frequency'},
        square=True,
        linewidths=0.5
    )
    
    plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create a version with raw counts
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'},
        square=True,
        linewidths=0.5
    )
    
    plt.title('Confusion Matrix (Raw Counts)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save raw counts version with different name
    raw_path = save_path.parent / f"{save_path.stem}_raw{save_path.suffix}"
    plt.savefig(raw_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_top_feature_importance(feature_importance: np.ndarray, feature_names: List[str], 
                                 save_path: Path, top_n: int = 10):
    """
    Plot top N most important features as a horizontal bar chart.
    
    Parameters
    ----------
    feature_importance : np.ndarray
        Feature importance array
    feature_names : List[str]
        List of feature names
    save_path : Path
        Path to save the image
    top_n : int
        Number of top features to display (default: 10)
    """
    # Get top N features
    top_indices = np.argsort(feature_importance)[-top_n:][::-1]
    top_importances = feature_importance[top_indices]
    top_names = [feature_names[i] if i < len(feature_names) else f'feature_{i}' 
                 for i in top_indices]
    
    # Create horizontal bar chart
    plt.figure(figsize=(12, max(6, top_n * 0.6)))
    
    y_pos = np.arange(len(top_names))
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_names)))
    
    bars = plt.barh(y_pos, top_importances, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for i, (bar, imp) in enumerate(zip(bars, top_importances)):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, 
                f'{imp:.4f}',
                ha='left', va='center', fontweight='bold', fontsize=9)
    
    plt.yticks(y_pos, top_names)
    plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
    plt.title(f'Top {top_n} Most Important Features', fontsize=16, fontweight='bold', pad=20)
    plt.gca().invert_yaxis()  # Highest importance at top
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_training_report(results_dir: Path, timestamp: str, accuracy: float, balanced_acc: float,
                        precision_per_class: np.ndarray, recall_per_class: np.ndarray, 
                        f1_per_class: np.ndarray, precision_macro: float, recall_macro: float,
                        f1_macro: float, precision_micro: float, recall_micro: float,
                        f1_micro: float, cm: np.ndarray, class_names: np.ndarray,
                        cv_scores: np.ndarray, cv_balanced_scores: np.ndarray,
                        feature_importance: np.ndarray, misclassifications: Dict, 
                        num_train_samples: int, num_test_samples: int, use_augmentation: bool):
    """
    Save comprehensive training report to files.
    
    Parameters
    ----------
    results_dir : Path
        Directory to save reports
    timestamp : str
        Timestamp for file naming
    All other parameters: training metrics and results
    """
    # Save JSON report with all metrics
    report = {
        'timestamp': timestamp,
        'model_info': {
            'num_train_samples': int(num_train_samples),
            'num_test_samples': int(num_test_samples),
            'total_samples': int(num_train_samples + num_test_samples),
            'use_augmentation': use_augmentation,
            'num_classes': len(class_names),
            'classes': class_names.tolist()
        },
        'overall_metrics': {
            'accuracy': float(accuracy),
            'balanced_accuracy': float(balanced_acc),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'precision_micro': float(precision_micro),
            'recall_micro': float(recall_micro),
            'f1_micro': float(f1_micro)
        },
        'per_class_metrics': {
            cls: {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1_score': float(f1_per_class[i]),
                'support': int(np.sum(cm[i, :]))
            }
            for i, cls in enumerate(class_names)
        },
        'cross_validation': {
            'accuracy_mean': float(cv_scores.mean()),
            'accuracy_std': float(cv_scores.std()),
            'accuracy_scores': cv_scores.tolist(),
            'balanced_accuracy_mean': float(cv_balanced_scores.mean()),
            'balanced_accuracy_std': float(cv_balanced_scores.std()),
            'balanced_accuracy_scores': cv_balanced_scores.tolist()
        },
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_labels': class_names.tolist(),
        'top_features': {
            int(idx): {
                'importance': float(feature_importance[idx]),
                'name': get_feature_names(27)[idx] if idx < len(get_feature_names(27)) else f'feature_{idx}'
            }
            for idx in np.argsort(feature_importance)[-20:][::-1]
        },
        'misclassifications': {
            'confusion_pairs': {
                f"{true_cls}->{pred_cls}": int(count)
                for (true_cls, pred_cls), count in misclassifications['confusion_pairs'].items()
            },
            'per_class_errors': {
                cls: {
                    'total': int(errors['total']),
                    'confused_as': {k: int(v) for k, v in errors['confused_as'].items()}
                }
                for cls, errors in misclassifications['per_class_errors'].items()
            },
            'high_confidence_errors_count': len(misclassifications['high_confidence_errors']),
            'low_confidence_correct_count': len(misclassifications['low_confidence_correct'])
        }
    }
    
    json_path = results_dir / f"training_report_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save text report
    txt_path = results_dir / f"training_report_{timestamp}.txt"
    with open(txt_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SHOT DETECTION MODEL TRAINING REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Training samples: {num_train_samples}\n")
        f.write(f"Test samples: {num_test_samples}\n")
        f.write(f"Total samples: {num_train_samples + num_test_samples}\n")
        f.write(f"Data augmentation: {use_augmentation} (applied to training data only)\n")
        f.write(f"Number of classes: {len(class_names)}\n")
        f.write(f"Classes: {', '.join(class_names)}\n\n")
        f.write("NOTE: All metrics below are computed on TEST SET (unseen data)\n\n")
        
        f.write("="*80 + "\n")
        f.write("OVERALL METRICS\n")
        f.write("="*80 + "\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Balanced Accuracy: {balanced_acc:.4f}\n")
        f.write(f"Precision (macro): {precision_macro:.4f}\n")
        f.write(f"Recall (macro): {recall_macro:.4f}\n")
        f.write(f"F1-Score (macro): {f1_macro:.4f}\n")
        f.write(f"Precision (micro): {precision_micro:.4f}\n")
        f.write(f"Recall (micro): {recall_micro:.4f}\n")
        f.write(f"F1-Score (micro): {f1_micro:.4f}\n\n")
        
        f.write("="*80 + "\n")
        f.write("CROSS-VALIDATION RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Mean accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n")
        f.write(f"Mean balanced accuracy: {cv_balanced_scores.mean():.4f} (+/- {cv_balanced_scores.std() * 2:.4f})\n")
        f.write(f"Per-fold accuracy: {cv_scores}\n")
        f.write(f"Per-fold balanced accuracy: {cv_balanced_scores}\n\n")
        
        f.write("="*80 + "\n")
        f.write("PER-CLASS METRICS\n")
        f.write("="*80 + "\n")
        f.write(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
        f.write("-" * 65 + "\n")
        for i, cls in enumerate(class_names):
            support = np.sum(cm[i, :])
            f.write(f"{cls:<15} {precision_per_class[i]:<12.4f} {recall_per_class[i]:<12.4f} "
                   f"{f1_per_class[i]:<12.4f} {support:<10}\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("CONFUSION MATRIX\n")
        f.write("="*80 + "\n")
        f.write(" " * 15)
        for cls in class_names:
            f.write(f"{cls:>12}")
        f.write("\n")
        for i, cls in enumerate(class_names):
            f.write(f"{cls:>15}")
            for j in range(len(class_names)):
                f.write(f"{cm[i, j]:>12}")
            f.write("\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("MISCLASSIFICATION ANALYSIS\n")
        f.write("="*80 + "\n")
        f.write("\nMost Common Confusion Pairs:\n")
        sorted_pairs = sorted(misclassifications['confusion_pairs'].items(), 
                             key=lambda x: x[1], reverse=True)
        for (true_cls, pred_cls), count in sorted_pairs:
            f.write(f"  {true_cls} -> {pred_cls}: {count} times\n")
        
        f.write("\nPer-Class Error Analysis:\n")
        for cls in class_names:
            errors = misclassifications['per_class_errors'][cls]
            if errors['total'] > 0:
                f.write(f"  {cls}: {errors['total']} errors\n")
                sorted_confusions = sorted(errors['confused_as'].items(), 
                                         key=lambda x: x[1], reverse=True)
                for confused_as, count in sorted_confusions:
                    f.write(f"    - Confused as {confused_as}: {count} times\n")
        
        f.write(f"\nHigh Confidence Errors (>0.7): {len(misclassifications['high_confidence_errors'])}\n")
        for err in misclassifications['high_confidence_errors'][:20]:
            f.write(f"  {err['true_class']} -> {err['pred_class']} "
                   f"(confidence: {err['confidence']:.3f}, sample: {err['sample_idx']})\n")
        
        f.write(f"\nLow Confidence Correct Predictions (<0.5): "
               f"{len(misclassifications['low_confidence_correct'])}\n")
        for corr in misclassifications['low_confidence_correct'][:20]:
            f.write(f"  {corr['class']} (confidence: {corr['confidence']:.3f}, "
                   f"sample: {corr['sample_idx']})\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("TOP 20 MOST IMPORTANT FEATURES\n")
        f.write("="*80 + "\n")
        feature_names = get_feature_names(27)
        top_indices = np.argsort(feature_importance)[-20:][::-1]
        for idx in top_indices:
            feat_name = feature_names[idx] if idx < len(feature_names) else f'feature_{idx}'
            f.write(f"  Feature {idx} ({feat_name}): {feature_importance[idx]:.6f}\n")
        
        # Save full feature importance with names
        f.write("\n" + "="*80 + "\n")
        f.write("ALL FEATURES BY IMPORTANCE (sorted)\n")
        f.write("="*80 + "\n")
        sorted_indices = np.argsort(feature_importance)[::-1]
        f.write(f"{'Rank':<6} {'Index':<8} {'Importance':<12} {'Feature Name'}\n")
        f.write("-" * 80 + "\n")
        for rank, idx in enumerate(sorted_indices, 1):
            feat_name = feature_names[idx] if idx < len(feature_names) else f'feature_{idx}'
            f.write(f"{rank:<6} {idx:<8} {feature_importance[idx]:<12.6f} {feat_name}\n")
    
    # Save confusion matrix as CSV
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_path = results_dir / f"confusion_matrix_{timestamp}.csv"
    cm_df.to_csv(cm_path)
    
    # Save feature importance with names as CSV
    feature_names = get_feature_names(27)
    feature_importance_df = pd.DataFrame({
        'feature_index': range(len(feature_importance)),
        'feature_name': [feature_names[i] if i < len(feature_names) else f'feature_{i}' 
                        for i in range(len(feature_importance))],
        'importance': feature_importance
    })
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
    fi_path = results_dir / f"feature_importance_{timestamp}.csv"
    feature_importance_df.to_csv(fi_path, index=False)
    
    # Save visualizations
    cm_img_path = results_dir / f"confusion_matrix_{timestamp}.png"
    fi_img_path = results_dir / f"feature_importance_top10_{timestamp}.png"
    
    plot_confusion_matrix(cm, class_names, cm_img_path)
    plot_top_feature_importance(feature_importance, feature_names, fi_img_path, top_n=10)
    
    print(f"  - JSON report: {json_path.name}")
    print(f"  - Text report: {txt_path.name}")
    print(f"  - Confusion matrix CSV: {cm_path.name}")
    print(f"  - Confusion matrix PNG: {cm_img_path.name}")
    print(f"  - Feature importance CSV: {fi_path.name}")
    print(f"  - Feature importance PNG (top 10): {fi_img_path.name}")


def discover_shot_types(data_dir: str):
    """
    Discover and print all unique shot types from pose CSV files.
    
    Parameters
    ----------
    data_dir : str
        Directory containing pose CSV files
    """
    shot_types = get_all_shot_types_from_pose_files(data_dir)
    
    print(f"\nFound {len(shot_types)} unique shot types:")
    for shot_type in sorted(shot_types):
        mapped = map_shot_to_class(shot_type, DEFAULT_SHOT_MAPPING)
        status = f" -> {mapped}" if mapped else " (UNMAPPED)"
        print(f"  {shot_type}{status}")
    
    # Validate mapping
    validation = validate_mapping(DEFAULT_SHOT_MAPPING, shot_types)
    if validation['unmapped']:
        print(f"\n⚠️  Warning: {len(validation['unmapped'])} unmapped shot types found!")
        print("   Consider updating DEFAULT_SHOT_MAPPING in shot_mapper.py")


def main():
    parser = argparse.ArgumentParser(description="Train Random Forest classifier for shot detection")
    parser.add_argument(
        '--data-dir',
        type=str,
        default='shot_detector/data',
        help='Directory containing pose CSV files (default: shot_detector/data)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='model_weights',
        help='Directory to save model and label encoder (default: model_weights)'
    )
    parser.add_argument(
        '--discover-shots',
        action='store_true',
        help='Discover and print all unique shot types from pose CSV files'
    )
    parser.add_argument(
        '--no-augmentation',
        action='store_true',
        help='Disable data augmentation'
    )
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=100,
        help='Number of boosting rounds in XGBoost (default: 100)'
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=6,
        help='Maximum depth of trees (default: 6)'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )
    parser.add_argument(
        '--scale-pos-weight',
        type=float,
        default=None,
        help='Scale positive weight for class imbalance (auto-calculated if None, default: None)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of dataset to use for testing (default: 0.2 = 20%%)'
    )
    
    args = parser.parse_args()
    
    if args.discover_shots:
        discover_shot_types(args.data_dir)
        return
    
    # Default augmentation config
    augmentation_config = {
        'mirror': True,
        'temporal_warp': True,
        'gaussian_noise': True,
        'frame_dropout': False,
        'reverse': False
    }
    
    train_xgboost_classifier(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        use_augmentation=not args.no_augmentation,
        augmentation_config=augmentation_config if not args.no_augmentation else None,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        cv_folds=args.cv_folds,
        scale_pos_weight=args.scale_pos_weight,
        test_size=args.test_size
    )


if __name__ == "__main__":
    main()
