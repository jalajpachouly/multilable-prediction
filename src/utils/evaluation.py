"""Model evaluation functions."""
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, f1_score, hamming_loss
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping
from typing import Dict, List, Callable


def cross_validation_score_multilabel(
    classifier, 
    X, 
    y: np.ndarray, 
    n_splits: int = 10
) -> Dict[str, float]:
    """
    Perform cross-validation and compute average Recall and F1-score.

    Parameters:
    - classifier: The classifier to evaluate.
    - X (sparse matrix or ndarray): Feature matrix.
    - y (ndarray): Label matrix.
    - n_splits (int): Number of cross-validation splits.

    Returns:
    - dict: Dictionary containing average Recall and F1-score.
    """
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    recall_scores = []
    f1_scores = []

    for fold, (train_index, test_index) in enumerate(mskf.split(X, y), 1):
        X_train_cv, X_test_cv = X[train_index], X[test_index]
        y_train_cv, y_test_cv = y[train_index], y[test_index]

        classifier.fit(X_train_cv, y_train_cv)
        y_pred_cv = classifier.predict(X_test_cv)

        recall = recall_score(y_test_cv, y_pred_cv, average='macro', zero_division=0)
        f1 = f1_score(y_test_cv, y_pred_cv, average='macro', zero_division=0)

        recall_scores.append(recall)
        f1_scores.append(f1)

        print(f"Fold {fold}: Recall = {recall:.4f}, F1-Score = {f1:.4f}")

    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)

    return {'Recall': avg_recall, 'F1': avg_f1}


def cross_validation_score_deep_learning(
    model_builder: Callable, 
    X: np.ndarray, 
    y: np.ndarray, 
    n_splits: int = 10, 
    epochs: int = 10, 
    batch_size: int = 32,
    patience: int = 3
) -> Dict[str, float]:
    """
    Perform cross-validation for a Deep Learning model and compute average Recall and F1-score.

    Parameters:
    - model_builder: Function to build the model.
    - X (ndarray): Feature matrix.
    - y (ndarray): Label matrix.
    - n_splits (int): Number of cross-validation splits.
    - epochs (int): Number of training epochs.
    - batch_size (int): Batch size for training.
    - patience (int): Early stopping patience.

    Returns:
    - dict: Dictionary containing average Recall and F1-score.
    """
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    recall_scores = []
    f1_scores = []

    for fold, (train_index, test_index) in enumerate(mskf.split(X, y), 1):
        X_train_cv, X_test_cv = X[train_index], X[test_index]
        y_train_cv, y_test_cv = y[train_index], y[test_index]

        model = model_builder()
        early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        model.fit(
            X_train_cv,
            y_train_cv,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test_cv, y_test_cv),
            callbacks=[early_stop],
            verbose=0
        )

        y_pred_cv_prob = model.predict(X_test_cv)
        y_pred_cv = (y_pred_cv_prob >= 0.5).astype(int)

        recall = recall_score(y_test_cv, y_pred_cv, average='macro', zero_division=0)
        f1 = f1_score(y_test_cv, y_pred_cv, average='macro', zero_division=0)

        recall_scores.append(recall)
        f1_scores.append(f1)

        print(f"Fold {fold}: {model_builder.__name__} Recall = {recall:.4f}, F1-Score = {f1:.4f}")

    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)

    return {'Recall': avg_recall, 'F1': avg_f1}


def evaluate_classifier(
    clf, 
    clf_name: str, 
    X_train, 
    y_train: np.ndarray, 
    X_test, 
    y_test: np.ndarray, 
    label_names: List[str]
) -> List[Dict]:
    """
    Train the classifier, make predictions, and evaluate performance.

    Parameters:
    - clf: The classifier to evaluate.
    - clf_name (str): Name of the classifier.
    - X_train (sparse matrix or ndarray): Training feature matrix.
    - y_train (ndarray): Training labels.
    - X_test (sparse matrix or ndarray): Testing feature matrix.
    - y_test (ndarray): Testing labels.
    - label_names (list): List of label names.

    Returns:
    - list of dicts: List containing evaluation metrics for each label.
    """
    print(f"\n===== Evaluating {clf_name} =====")

    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    hamming_loss_value = hamming_loss(y_test, predictions)

    metrics = []
    n_labels = y_test.shape[1]

    for label_idx in range(n_labels):
        y_true_label = y_test[:, label_idx]
        y_pred_label = predictions[:, label_idx]

        recall = recall_score(y_true_label, y_pred_label, zero_division=0)
        f1 = f1_score(y_true_label, y_pred_label, zero_division=0)

        metrics.append({
            'Model': clf_name,
            'Label': label_names[label_idx],
            'Recall': recall,
            'F1': f1,
            'Hamming Loss': hamming_loss_value
        })

    print(f"Hamming Loss for {clf_name}: {hamming_loss_value}")
    return metrics


def evaluate_deep_learning_model(
    model, 
    X_test, 
    y_test: np.ndarray, 
    model_name: str, 
    label_names: List[str]
) -> List[Dict]:
    """
    Evaluate the Deep Learning model on the test set.

    Parameters:
    - model: The trained Keras model.
    - X_test (sparse matrix or ndarray): Testing feature matrix.
    - y_test (ndarray): Testing labels.
    - model_name (str): Name of the model for reporting.
    - label_names (list): List of label names.

    Returns:
    - list of dicts: List containing evaluation metrics for each label.
    """
    print(f"\n===== Evaluating {model_name} Model =====")

    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()
    
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob >= 0.5).astype(int)
    hamming_loss_value = hamming_loss(y_test, y_pred)

    metrics = []
    n_labels = y_test.shape[1]

    for label_idx in range(n_labels):
        y_true_label = y_test[:, label_idx]
        y_pred_label = y_pred[:, label_idx]

        recall = recall_score(y_true_label, y_pred_label, zero_division=0)
        f1 = f1_score(y_true_label, y_pred_label, zero_division=0)

        metrics.append({
            'Model': model_name,
            'Label': label_names[label_idx],
            'Recall': recall,
            'F1': f1,
            'Hamming Loss': hamming_loss_value
        })

    print(f"Hamming Loss for {model_name} Model: {hamming_loss_value}")
    return metrics
