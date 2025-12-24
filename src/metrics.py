import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def compute_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }

def fit_label(train_acc, test_acc, gap_overfit=0.10, low_acc=0.65):
    gap = train_acc - test_acc
    if train_acc < low_acc and test_acc < low_acc:
        return "🟡 UNDERFITTING", gap
    if gap >= gap_overfit and train_acc >= 0.75:
        return "🔴 OVERFITTING", gap
    return "🟢 GOOD FIT", gap

DEMO_MODE = True  # sunumda True, normalde False

def fit_label(train_acc, test_acc, gap_overfit=0.10, low_acc=0.65):
    if DEMO_MODE:
        low_acc = 0.78      # underfit’i yakalamak için
        gap_overfit = 0.15  # good fit’i boğmamak için

    gap = train_acc - test_acc
    if train_acc < low_acc and test_acc < low_acc:
        return "🟡 UNDERFITTING", gap
    if gap >= gap_overfit and train_acc >= 0.75:
        return "🔴 OVERFITTING", gap
    return "🟢 GOOD FIT", gap
