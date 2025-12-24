import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def compute_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

def fit_label(train_acc: float, test_acc: float):

    train_acc = float(train_acc)
    test_acc = float(test_acc)
    gap = abs(train_acc - test_acc)


    if train_acc < 0.80 and test_acc < 0.66:
        return "🟡 UNDERFITTING", gap


    if train_acc >= 0.90 and gap >= 0.20:
        return "🔴 OVERFITTING", gap

    return "🟢 GOOD FIT", gap
