from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
import warnings

from src.metrics import compute_metrics, fit_label


def train_and_evaluate(model, preprocessor, X_train, X_test, y_train, y_test):
    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", model),
    ])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        pipe.fit(X_train, y_train)


    y_pred_train = pipe.predict(X_train)
    train_m = compute_metrics(y_train, y_pred_train)


    y_pred_test = pipe.predict(X_test)
    test_m = compute_metrics(y_test, y_pred_test)

    label, gap = fit_label(train_m["accuracy"], test_m["accuracy"])

    return {
        "pipeline": pipe,
        "train": train_m,
        "test": test_m,
        "fit_label": label,
        "gap": float(gap)
    }
