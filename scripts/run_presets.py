import pandas as pd

from src.utils import load_csv, build_preprocessor, split_data
from src.models import ModelConfig, build_model
from src.trainer import train_and_evaluate

def print_result(title, result):
    tr = result["train"]
    te = result["test"]
    print("\n" + "="*80)
    print(title)
    print(f"FIT: {result['fit_label']} | gap={result['gap']:.3f}")
    print(f"TRAIN  acc={tr['accuracy']:.4f}  prec={tr['precision']:.4f}  rec={tr['recall']:.4f}  f1={tr['f1']:.4f}")
    print(f"TEST   acc={te['accuracy']:.4f}  prec={te['precision']:.4f}  rec={te['recall']:.4f}  f1={te['f1']:.4f}")
    print("="*80)

def run_one(csv_path, target_col, scaling, one_hot, test_split, cfg):
    df = load_csv(csv_path)
    X, y, preprocessor = build_preprocessor(df, target_col, scaling, one_hot)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_split, random_state=42)
    model = build_model(cfg)
    return train_and_evaluate(model, preprocessor, X_train, X_test, y_train, y_test)

def main():
    csv_path = "data/student.csv"
    target_col = "target"

    # ✅ GARANTİ UNDERFITTING: Perceptron
    res1 = run_one(
        csv_path, target_col,
        scaling="standard", one_hot=True, test_split=0.35,
        cfg=ModelConfig(model_name="Perceptron", random_state=42)
    )
    print_result("UNDERFITTING (Perceptron | standard | test_split=0.35)", res1)

    # ✅ GARANTİ GOOD FIT: Decision Tree depth=3
    res2 = run_one(
        csv_path, target_col,
        scaling="none", one_hot=True, test_split=0.25,
        cfg=ModelConfig(model_name="Decision Tree", tree_max_depth=2, random_state=42)
    )
    print_result("GOOD FIT (Decision Tree depth=3 | none | test_split=0.25)", res2)

    # ✅ GARANTİ OVERFITTING: Decision Tree depth=20
    res3 = run_one(
        csv_path, target_col,
        scaling="none", one_hot=True, test_split=0.25,
        cfg=ModelConfig(model_name="Decision Tree", tree_max_depth=20, random_state=42)
    )
    print_result("OVERFITTING (Decision Tree depth=20 | none | test_split=0.25)", res3)

    # MLP (çalıştığını göstermek için)
    res4 = run_one(
        csv_path, target_col,
        scaling="standard", one_hot=True, test_split=0.25,
        cfg=ModelConfig(model_name="MLP", mlp_hidden_layers=2, random_state=42)
    )
    print_result("MLP (comparison | hidden_layers=2 | standard | test_split=0.25)", res4)

if __name__ == "__main__":
    main()
