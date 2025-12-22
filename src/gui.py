import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt

from src.utils import load_csv, build_preprocessor, split_data
from src.models import ModelConfig, build_model
from src.trainer import train_and_evaluate


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Overfitting & Underfitting Visual Lab")
        self.geometry("980x650")

        self.csv_path = tk.StringVar(value="")
        self.target_col = tk.StringVar(value="target")

        self.scaling = tk.StringVar(value="standard")   # none|standard|minmax
        self.one_hot = tk.BooleanVar(value=True)

        self.model_name = tk.StringVar(value="Decision Tree")

        self.test_split = tk.DoubleVar(value=0.25)

        self.mlp_layers = tk.IntVar(value=2)   # 1..5
        self.tree_depth = tk.IntVar(value=5)   # 1..20

        self._build_ui()

    def _build_ui(self):
        pad = 10
        top = ttk.Frame(self, padding=pad)
        top.pack(fill="x")

        ttk.Button(top, text="CSV Seç", command=self.pick_csv).pack(side="left")
        ttk.Label(top, textvariable=self.csv_path, width=60).pack(side="left", padx=8)

        mid = ttk.Frame(self, padding=pad)
        mid.pack(fill="x")

        ttk.Label(mid, text="Target Column:").grid(row=0, column=0, sticky="w")
        ttk.Entry(mid, textvariable=self.target_col, width=20).grid(row=0, column=1, sticky="w", padx=6)

        ttk.Label(mid, text="Scaling:").grid(row=0, column=2, sticky="w")
        ttk.Combobox(mid, textvariable=self.scaling, values=["none","standard","minmax"], width=12, state="readonly").grid(row=0, column=3, sticky="w", padx=6)

        ttk.Checkbutton(mid, text="One-Hot Encoding", variable=self.one_hot).grid(row=0, column=4, sticky="w", padx=10)

        ttk.Label(mid, text="Model:").grid(row=1, column=0, sticky="w", pady=(8,0))
        ttk.Combobox(mid, textvariable=self.model_name, values=["Perceptron","MLP","Decision Tree"], width=18, state="readonly").grid(row=1, column=1, sticky="w", padx=6, pady=(8,0))

        ttk.Label(mid, text="Test Split (0.05-0.95):").grid(row=1, column=2, sticky="w", pady=(8,0))
        ttk.Entry(mid, textvariable=self.test_split, width=8).grid(row=1, column=3, sticky="w", padx=6, pady=(8,0))

        sliders = ttk.Frame(self, padding=pad)
        sliders.pack(fill="x")

        ttk.Label(sliders, text="MLP Hidden Layers (1-5):").grid(row=0, column=0, sticky="w")
        ttk.Scale(sliders, from_=1, to=5, variable=self.mlp_layers, orient="horizontal").grid(row=0, column=1, sticky="ew", padx=10)
        ttk.Label(sliders, textvariable=self.mlp_layers, width=3).grid(row=0, column=2, sticky="w")

        ttk.Label(sliders, text="Tree Max Depth (1-20):").grid(row=1, column=0, sticky="w", pady=(8,0))
        ttk.Scale(sliders, from_=1, to=20, variable=self.tree_depth, orient="horizontal").grid(row=1, column=1, sticky="ew", padx=10, pady=(8,0))
        ttk.Label(sliders, textvariable=self.tree_depth, width=3).grid(row=1, column=2, sticky="w", pady=(8,0))

        sliders.columnconfigure(1, weight=1)

        action = ttk.Frame(self, padding=pad)
        action.pack(fill="x")
        ttk.Button(action, text="Train & Evaluate", command=self.run).pack(side="left")

        self.fit_label_var = tk.StringVar(value="Durum: -")
        ttk.Label(action, textvariable=self.fit_label_var, font=("Arial", 14, "bold")).pack(side="left", padx=14)

        self.metrics_text = tk.Text(self, height=14)
        self.metrics_text.pack(fill="both", expand=True, padx=pad, pady=pad)

        ttk.Button(self, text="Train vs Test Accuracy Grafiği", command=self.plot_last).pack(pady=(0,10))

        self.last_train_acc = None
        self.last_test_acc = None

    def pick_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files","*.csv")])
        if path:
            self.csv_path.set(path)

    def run(self):
        try:
            if not self.csv_path.get():
                raise ValueError("Önce CSV seçmelisin.")
            df = load_csv(self.csv_path.get())

            X, y, preprocessor = build_preprocessor(
                df=df,
                target_col=self.target_col.get().strip(),
                scaling=self.scaling.get(),
                one_hot=bool(self.one_hot.get())
            )

            X_train, X_test, y_train, y_test = split_data(X, y, test_size=float(self.test_split.get()))

            cfg = ModelConfig(
                model_name=self.model_name.get(),
                mlp_hidden_layers=int(round(self.mlp_layers.get())),
                tree_max_depth=int(round(self.tree_depth.get()))
            )
            model = build_model(cfg)

            result = train_and_evaluate(model, preprocessor, X_train, X_test, y_train, y_test)

            tr = result["train"]
            te = result["test"]

            self.last_train_acc = tr["accuracy"]
            self.last_test_acc = te["accuracy"]

            self.fit_label_var.set(f"Durum: {result['fit_label']} | gap={result['gap']:.3f}")

            self.metrics_text.delete("1.0", tk.END)
            self.metrics_text.insert(tk.END, "=== TRAIN METRICS ===\n")
            self.metrics_text.insert(tk.END, f"Accuracy: {tr['accuracy']:.4f}\nPrecision: {tr['precision']:.4f}\nRecall: {tr['recall']:.4f}\nF1: {tr['f1']:.4f}\n\n")
            self.metrics_text.insert(tk.END, "Confusion Matrix (Train):\n")
            self.metrics_text.insert(tk.END, f"{tr['confusion_matrix']}\n\n")

            self.metrics_text.insert(tk.END, "=== TEST METRICS ===\n")
            self.metrics_text.insert(tk.END, f"Accuracy: {te['accuracy']:.4f}\nPrecision: {te['precision']:.4f}\nRecall: {te['recall']:.4f}\nF1: {te['f1']:.4f}\n\n")
            self.metrics_text.insert(tk.END, "Confusion Matrix (Test):\n")
            self.metrics_text.insert(tk.END, f"{te['confusion_matrix']}\n")

        except Exception as e:
            messagebox.showerror("Hata", str(e))

    def plot_last(self):
        if self.last_train_acc is None:
            messagebox.showinfo("Bilgi", "Önce Train & Evaluate çalıştır.")
            return
        plt.figure()
        plt.bar(["Train Acc", "Test Acc"], [self.last_train_acc, self.last_test_acc])
        plt.ylim(0, 1)
        plt.title("Train vs Test Accuracy")
        plt.show()

if __name__ == "__main__":
    app = App()
    app.mainloop()
