from dataclasses import dataclass
from typing import Literal
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

ModelName = Literal["Perceptron", "MLP", "Decision Tree"]

@dataclass
class ModelConfig:
    model_name: ModelName
    mlp_hidden_layers: int = 2
    tree_max_depth: int = 5
    random_state: int = 42

def build_model(cfg: ModelConfig):

    if cfg.model_name == "Perceptron":
        return Perceptron(
            max_iter=2000,
            tol=1e-3,
            random_state=cfg.random_state
        )

    if cfg.model_name == "MLP":
        hidden = tuple([32] * cfg.mlp_hidden_layers)

        return MLPClassifier(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            max_iter=800,
            alpha=1e-4,
            early_stopping=False,   # ❗ kapalı
            random_state=cfg.random_state
        )

    if cfg.model_name == "Decision Tree":
        return DecisionTreeClassifier(
            max_depth=cfg.tree_max_depth,
            random_state=cfg.random_state
        )

    raise ValueError("Model tanınmadı")
