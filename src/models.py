from dataclasses import dataclass
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

@dataclass
class ModelConfig:
    model_name: str
    mlp_hidden_layers: int = 1          # slider: 1..5
    mlp_hidden_units: int = 32          # sabit (istersen slider yapılır)
    tree_max_depth: int = 3             # slider: 1..20
    random_state: int = 42

def build_model(cfg: ModelConfig):
    name = cfg.model_name

    if name == "Perceptron":
        return Perceptron(max_iter=2000, tol=1e-3, random_state=cfg.random_state)

    if name == "MLP":
        hidden = tuple([cfg.mlp_hidden_units] * cfg.mlp_hidden_layers)
        return MLPClassifier(
            hidden_layer_sizes=hidden,
            max_iter=2000,
            random_state=cfg.random_state,
            early_stopping=True,
            n_iter_no_change=15
        )

    if name == "Decision Tree":
        return DecisionTreeClassifier(
            max_depth=cfg.tree_max_depth,
            random_state=cfg.random_state
        )

    raise ValueError("model_name: Perceptron | MLP | Decision Tree olmalı")
