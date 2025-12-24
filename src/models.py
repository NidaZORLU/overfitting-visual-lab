from dataclasses import dataclass
from typing import Literal, Optional

from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


ModelName = Literal["Perceptron", "MLP", "Decision Tree"]


@dataclass
class ModelConfig:
    model_name: ModelName
    mlp_hidden_layers: int = 2          # GUI slider (1-5)
    tree_max_depth: int = 5             # GUI slider (1-20)
    random_state: int = 42


def build_model(cfg: ModelConfig):
    if cfg.model_name == "Perceptron":
        # Basit lineer model
        return Perceptron(random_state=cfg.random_state, max_iter=2000, tol=1e-3)

    if cfg.model_name == "MLP":
        # Kritik nokta: MLP'nin gerçekten öğrenmesi için iterasyon ve öğrenme oranı.
        # hidden_layer_sizes: katman sayısına göre her katmana 32 nöron koyuyoruz.
        hidden = tuple([32] * int(cfg.mlp_hidden_layers))

        return MLPClassifier(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            max_iter=1000,
            alpha=1e-3,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=cfg.random_state,
        )


    if cfg.model_name == "Decision Tree":
        return DecisionTreeClassifier(
            max_depth=int(cfg.tree_max_depth),
            random_state=cfg.random_state
        )

    raise ValueError(f"Unknown model: {cfg.model_name}")
