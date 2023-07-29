from transformers import PretrainedConfig
from typing import List


class Dog_Config(PretrainedConfig):
    model_type = "simple_image_classification"
    """output_dim: number of category in classification"""

    def __init__(
        self,
        input_dim: int = 3,
        output_dim: int = 4,
        dropout: float = 0.3,
        hidden_layer1: int = 512,
        hidden_layer2: int = 128,
        **kwargs,
    ):
        self.input_dim: int = (input_dim,)
        self.output_dim: int = (output_dim,)
        self.dropout: float = dropout
        self.hidden_layer1: int = (hidden_layer1,)
        self.hidden_layer2: int = hidden_layer2
        super().__init__(**kwargs)
