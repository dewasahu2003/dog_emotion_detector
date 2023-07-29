import torch
from transformers import PreTrainedModel
from configuration import Dog_Config
from base_model import Dog


class ClassificationModelForDogEmotion(PreTrainedModel):
    config_class = Dog_Config

    def __init__(self, config: Dog_Config):
        super().__init__(config)
        self.model = Dog(
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            dropout=config.dropout,
            hidden_layer1=config.hidden_layer1,
            hidden_layer2=config.hidden_layer2,
        )

    def forward(self, tensor, labels=None):
        logits = self.model(tensor)
        if labels is not None:
            loss = torch.nn.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
