from typing import Dict, Any
from abc import ABC, abstractmethod
import torch

from gpt_from_scratch.data.utils import get_batch

class BaseTrainer(ABC):
    def __init__(self, vocab: list, model_params: Dict[str, Any]) -> None:
        self.vocab_size = len(vocab)
        self.model_params = model_params
        self.batch_size = self.model_params["batch_size"]
        self.block_size = self.model_params["block_size"]

    @abstractmethod
    def _build_model(self) -> torch.nn.Module:
        pass

    # Train a model based on passed in params
    def train(self, data: torch.Tensor) -> None:
        self.model = self._build_model()

        if self.model_params["optimizer"]["type"] == "adam":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.model_params["optimizer"]["lr"])

        for step in range(self.model_params["batch_cnt"]):
            # Zero the gradients
            optimizer.zero_grad(set_to_none=True)

            # Sample a batch of data
            xb, yb = get_batch(data, self.batch_size, self.block_size)

            # Evaluate the loss
            logits, loss = self.model(xb, yb)
            loss.backward()
            optimizer.step()
