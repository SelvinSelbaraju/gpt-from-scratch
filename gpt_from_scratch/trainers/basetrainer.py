from typing import Dict, Any, Tuple
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

    # Get random batches of train and validation
    # Average loss amongst those
    def _estimate_loss(
        self, train_data: torch.Tensor, val_data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        eval_iters = 200
        self.model.eval()
        train_losses = torch.zeros(eval_iters)
        val_losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            xb_train, yb_train = get_batch(
                train_data, self.batch_size, self.block_size
            )
            xb_val, yb_val = get_batch(
                val_data, self.batch_size, self.block_size
            )
            _, train_losses[i] = self.model(xb_train, yb_train)
            _, val_losses[i] = self.model(xb_val, yb_val)
        self.model.train()
        return train_losses.mean(), val_losses.mean()

    # Train a model based on passed in params
    def train(self, train_data: torch.Tensor, val_data: torch.Tensor) -> None:
        self.model = self._build_model()

        if self.model_params["optimizer"]["type"] == "adam":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.model_params["optimizer"]["lr"],
            )

        for step in range(self.model_params["batch_cnt"]):
            # Zero the gradients
            optimizer.zero_grad(set_to_none=True)

            # Sample a batch of data
            xb, yb = get_batch(train_data, self.batch_size, self.block_size)

            # Evaluate the loss
            logits, loss = self.model(xb, yb)
            loss.backward()
            optimizer.step()

            # Optionally print the metric
            if step % 1000 == 0 and step > 0:
                train_loss, val_loss = self._estimate_loss(
                    train_data, val_data
                )
                print(
                    f"Iteration {step} - Train loss: {train_loss}, "
                    f"Val loss: {val_loss}"
                )
