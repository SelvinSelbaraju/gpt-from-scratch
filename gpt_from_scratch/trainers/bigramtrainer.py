from torch import nn

from gpt_from_scratch.models.bigram import BigramModel
from gpt_from_scratch.trainers.basetrainer import BaseTrainer


class BigramTrainer(BaseTrainer):
    def _build_model(self) -> nn.Module:
        return BigramModel(self.block_size, self.vocab_size)
