from typing import Tuple
import torch


def load_txt_file(data_path: str) -> str:
    with open(data_path, "r") as f:
        text = f.read()
    return text


def train_test_split(
    data: torch.Tensor, train_size: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    if train_size < 0.0 or train_size > 1.0:
        raise ValueError(
            f"train_size should be between 0 and 1 but got {train_size}"
        )
    n = int(len(data) * train_size)
    return data[:n], data[n:]


def get_batch(data: torch.Tensor, batch_size: int, block_size: int):
    # A sample can start anywhere between 0 and len-block-size
    # The highest possible is one below the integer we put
    # We generate a random start point for each parallel batch
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # Stack each context on top of each other as rows of a tensor
    # Dimension will be batch_size by block_size
    x = torch.stack([data[i : i + block_size] for i in ix])  # noqa: E203
    y = torch.stack(
        [data[i + 1 : i + 1 + block_size] for i in ix]  # noqa: E203
    )
    return x, y
