"""CLI App for finetuning resnet18 on local data."""
from enum import Enum
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from tqdm import tqdm

from training.dataset import HotdogNotHotdogDataset
from training.model import resnet18_N_class_head
from training.transforms import transforms

# imagenet normalization
normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Onehot(Enum):
    TRUE = Tensor([0, 1])[None, :]
    FALSE = Tensor([1, 0])[None, :]


def finetune_resnet_18(
    dataset_path: Path, model_output_path: Path, max_epochs: int = 10, test_n: int = 10
) -> None:
    """Finetune a resnet18 model on your dataset."""
    dataset = HotdogNotHotdogDataset(
        hotdog_dir=dataset_path / "hot_dog",
        not_hotdog_dir=dataset_path / "not_hot_dog",
    )
    dataloader = DataLoader(dataset=dataset)

    model = resnet18_N_class_head(2)

    loss_fn = CrossEntropyLoss()

    optimizer = Adam(model.parameters(), lr=0.0001)

    for epoch in range(max_epochs):
        with tqdm(dataloader, desc=f"Epoch: {epoch}", unit="images") as dl:
            for i, (batch, label) in enumerate(dl):
                batch.to(device)
                batch = normalize((batch / 255).float())

                optimizer.zero_grad()

                batch = transforms(batch)

                out = model(batch)

                loss = loss_fn(out, Onehot.TRUE if label.item() else Onehot.FALSE)
                loss.backward()

                optimizer.step()

                running_loss += loss.item()
                if i % test_n == test_n - 1:
                    dl.set_description_str(
                        f"Batch {i+1}, Loss: {running_loss / test_n}"
                    )
                    running_loss = 0.0
        # checkpoint every epoch
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            model_output_path / f"checkpoint{epoch}.pt",
        )


if __name__ == "__main__":
    """Little CLI to train with."""
    finetune_resnet_18(dataset_path="data", model_output_path=None)
