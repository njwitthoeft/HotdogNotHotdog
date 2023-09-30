"""Model utilities."""
from torch.nn import Module, Linear
from torchvision.models import resnet18, ResNet18_Weights


def resnet18_N_class_head(N: int) -> Module:
    """Create a network with an `N` class linear classifier head."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    fc_in = model.fc.in_features

    model.fc = Linear(in_features=fc_in, out_features=N)

    return model
