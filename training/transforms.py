"""Transformation utilities."""
from torchvision.transforms import v2

normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transforms = v2.Compose(
    [
        # ...
        v2.RandomResizedCrop(
            size=(224, 224),
            antialias=True,
            scale=0.25,
        ),  # Or Resize(antialias=True)
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip,
        v2.RandomRotation(180),
        v2.ColorJitter(brightness=0.5, hue=0.3),
        v2.RandomPerspective(),
        v2.ElasticTransform(),
    ]
)
