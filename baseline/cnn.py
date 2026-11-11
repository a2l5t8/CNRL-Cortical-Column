"""Small gradient-trained comparison, independent of the spiking model.

Input: [batch, number_of_fixations, patch_height, patch_width].
Architecture matches the original CNN grid experiment.
"""

import torch
from torch import nn


class PatchCNN(nn.Module):
    def __init__(self, grid=3, classes=2, position=True, kernel_size=5):
        super().__init__()
        self.position = position
        self.grid = grid
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(5),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(400 * (grid * grid if position else 1), 64),
            nn.ReLU(),
            nn.Linear(64, classes),
        )

    def forward(self, patches):
        batch, fixations, height, width = patches.shape
        if fixations != self.grid**2:
            raise ValueError("Patch count does not match fixation grid")
        features = self.encoder(patches.reshape(batch * fixations, 1, height, width))
        features = features.reshape(batch, fixations, 400)
        return self.head(features.flatten(1) if self.position else features.mean(1))


def fit(
    model,
    train_patches,
    train_labels,
    validation_patches,
    validation_labels,
    epochs=20,
    batch_size=128,
    seed=42,
    *,
    prepare=lambda batch: batch,
):
    """Train only on training tensors; return final model and per-epoch accuracy.

    Seed torch before constructing the model to reproduce its initialization.
    The optional prepare function transforms each input batch before forwarding.
    """
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    generator = torch.Generator(device=train_labels.device).manual_seed(seed + 10000)
    history = []
    for epoch in range(epochs):
        model.train()
        for indices in torch.randperm(
            len(train_labels), device=train_labels.device, generator=generator
        ).split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(
                model(prepare(train_patches[indices].to(device))),
                train_labels[indices].to(device),
            )
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            predictions = torch.cat(
                [
                    model(prepare(batch.to(device))).argmax(1)
                    for batch in validation_patches.split(batch_size)
                ]
            )
            accuracy = (
                (predictions == validation_labels.to(device)).float().mean().item()
            )
        history.append(dict(epoch=epoch + 1, validation_accuracy=accuracy))
    return model, history
