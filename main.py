import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import numpy as np

from model import DeepLGR
from data import TaxiBJDataset, load_taxibj

# Set seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Set the device and batch size according to the paper
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
batch_size = 16

# Load data
data, dates = load_taxibj(
    "data/TaxiBJ/BJ15_M32x32_T30_InOut.h5"
)

# Configure sample size for closeness, period and trend
t_params = (5, 3, 3)

# Construct dataset
dataset = TaxiBJDataset(
    data,
    dates,
    t_params=t_params
)

# Set the ratio for the split
train_ratio = 0.8
validation_ratio = 0.9

# Calculate the size of the training and validation dataset
dataset_size = len(dataset)
train_size = int(train_ratio * dataset_size)
validation_size = int(validation_ratio * dataset_size)

# Create datasets
train_dataset = Subset(
    dataset,
    range(0, train_size)
)

val_dataset = Subset(
    dataset,
    range(train_size, validation_size)
)

test_dataset = Subset(
    dataset,
    range(validation_size, dataset_size)
)

# Load the datasets
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Test samples:", len(test_dataset))


def train_loop(
    model,
    train_loader,
    val_loader,
    device,
    epochs=100,
    learning_rate=0.005,
):

    # Keep track of the best epoch
    best_epoch = 0

    # L1Loss is equivalent to MAE used in the paper
    criterion = nn.L1Loss()

    # Use the Adam optimizer with the learning rate like in the paper
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    best_val_mae = float("inf")

    history = {
        "train_loss": [],
        "val_mae": [],
        "val_smape": []
    }

    for epoch in range(epochs):
        # Set model to training mode
        model.train()

        # Keep track of loss
        total_loss = 0.0
        total_samples = 0

        # Since the model uses three input tensors as input and concatenates them itself,
        # we need to retrieve them separately and store them all on the used device.
        # Naturally we also need to store the targets.
        for closeness, period, trend, target in train_loader:
            closeness = closeness.to(device)
            period = period.to(device)
            trend = trend.to(device)
            target = target.to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            predictions = model((
                closeness,
                period,
                trend
            ))

            # Calculate loss
            loss = criterion(predictions, target)

            # Backpropagation
            loss.backward()

            # Update parameters
            optimizer.step()

            # Keep track of total loss
            total_loss += loss.item() * target.size(0)
            total_samples += target.size(0)

        # Calculate MAE
        train_loss = total_loss / total_samples

        # Calculate performance on evaluation set
        val_mae, val_smape = evaluate(
            model,
            val_loader,
            device
        )

        # Store loss
        history["train_loss"].append(train_loss)
        history["val_mae"].append(val_mae)
        history["val_smape"].append(val_smape)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train MAE: {train_loss:.4f} "
            f"Val MAE: {val_mae:.4f} "
            f"Val SMAPE: {val_smape:.4f}"
        )

        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch

            torch.save(
                model.state_dict(),
                f"best_model_epoch_{best_epoch}.pth"
            )

    return model, history, best_epoch


def evaluate(model, dataloader, device):
    # Set mode to test mode
    model.eval()

    # Keep track of both total absolute error (for MAE) and smape metric
    total_absolute_error = 0.0
    total_smape = 0.0
    total_elements = 0

    with torch.no_grad():
        for closeness, period, trend, target in dataloader:
            # Again, store each input tensor and the labels on the correct device
            closeness = closeness.to(device)
            period = period.to(device)
            trend = trend.to(device)
            target = target.to(device)

            # Obtain predictions
            predictions = model((
                closeness,
                period,
                trend
            ))

            # Store total absolute error
            total_absolute_error += torch.abs(
                predictions - target
            ).sum().item()

            # Store SMAPE error
            denominator = (
                torch.abs(target) +
                torch.abs(predictions)
            ).clamp_min(1e-8)

            total_smape += (
                torch.abs(predictions - target) /
                denominator
            ).sum().item()

            total_elements += target.numel()

    # Calculate evaluation metrics
    mae = total_absolute_error / total_elements
    smape = total_smape / total_elements

    return mae, smape

# Initiate the model
model = DeepLGR(
    in_channels=2,
    out_channels=2,
    n_residuals=9,
    n_filters=64,
    t_params=t_params,
    height=32,
    width=32,
    pred_step=1,
    flag_global=True,
    predictor="td"
).to(device)

# Train the model
model, history, best_epoch = train_loop(
    model,
    train_loader,
    val_loader,
    device,
    epochs=100,
    learning_rate=0.005
)

# Evaluate the best model
model.load_state_dict(torch.load(f"best_model_epoch_{best_epoch}.pth", weights_only=True))

test_mae, test_smape = evaluate(
    model,
    test_loader,
    device
)

print(f"Test MAE: {test_mae:.4f}")
print(f"Test SMAPE: {test_smape:.4f}")