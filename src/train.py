import torch
import torch.nn as nn
import torch.optim as optim
from models import CNN_TUMOR, ResNet152_TUMOR
from load_dataset import get_dataloaders
from eval_and_plots import save_training_plots, save_history_json
from config import DEVICE, EPOCHS, LR
from tqdm import tqdm
from sklearn.metrics import classification_report
import os

# Define model parameters
params_model = {
    "shape_in": (3, 256, 256),     # C x H x W (adjust if needed)
    "initial_filters": 8,
    "num_fc1": 100,
    "dropout_rate": 0.25,
    "num_classes": 2              # Change if you have more/less classes
}

# Accuracy calculation helper
def calculate_accuracy(preds, labels):
    _, predicted = torch.max(preds, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)

# Training function
def train(model_name):
    name_lower = model_name.lower()

    if name_lower == "cnn_tumor":
        model = CNN_TUMOR(params_model).to(DEVICE)
    elif name_lower == "resnet152_tumor":
        model = ResNet152_TUMOR(num_classes=2, pretrained=True, freeze_backbone=True).to(DEVICE)
    else:
        raise ValueError(f"Unknown model name: {model_name} — use 'CNN_TUMOR' or 'ResNet152_TUMOR'.")

    train_loader, val_loader = get_dataloaders()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        # -------- Unfreeze at halfway --------
        if name_lower == "resnet152_tumor" and epoch == EPOCHS // 2:
            for param in model.model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=LR / 10)  # smaller LR for fine-tuning
            print("\nUnfroze ResNet backbone for fine-tuning.")
        # -------- Training --------
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

        for batch in tqdm(train_loader, desc="Training", leave=False):
            images = batch['image'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += calculate_accuracy(outputs, labels)

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        # -------- Validation --------
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                images = batch['image'].to(DEVICE)
                labels = batch['label'].to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_acc += calculate_accuracy(outputs, labels)

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
              f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # ---- Final Classification Reports ----
    os.makedirs(f"results/{model_name}", exist_ok=True)

    # TRAIN REPORT
    y_true_train, y_pred_train = [], []
    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            images = batch['image'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true_train.extend(labels.cpu().numpy())
            y_pred_train.extend(preds.cpu().numpy())

    train_report = classification_report(y_true_train, y_pred_train, digits=4)
    print("\nFinal Train Classification Report:\n", train_report)

    # VAL REPORT
    y_true_val, y_pred_val = [], []
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true_val.extend(labels.cpu().numpy())
            y_pred_val.extend(preds.cpu().numpy())

    val_report = classification_report(y_true_val, y_pred_val, digits=4)
    print("\nFinal Validation Classification Report:\n", val_report)

    # Save reports to file
    with open(f"results/{model_name}/classification_report.txt", "w") as f:
        f.write("=== TRAIN REPORT ===\n")
        f.write(train_report + "\n")
        f.write("=== VALIDATION REPORT ===\n")
        f.write(val_report + "\n")

    # Save model
    os.makedirs('results', exist_ok=True)
    torch.save(model.state_dict(), f"models/{model_name}.pth")

    return history


if __name__ == "__main__":
    print("Starting training...")
    model_name = 'CNN_TUMOR'      # or use ResNet152_TUMOR
    history = train(model_name)
    print("Training completed.")

    save_training_plots(history, history, model_name)
    save_history_json(history, model_name)
    print(f"Results saved to 'results/{model_name}/' folder.")




