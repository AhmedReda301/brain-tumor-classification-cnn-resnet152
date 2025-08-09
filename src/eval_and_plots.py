import matplotlib.pyplot as plt
import os
import json

def save_training_plots(loss_history, metric_history, model_name):
    output_dir = os.path.join("results", model_name)
    os.makedirs(output_dir, exist_ok=True)

    # Plot Loss
    plt.figure()
    plt.plot(loss_history["train_loss"], label='Train Loss')
    plt.plot(loss_history["val_loss"], label='Val Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "loss.png"))
    plt.close()

    # Plot Accuracy
    plt.figure()
    plt.plot(metric_history["train_acc"], label='Train Accuracy')
    plt.plot(metric_history["val_acc"], label='Validation Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "accuracy.png"))
    plt.close()


def save_history_json(history, model_name):
    output_dir = os.path.join("results", model_name)
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "history.json")

    with open(save_path, "w") as f:
        json.dump(history, f, indent=4)
