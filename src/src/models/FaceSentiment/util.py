import numpy as np
import torch
from torchvision import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score

def undersample_imagefolder_dataset(dataset):
    # Count number of samples per class
    class_counts = np.bincount([label for _, label in dataset.samples])
    min_class_count = min(class_counts)  # Find the minimum number of samples in any class
    
    # Create an array to hold indices of selected samples for each class
    undersample_indices = []
    
    # Loop over each class and randomly select 'min_class_count' samples
    for class_idx in range(len(class_counts)):
        class_indices = np.where(np.array([label for _, label in dataset.samples]) == class_idx)[0]
        np.random.shuffle(class_indices)  # Shuffle indices for random selection
        undersample_indices.extend(class_indices[:min_class_count])  # Take min_class_count samples from each class
    
    # Subset the dataset based on the undersampled indices
    undersampled_samples = [dataset.samples[i] for i in undersample_indices]
    
    # Create a new ImageFolder-like dataset with the undersampled data
    undersampled_dataset = datasets.ImageFolder(root=dataset.root, transform=dataset.transform)
    undersampled_dataset.samples = undersampled_samples
    undersampled_dataset.targets = [s[1] for s in undersampled_samples]  # Update the targets list

    return undersampled_dataset

def get_true_and_pred_labels(model, dataloader, device):
    model.eval()
    true_labels = []
    pred_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())
    
    return np.array(true_labels), np.array(pred_labels)

def plot_results(history):
    acc = history['train_acc']
    val_acc = history['val_acc']
    loss = history['train_loss']
    val_loss = history['val_loss']
    precision = history.get('val_precision', [])
    recall = history.get('val_recall', [])
    f1 = history.get('val_f1', [])
    
    epochs = range(1, len(acc) + 1)
    
    plt.figure(figsize=(24, 12))
    
    # Plotting Training and Validation Accuracy with markers
    plt.subplot(2, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    # Plotting Training and Validation Loss with markers
    plt.subplot(2, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Plotting Validation Precision, Recall, and F1 Score
    if precision and recall and f1:  # Only plot if these metrics are available
        plt.subplot(2, 2, 3)
        plt.plot(epochs, precision, 'go-', label='Validation Precision')  # 'go-' means green color, circle marker, solid line
        plt.plot(epochs, recall, 'mo-', label='Validation Recall')  # 'mo-' means magenta color, circle marker, solid line
        plt.plot(epochs, f1, 'co-', label='Validation F1 Score')  # 'co-' means cyan color, circle marker, solid line
        plt.grid(True)
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Metrics')
    
    plt.tight_layout()
    plt.show()

def get_best_epoch(history):
    val_acc = history['val_acc']
    val_f1 = history.get('val_f1', [])

    if val_f1:  # If F1 scores are available, use them to determine the best epoch
        best_epoch = val_f1.index(max(val_f1)) + 1
        best_f1 = max(val_f1)
        print(f'Best Validation F1 Score {best_f1}, is for epoch {best_epoch}')
    else:  # Otherwise, fall back to accuracy
        best_epoch = val_acc.index(max(val_acc)) + 1
        best_acc = max(val_acc)
        print(f'Best Validation Accuracy Score {best_acc}, is for epoch {best_epoch}')
    
    return best_epoch

# Function to plot confusion matrix
def plot_confusion_matrix(model, dataloader, class_names, device):
    y_true, y_pred = get_true_and_pred_labels(model, dataloader, device)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    
    plt.title("Confusion Matrix")
    plt.xticks(rotation=90)  # Rotate x-axis labels to be vertical
    plt.show()

def plot_per_class_accuracy(model, dataloader, class_names, device):
    y_true, y_pred = get_true_and_pred_labels(model, dataloader, device)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate per-class accuracy: correct predictions / total samples for each class
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    # Plot per-class accuracy
    plt.figure(figsize=(12, 8))
    plt.barh(class_names, per_class_accuracy, color='skyblue')
    plt.xlabel('Accuracy')
    plt.title('Per-Class Accuracy')
    
    # Show accuracy values on the bars
    for i, v in enumerate(per_class_accuracy):
        plt.text(v + 0.01, i, f"{v:.2f}", color='blue', va='center')
    
    plt.xlim([0, 1])  # Accuracy is a value between 0 and 1
    plt.grid(True, axis='x')
    plt.show()
