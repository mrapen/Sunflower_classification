"""This module has the functions of analyzing the training model"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as m


def analyze_model_performance(model, test_dl, device, classes):
    """
    Analyze model performance on test dataset.
    Returns predictions, ground truth and various metrics.
    """
    model.eval()
    all_predictions = []
    all_labels = []
    class_names = list(classes.keys())
    target_names = [str(classes[key]) for key in class_names]
    print("Analyzing model performance...")
    with torch.no_grad():
        for batch in test_dl:
            qry_im = batch["qry_im"].to(device)
            qry_gt = batch["qry_gt"].to(device)
            # Get model predictions
            outputs = model(qry_im)
            _, predictions = torch.max(outputs, 1)
            # Store predictions and labels
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(qry_gt.cpu().numpy())
    # Calculate accuracy
    accuracy = m.accuracy_score(all_labels, all_predictions)
    print(f"Test Accuracy: {accuracy:.4f}")
    # Generate and plot confusion matrix
    cm = m.confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap="Blues")
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    # Add text annotations to the confusion matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    # Print classification report
    print("\nClassification Report:")
    print(m.classification_report(all_labels,
                                all_predictions,
                                labels=class_names,
                                target_names=target_names))
    return all_predictions, all_labels, accuracy

def analyze_class_performance(all_labels, all_preds, classes):
    """
    Analyze per-class performance metrics
    """
    class_names = list(classes.keys())
    class_indices = list(classes.values())
    # Calculate per-class accuracy
    class_accuracies = {}
    for i, class_name in enumerate(class_names):
        mask = all_labels == class_indices[i]
        if np.sum(mask) > 0:  # Avoid division by zero
            class_accuracies[class_name] = np.mean(all_preds[mask] == all_labels[mask])
        else:
            class_accuracies[class_name] = 0
    # Plot class accuracies
    plt.figure(figsize=(10, 6))
    plt.bar(class_accuracies.keys(), class_accuracies.values())
    plt.ylabel('Accuracy')
    plt.xlabel('Class')
    plt.title('Per-Class Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    return class_accuracies
