# ==================================
# calculate_metrics() for evaluation
# ==================================

# Import required packages
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics(y_true, y_pred):
    """
    Calculate accuracy, precision, recall, and F1 score.
    Args:
        y_true (list): A list of true labels.
        y_pred (list): A list of predicted labels.
    """
    accuracy = accuracy_score(y_true, y_pred)
    avg_f1 = f1_score(y_true, y_pred, average='macro')
    avg_recall = recall_score(y_true, y_pred, average='macro')
    return accuracy, avg_f1, avg_recall

def calculate_metrics2(labels, predictions):
    human_rec, machine_rec, avg_rec = calculate_metrics(labels, predictions)

    correct = sum(1 for pred, label in zip(predictions, labels) if pred == label)
    acc = correct / len(labels) if len(labels) > 0 else 0

    true_positives = sum(1 for pred, label in zip(predictions, labels) if pred == 1 and label == 1)
    false_positives = sum(1 for pred, label in zip(predictions, labels) if pred == 1 and label == 0)
    false_negatives = sum(1 for pred, label in zip(predictions, labels) if pred == 0 and label == 1)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return human_rec, machine_rec, avg_rec, acc, precision, recall, f1