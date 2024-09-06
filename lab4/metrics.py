import numpy as np


def binary_classification_metrics(prediction, ground_truth):
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
    return accuracy, precision, recall, f1


def multiclass_accuracy(pred, gt):
    # Get the index of the predicted class for each sample
    if gt.shape != pred.shape:
        raise ValueError("Shape of ground truth and predictions must match")

        # Calculate the number of correct predictions
    correct_predictions = np.sum(gt == pred)

    # Calculate accuracy
    accuracy = correct_predictions / len(gt)

    return accuracy