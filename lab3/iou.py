import numpy as np


def calculate_iou(gt, pred, confidence=0.0):
    # Thresholding prediction mask based on confidence
    pred_thresholded = pred > confidence

    intersection = np.logical_and(gt, pred_thresholded)
    union = np.logical_or(gt, pred_thresholded)
    iou = np.sum(intersection) / (np.sum(union) + 1e-5)
    return iou


if __name__ == '__main__':
    gt = np.array([1, 1, 0, 1])
    pred = np.array([0.6, 1, 0.1, 0.7])
    iou = calculate_iou(gt, pred)
    print(iou)