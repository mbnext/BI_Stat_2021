import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    n = len(y_true)
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = precision * recall / (precision + recall)
    accuracy = (true_positive + true_negative) / n

    return precision, recall, f1, accuracy


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    n = len(y_true)

    true_positive = 0

    for i in range(n):
        if y_pred[i] == y_true[i]:
            true_positive += 1

    accuracy = true_positive / n

    return accuracy


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    n = len(y_true)
    upper_sum = 0
    lower_sum = 0
    mean_y_true = y_true.mean()

    for i in range(n):
        upper_sum += (y_true[i] - y_pred[i])**2
        lower_sum += (y_true[i] - mean_y_true)**2

    r_squared_value = 1 - (upper_sum / lower_sum)

    return r_squared_value


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    n = len(y_true)

    sq_sum = 0
    for i in range(n):
        sq_sum += (y_true[i] - y_pred[i])**2

    mse_value = sq_sum / n

    return mse_value


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """
    n = len(y_true)

    abs_sum = 0
    for i in range(n):
        abs_sum += abs(y_true[i] - y_pred[i])

    mae_value = abs_sum / n

    return mae_value