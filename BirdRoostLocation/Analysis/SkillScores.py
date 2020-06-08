import numpy as np
from sklearn.metrics import roc_curve, auc


def get_skill_scores(predictions, truths):
    """Compute skill score and accuracy metrics for the machine learning
    results.
    
    Args:
        predictions: A numpy array of probabilities produced by the machine
            learning results.
        truths: A numpy array of ground truth to compare the predicted values
            against. Roost = 1 and NoRoost = 0
    
    Returns:
        ACC: Accuracy
        TPR: True positive rate
        TNR: True negative rate
        ROC_AUC: ROC area under curve
    """
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0

    predictions = predictions[~np.isnan(predictions)]
    truths = truths[~np.isnan(truths)]

    for prediction, truth in zip(np.around(predictions), truths):
        if prediction == 0 and truth == 0:
            TN += 1
        elif prediction == 0 and truth == 1:
            FN += 1
        elif prediction == 1 and truth == 0:
            FP += 1
        elif prediction == 1 and truth == 1:
            TP += 1

    ACC = (TP + TN) / (TP + FP + FN + TN)
    if (TP + FN) > 0:
        TPR = (TP) / (TP + FN)
    else:
        TPR = 0
    if (TN + FP) > 0:
        TNR = (TN) / (TN + FP)
    else:
        TNR = 0
    fpr, tpr, _ = roc_curve(truths, predictions)
    ROC_AUC = auc(fpr, tpr)

    return ACC, TPR, TNR, ROC_AUC


def get_skill_scores_regression(predictions, truths, cutoff):
    T = 0.0
    F = 0.0

    predictions = predictions[~np.isnan(predictions)]
    truths = truths[~np.isnan(truths)]

    for prediction, truth in zip(predictions, truths):
        print(prediction)
        print(truth)
        diff = abs(prediction - truth)
        if diff < cutoff:
            T += 1
        elif diff >= cutoff:
            F += 1

    ACC = T / (T + F)

    return ACC


def print_skill_scores(ACC, TPR, TNR, ROC_AUC):
    print("\tACC", ACC)
    print("\tTPR", TPR)
    print("\tTNR", TNR)
    print("\tAUC", ROC_AUC)

    # This command prints a row of my latex table
    print(f"\t& {ACC:.3f} & {TPR:.3f} & {TNR:.3f} & {ROC_AUC:.3f} \\\\")
