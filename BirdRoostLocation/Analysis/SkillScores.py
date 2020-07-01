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


# def get_skill_scores_regression(predictions, truths, cutoff):
#     T = 0.0
#     F = 0.0

#     predictions = predictions[~np.isnan(predictions)]
#     truths = truths[~np.isnan(truths)]

#     for prediction, truth in zip(predictions, truths):
#         print(prediction)
#         print(truth)
#         diff = abs(prediction - truth)
#         if diff < cutoff:
#             T += 1
#         elif diff >= cutoff:
#             F += 1

#     ACC = T / (T + F)

#     return ACC


def get_skill_scores_localization(predictions, truths):
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0

    predictions = predictions[~np.isnan(predictions)]
    truths = truths[~np.isnan(truths)]

    for prediction, truth in zip(predictions, truths):
        print(prediction.shape)
        print(truth.shape)

        overlap = np.equal(prediction, truth)
        disjoint = np.not_equal(prediction, truth)

        len_overlap = np.count_nonzero(overlap == True)
        dice = len_overlap / 57600  # 240*240=57600
        print(dice)

        true_pos = np.logical_and(prediction, overlap)
        true_neg = np.logical_xor(overlap, true_pos)

        false_pos = np.logical_and(prediction, disjoint)
        false_neg = np.logical_xor(disjoint, false_pos)

        TP += np.count_nonzero(true_pos == True) / 57600
        TN += np.count_nonzero(true_neg == True) / 57600
        FP += np.count_nonzero(false_pos == True) / 57600
        FN += np.count_nonzero(false_neg == True) / 57600

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

    return ACC, TPR, TNR, ROC_AUC, dice


def print_skill_scores(ACC, TPR, TNR, ROC_AUC, dice=None):
    print("\tACC", ACC)
    print("\tTPR", TPR)
    print("\tTNR", TNR)
    print("\tAUC", ROC_AUC)
    print("\tdice", dice)
