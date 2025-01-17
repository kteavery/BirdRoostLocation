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


def get_skill_scores_localization(predictions, truths):
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    dice_list = []
    tpr = []
    fpr = []

    predictions = np.round(predictions)
    truths = np.round(truths)

    for prediction, truth in zip(predictions, truths):

        overlap = np.equal(prediction, truth)
        disjoint = np.not_equal(prediction, truth)

        len_overlap = np.count_nonzero(overlap == True)
        dice = len_overlap / 57600  # 240*240 = 57600
        dice_list.append(dice)
        print(dice)

        true_pos = np.logical_and(prediction, overlap)
        true_neg = np.logical_xor(overlap, true_pos)

        false_pos = np.logical_and(prediction, disjoint)
        false_neg = np.logical_xor(disjoint, false_pos)

        tp = np.count_nonzero(true_pos == True) / 57600
        tn = np.count_nonzero(true_neg == True) / 57600
        fp = np.count_nonzero(false_pos == True) / 57600
        fn = np.count_nonzero(false_neg == True) / 57600

        print(tp)

        fpr.append(fp / (fp + tn))
        tpr.append(tp / (tp + fn))

        TP += tp
        TN += tn
        FP += fp
        FN += fn

    ACC = (TP + TN) / (TP + FP + FN + TN)
    if (TP + FN) > 0:
        TPR = (TP) / (TP + FN)
    else:
        TPR = 0
    if (TN + FP) > 0:
        TNR = (TN) / (TN + FP)
    else:
        TNR = 0
    try:
        ROC_AUC = auc(np.sort(np.array(fpr)[::-1]), np.sort(np.array(tpr)[::-1]))
    except Exception as e:
        print(e)
        ROC_AUC = 0.0

    return ACC, TPR, TNR, ROC_AUC, np.mean(np.array(dice_list)), fpr, tpr


def print_skill_scores(ACC, TPR, TNR, ROC_AUC, dice=None):
    print("\tACC", ACC)
    print("\tTPR", TPR)
    print("\tTNR", TNR)
    print("\tAUC", ROC_AUC)
    print("\tdice", dice)
