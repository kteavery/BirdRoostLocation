import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
from BirdRoostLocation import utils
from BirdRoostLocation import LoadSettings as settings
import os
import pandas


def create_roc_curve(y_test, y_pred, y_label, title=None, save_file=None):
    """Given predicted y, actual y, and label, draw ROC curve for each label.
    Args:
        y_test: Actual value of Y. Numpy array of shape (n, num_samples)
        y_pred: Predicted value of Y. Numpy array of shape (n, num_samples)
        y_label: Label of each curve that will be displayed, Numpy array of
            shape (n)
    """
    n_classes = len(y_label)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        print(i, y_test[i].shape)
        print(i, y_pred[i].shape)
        fpr[i], tpr[i], _ = roc_curve(y_test[i], y_pred[i])
        # if np.isnan(np.array(fpr[i])).any():
        #     fpr[i] = [1.0] * len(fpr[i])
        #     fpr[i] = np.array(fpr[i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        print("AUC: ")
        print(roc_auc[i])

    roc_plot(fpr, tpr, roc_auc, y_label, title, save_file)


def roc_plot(fpr, tpr, roc_auc, y_label, title=None, save_file=None):
    """Plot the roc curve.
    Args:
        fpr: False Positive Rate
        tpr: True Positive Rate
        roc_auc: ROC Accuracy Score
        y_label: Label for each ROC curve
    """
    # Plot all ROC curves

    markers = ["P", "<", "8", "d"]
    plt.figure(figsize=(5, 5))
    print(tpr[2])
    print(fpr[2])

    for i, label in zip(range(len(y_label)), y_label):
        print(i)
        print(label)
        plt.plot(
            fpr[i],
            tpr[i],
            lw=1.5,
            marker=markers[i],
            markersize=7,
            markevery=0.06,
            label="{0} (AUC = {1:0.2f})".format(label, roc_auc[i]),
        )
        plt.show()

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.yticks(fontsize=13.5)
    plt.xticks(fontsize=13.5)
    plt.xlabel("False Positive Rate", fontsize=13.5)
    plt.ylabel("True Positive Rate", fontsize=13.5)
    if title is not None:
        plt.title(title, fontsize=14.5)
    else:
        plt.title("Detection - ROC curve", fontsize=14.5)
    plt.legend(loc="lower right", fontsize=11)
    # if save_file is not None:
    #     plt.savefig(save_file)
    plt.show()


def roc_curve_from_csv(curves):
    y_predicted_values = []
    ground_truths = []

    for i in range(0, 5):
        for curve in curves:
            df = pandas.read_csv(
                settings.WORKING_DIRECTORY
                + "true_predictions_"
                + curve
                + str(i)
                + ".csv",
                names=["filenames", "truth", "predictions"],
            )

            # print(df.head())
            truth = df["truth"]
            prediction = df["predictions"]

            # ACC, TPR, TNR, ROC_AUC = get_skill_scores(prediction, truth)

            print(curve)
            print(prediction.shape)
            print(truth.shape)
            print(prediction)

            prediction = prediction[~np.isnan(prediction)]

            truth = truth[~np.isnan(truth)]
            print(prediction.shape)
            print(truth.shape)

            y_predicted_values.append(prediction)
            ground_truths.append(truth)
            print("APPENDED")
            print(np.array(y_predicted_values).shape)
            print(np.array(ground_truths).shape)

    y_predicted_values = np.array(y_predicted_values)
    ground_truths = np.array(ground_truths)
    print(y_predicted_values.shape)
    print(ground_truths.shape)
    # print(ground_truths)

    create_roc_curve(
        ground_truths,
        y_predicted_values,
        curves,
        title="Detection ROC Curve",
        save_file=settings.WORKING_DIRECTORY + "detection_roc.png",
    )


def main():
    curves = ["Reflectivity", "Velocity", "Rho_HV", "Zdr"]
    roc_curve_from_csv(curves)


if __name__ == "__main__":
    main()
