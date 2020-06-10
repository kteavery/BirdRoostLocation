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
        roc_auc[i] = auc(fpr[i], tpr[i])

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

    for i, label in zip(range(len(y_label)), y_label):
        plt.plot(
            fpr[i],
            tpr[i],
            lw=1.5,
            marker=markers[i],
            markersize=7,
            markevery=0.08,
            label="{0} (AUC = {1:0.2f})".format(label, roc_auc[i]),
        )

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
    plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file, format="eps", dpi=1000)
    plt.show()


def roc_curve_from_csv(curves):
    y_predicted_values = []
    ground_truths = []

    for curve in curves:
        df = pandas.read_csv(
            settings.WORKING_DIRECTORY + "/true_predictions_" + curve + ".csv",
            names=["filenames", "truth", "predictions"],
        )

        print(df.head())
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

    y_predicted_values = np.array(y_predicted_values)
    ground_truths = np.array(ground_truths)

    create_roc_curve(
        ground_truths,
        y_predicted_values,
        curves,
        title="Detection ROC Curve",
        save_file="detection_roc.png",
    )


def run_with_example_data():
    y_score = np.array(
        [
            [
                0.91,
                0.81,
                0.71,
                0.61,
                0.551,
                0.541,
                0.531,
                0.521,
                0.51,
                0.5051,
                0.45,
                0.38,
                0.38,
                0.37,
                0.316,
                0.315,
                0.314,
                0.313,
                0.310,
                0.11,
            ],
            [
                0.95,
                0.9,
                0.75,
                0.6,
                0.56,
                0.54,
                0.53,
                0.52,
                0.51,
                0.505,
                0.4,
                0.39,
                0.38,
                0.37,
                0.36,
                0.30,
                0.34,
                0.30,
                0.30,
                0.1,
            ],
        ]
    )
    y_test = np.array(
        [
            [1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
            [1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0],
        ]
    )
    y_class = ["example_1", "example_2"]

    create_roc_curve(y_pred=y_score, y_test=y_test, y_label=y_class)


def main():
    curves["Reflectivity", "Velocity", "Rho_HV", "Zdr", "Aggregate"]
    roc_curve_from_csv(curves)


if __name__ == "__main__":
    main()
