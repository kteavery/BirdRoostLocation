import pandas as pd
from BirdRoostLocation import LoadSettings as settings


def replaceRoosts(newcsv, oldcsv):
    # table join on filename
    joined = oldcsv.join(newcsv.set_index("AWS_file"), on="AWS_file")
    print(joined.head())
    joined.to_csv(settings.WORKING_DIRECTORY + "/true_ml_relabels.csv", index=False)


def main():
    newcsv = pd.read_csv(settings.WORKING_DIRECTORY + "/processed_relabels.csv").drop(
        ["flag"], axis=1
    )
    oldcsv = pd.read_csv(settings.WORKING_DIRECTORY + "/true_ml_labels.csv").drop(
        ["lat", "lon"], axis=1
    )
    replaceRoosts(oldcsv, newcsv)


if __name__ == "__main__":
    main()

