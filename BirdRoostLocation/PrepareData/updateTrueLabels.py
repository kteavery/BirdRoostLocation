import pandas as pd
from BirdRoostLocation import LoadSettings as settings


def replaceRoosts(newcsv, oldcsv):
    # table join on filename
    joined = newcsv.join(oldcsv.set_index("AWS_file"), on="AWS_file").drop(
        ["lat", "lon"], axis=1
    )
    # toDelete = oldcsv[oldcsv.AWS_file.isin(joined["AWS_file"].tolist())]
    # oldLabels = oldcsv.drop(oldcsv.loc[toDelete.index])
    oldcsv[~oldcsv.AWS_file.isin(joined.AWS_file)]
    oldcsv = oldcsv.rename(columns={"lat": "latitude", "lon": "longitude"})
    combined = pd.concat([joined, oldcsv])
    print(joined.head())
    print(oldcsv.head())
    print(combined.head())

    combined.to_csv(
        settings.WORKING_DIRECTORY + "/true_ml_relabels_edited.csv", index=False
    )


def main():
    newcsv = pd.read_csv(settings.WORKING_DIRECTORY + "/processed_relabels.csv").drop(
        ["flag"], axis=1
    )
    oldcsv = pd.read_csv(settings.WORKING_DIRECTORY + "/true_ml_labels.csv")
    replaceRoosts(newcsv, oldcsv)


if __name__ == "__main__":
    main()
