import BirdRoostLocation.LoadSettings as settings
import os
import pandas


def create_subset_labels(csv_input_path, subset_path, csv_output_path):
    full = pandas.read_csv(csv_input_path)
    subset = pandas.read_csv(subset_path)

    full_basenames = {}
    subset_basenames = []
    full_file_list = list(full["AWS_file"])
    subset_file_list = list(subset["file_names"])

    for i, file_name in enumerate(full_file_list):
        fbasename = file_name[:23]
        full_basenames[fbasename] = i

    for i, file_name in enumerate(subset_file_list):
        sbasename = file_name[2:25]
        if sbasename in full_basenames:
            subset_basenames.append(full_basenames[sbasename])

    output_pd = full.loc[subset_basenames]
    output_pd.to_csv(csv_output_path, index=False)


def main():
    create_subset_labels(
        csv_input_path=settings.LABEL_CSV,
        subset_path=settings.SUBSET_CSV,
        csv_output_path=settings.SUBSET_LABEL_CSV,
    )


if __name__ == "__main__":
    os.chdir(settings.WORKING_DIRECTORY)
    main()
