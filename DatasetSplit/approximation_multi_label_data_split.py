# pylint: disable=missing-module-docstring, line-too-long, import-error, wrong-import-position

import collections

import numpy as np
import pandas as pd
import plotly.express as px
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def create_one_hot_dataframe(data_dict: dict[str, list[str]]):
    """create dataframe for splitting data.

    Args:
        data_dict (dict[str, list[str]]): key: image_file, values: list of labels.

    Returns:
        pd.DataFrame: one hot DataFrame with these columns: images_file | labels (multi) | label_1 | label_2 | ... | label_n.
        dict[str, int]: class count.
    """

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(list(data_dict.items()), columns=["images_file", "labels"])

    # Get list of labels
    labels = list(df["labels"])

    # Flatten the list using list comprehension
    flattened_labels = [item for sublist in labels for item in sublist]

    class_count = collections.Counter(sorted(flattened_labels))

    # Get all unique numbers from the 'labels' lists
    all_numbers = sorted(set(number for sublist in df["labels"] for number in sublist))

    # Create a new DataFrame for one-hot encoding
    one_hot_df = df[["images_file", "labels"]].copy()

    # Create one-hot encoded columns for each unique number
    for number in all_numbers:
        one_hot_df[number] = one_hot_df["labels"].apply(lambda x: 1 if str(number) in x else 0)

    return one_hot_df, class_count


def approximation_multi_label_split(
    labels_path: list[str], labels_dir: str, n_splits: int = 5, shuffle: bool = True, random_state: int = 1234
):
    """approximately split multi label data.

    Args:
        labels_path (list[str]): labels files in .txt yolo format (os.listdir is recommended).
        labels_dir (str): root label directory.
        n_splits (int, optional): number of kflod split. Defaults to 5.
        shuffle (bool, optional): shuffle data or not. Defaults to True.
        random_state (int, optional): random state for reproducing the result. Defaults to 1234.

    Returns:
        list[int]: train index.
        list[int]: test index.
    """

    # Initialize dict to save data
    data_dict = {}

    # Read every label file and updata data_dict
    for label_file in labels_path:
        with open(labels_dir + label_file, "r") as file_:
            lines = file_.readlines()
        file_.close()

        data_dict[label_file] = []

        for line in lines:
            line = line.split(" ")

            if line[0] not in data_dict[label_file]:
                data_dict[label_file].append(line[0])

    # Create one_hot dataframe from data_dict
    one_hot_df, class_count = create_one_hot_dataframe(data_dict=data_dict)

    # Create figure
    fig = px.parallel_categories(one_hot_df[class_count.keys()], title="Parallel categories plot of targets")

    # X: list of image file, Y: list of one hot encoded labels
    X, Y = one_hot_df["images_file"].to_numpy(), one_hot_df[class_count.keys()].to_numpy(dtype=np.float32)

    # Create split generator
    multi_label_split = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    # Split data
    train_index, test_index = next(iter(multi_label_split.split(X, Y)))

    print("Num train: ", len(train_index))
    print("Num test: ", len(test_index))
    print("TRAIN:", train_index, "TEST:", test_index)

    # Get y_train, y_test
    y_train, y_test = Y[train_index], Y[test_index]

    #
    kfold_train_df = pd.DataFrame(columns=class_count.keys(), data=y_train)
    kfold_test_df = pd.DataFrame(columns=class_count.keys(), data=y_test)

    # Create figure
    fig_train = px.parallel_categories(kfold_train_df[class_count.keys()], title="categories plot of y_train")
    fig_test = px.parallel_categories(kfold_test_df[class_count.keys()], title="categories plot of y_test")

    # Save as PNG
    fig.write_image("original_class_proportion.png")
    fig_train.write_image("split_train_class_proportion.png")
    fig_test.write_image("split_test_class_proportion.png")

    # Show
    fig.show()
    fig_train.show()
    fig_test.show()

    return train_index, test_index
