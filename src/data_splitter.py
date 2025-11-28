"""
Split a processed CSV dataset into train/test sets and save them separately.

This module defines the ``DataSplitter`` class, which loads a pre-processed CSV file
(containing one or more label columns whose names start with ``label_``), separates
features from labels, performs a stratified train-test split, and saves the resulting
train and test sets as new CSV files in the ``data/`` directory.
"""

import logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define BASE_DIR as the project's root: Go up two parent directories from this script's location
BASE_DIR = Path(__file__).parent.parent

class DataSplitter:
    """
    Load a processed dataset, split it into train/test sets, and persist the splits.

    The class expects the input CSV to contain feature columns plus one or more
    label columns whose names begin with ``label_``. The split is stratified on the
    label columns to preserve class distribution.

    Attributes
    ----------
    data_path : Path
        Path to the input CSV file.
    dataset_name : str
        Name of the dataset derived from the input file stem.
    data : pandas.DataFrame
        The full dataset loaded from ``data_path``.
    test_size : float
        Proportion of the dataset to include in the test split (default 0.2).
    random_state : int
        Random seed for reproducible splitting (default 42).

    """

    def __init__(
        self, data_path: Path, test_size: float = 0.2, random_state: int = 42
    ) -> None:
        """
        Initialise the data module with the path to a processed dataset.

        Parameters
        ----------
        data_path : Path
            Path to the processed CSV file containing features and label columns.
        test_size : float, optional
            Fraction of data to reserve for the test set (default 0.2).
        random_state : int, optional
            Seed for the random number generator (default 42).
        """
        self.data_path = data_path
        self.dataset_name = data_path.stem  # Extract dataset name from file stem
        try:
            self.data = pd.read_csv(data_path)
        except FileNotFoundError as e:
            logging.error(f"File not found: {data_path}")
            raise e

        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, train_output_path: Path, test_output_path: Path) -> None:
        """
        Perform a stratified train-test split and save the resulting datasets.

        Features and labels are first separated with :meth:`_separate_features_and_labels`.
        The data is then split using ``sklearn.train_test_split`` with stratification
        on the label columns. Finally, the train and test sets are saved via `_save_data`.
        """

        # split the data into features and labels
        # features, X_features are the inputs to the model later.
        # labels, Y_labels are what we want to predict using the model.
        X_features, Y_labels = self._seperate_features_and_labels()

        logging.info("Splitting data into training and testing sets.")
        # Perform train-test split. Stratification ensures even distiribution of labels.
        # Straification is based on Y_labels, meaning the test and train sets will have similar proportions of each label class.
        X_features_train, X_features_test, Y_labels_train, Y_labels_test = train_test_split(
            X_features, Y_labels, test_size=self.test_size, random_state=self.random_state, stratify=Y_labels
        )

        self._save_data(X_features_train, X_features_test, Y_labels_train, Y_labels_test, train_output_path, test_output_path)

    def _seperate_features_and_labels(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Separate feature columns from label columns.

        Label columns are identified as any column whose name starts with ``label_``.
        If no such column exists, a ``KeyError`` is raised.

        Returns
        -------
        X_features : pd.DataFrame
            DataFrame containing only the feature columns.
        Y_labels : pd.DataFrame
            DataFrame containing only the label columns.
        """
        logging.info("Separating features and labels.")

        # Loop over all columns to identify label columns
        label_col = [col for col in self.data.columns if col.startswith("label_")]
        if not label_col:
            raise KeyError(
                "No label columns found. Expected columns starting with 'label_'."
            )

        # Extract feature columns by dropping label columns
        X_features = self.data.copy()
        X_features = X_features.drop(columns=label_col)

        # Extract label columns, labels are what we want to predict using the model.
        Y_labels = self.data[label_col].copy()
        Y_labels = Y_labels.rename(columns={label_col[0]: "label"})

        logging.info(f"Features shape: {X_features.shape}, Labels shape: {Y_labels.shape}")
        return X_features, Y_labels

    def _save_data(
        self,
        X_features_train: pd.DataFrame,
        X_features_test: pd.DataFrame,
        Y_labels_train: pd.DataFrame,
        Y_labels_test: pd.DataFrame,
        train_output_path: Path,
        test_output_path: Path,
    ) -> None:
        """
        Concatenate features and labels for train/test sets and save them to CSV.

        The resulting files are named ``{dataset_name}_train_data.csv`` and
        ``{dataset_name}_test_data.csv`` and are written to the ``data/`` directory.

        Parameters
        ----------
        X_features_train, X_features_test : pd.DataFrame
            Feature matrices for training and testing.
        Y_labels_train, Y_labels_test : pd.DataFrame
            Label DataFrames for training and testing.
        """

        logging.info("Saving split datasets to CSV files.")

        train_df = pd.concat([X_features_train, Y_labels_train], axis=1)
        logging.info("Train data shape: %s", train_df.shape)
        test_df = pd.concat([X_features_test, Y_labels_test], axis=1)
        logging.info("Test data shape: %s", test_df.shape)

        train_path = Path(
            train_output_path
        ).resolve()
        test_path = Path(test_output_path).resolve()

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        logging.info("Train data saved to %s", train_path)
        logging.info("Test data saved to %s", test_path)
