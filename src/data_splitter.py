"""
Split a processed CSV dataset into train/test sets and save them separately.

This module defines the ``DataSplitter`` class, which loads a pre-processed CSV file
(containing one or more label columns whose names start with ``label_``), separates
features from labels, performs a stratified train-test split, and applies imputation
and standardisation to the feature sets. Finally, it combines the features and labels
and saves the resulting train and test sets as new CSV files in the ``data/`` directory.
"""

import logging
import numpy as np
import joblib
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

# Define BASE_DIR as the project's root: Go up two parent directories from this script's location
BASE_DIR = Path(__file__).resolve().parent.parent

class DataSplitter:
    """
    Load a processed dataset, split it into train/test sets,
    apply imputation and standardisation to the feature sets, and persist the splits.

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
    imputer : IterativeImputer
        Configured imputer for handling missing feature values.
    scaler : StandardScaler
        Configured scaler for standardising numeric features.
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

        # Use Bayesian Ridge regression model as base estimator for MICE (filling in missing values)
        self.imputer = IterativeImputer(
            estimator=BayesianRidge(),  # Robust regression for outlier resistance
            max_iter=100,  # More iterations for convergence (if needed)
            tol=1e-12,  # Convergence tolerance
            initial_strategy="median",  # Start with simple median fill
            imputation_order="arabic",  # Preserve column order
            random_state=42,  # Ensure reproducibility
            verbose=0,
        )

        # StandardScaler is a preprocessing tool from scikit-learn that standardises numeric features by transforming them to have:
        # Mean = 0
        # Standard deviation = 1 (unit variance)
        self.scaler = StandardScaler()

        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, train_output_path: Path, test_output_path: Path) -> None:
        """
        Perform a stratified train-test split, apply imputation and standardisation,
        and save the resulting datasets.

        Features and labels are first separated with :meth:`_separate_features_and_labels`.
        The data is then split using ``sklearn.train_test_split`` with stratification
        on the label columns. Imputation and standardisation are applied to the feature sets
        using the configured transformers.
        Finally, the train and test sets are saved via `_save_data`.
        """

        # split the data into features and labels
        # features, X_features are the inputs to the model later.
        # labels, Y_labels are what we want to predict using the model.
        X_features, Y_labels = self._separate_features_and_labels()

        logging.info("Splitting data into training and testing sets.")
        # Perform train-test split. Stratification ensures even distribution of labels.
        # Stratification is based on Y_labels, meaning the test and train sets will have similar proportions of each label class.
        X_features_train, X_features_test, Y_labels_train, Y_labels_test = (
            train_test_split(
                X_features,
                Y_labels,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=Y_labels,
            )
        )

        # Fit the imputer on the FULL training feature set and apply to both train/test
        logging.info(
            "Fitting IterativeImputer on training data and transforming train/test sets..."
        )

        # Fit only on training data
        X_train_imputed = pd.DataFrame(
            self.imputer.fit_transform(X_features_train),
            columns=X_features_train.columns,
            index=X_features_train.index,
        )

        # Transform test data using the fitted imputer (no fit!)
        X_test_imputed = pd.DataFrame(
            self.imputer.transform(X_features_test),
            columns=X_features_test.columns,
            index=X_features_test.index,
        )

        # Save the fitted imputer for future use
        self._save_model(
            self.imputer,
            save_path=f"models/{self.dataset_name}_iterative_imputer.joblib",
        )

        # Replace the original feature DataFrames
        X_features_train = X_train_imputed
        X_features_test = X_test_imputed

        logging.info("Imputation completed successfully.")

        logging.info("Standardising training and testing feature sets...")
        # Standardise the feature sets
        X_features_train = self._scale_data(X_features_train)
        X_features_test = self._transform_data(X_features_test)
        logging.info("Standardisation completed successfully.")

        self._save_data(
            X_features_train,
            X_features_test,
            Y_labels_train,
            Y_labels_test,
            train_output_path,
            test_output_path,
        )

    def _separate_features_and_labels(self) -> tuple[pd.DataFrame, pd.DataFrame]:
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

        logging.info(
            f"Features shape: {X_features.shape}, Labels shape: {Y_labels.shape}"
        )
        return X_features, Y_labels

    def _scale_data(self, train_data: pd.DataFrame) -> pd.DataFrame:
        """
        Standardise all numeric columns in the imputed feature matrix.

        Numeric columns are transformed in-place using the configured
        ``StandardScaler`` to have zero mean and unit variance, and
        persist the fitted scaler.
        """
        logging.info("Starting data standardisation preprocess...")
        # Select only numeric types to avoid scaling one-hot columns
        train_data = train_data.copy()
        numeric_cols = train_data.select_dtypes(include=[np.number]).columns
        logging.debug("Numeric columns selected for scaling: %s", numeric_cols.tolist())
        train_data[numeric_cols] = self.scaler.fit_transform(train_data[numeric_cols])
        logging.info("Standardisation completed.")
        self._save_model(
            self.scaler, save_path=f"models/{self.dataset_name}_standard_scaler.joblib"
        )

        return train_data

    def _transform_data(self, test_data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted scaler to standardise numeric columns in test data.

        This method is intended for use on validation or test datasets to ensure
        consistent scaling based on the training data's distribution.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix to be standardised.

        Returns
        -------
        pd.DataFrame
            Standardised feature matrix.
        """
        logging.info("Applying standardisation to new data...")
        test_data = test_data.copy()
        numeric_cols = test_data.select_dtypes(include=[np.number]).columns
        logging.debug("Numeric columns selected for scaling: %s", numeric_cols.tolist())
        test_data[numeric_cols] = self.scaler.transform(test_data[numeric_cols])
        logging.info("Standardisation applied.")
        return test_data

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

        train_path = Path(train_output_path).resolve()
        test_path = Path(test_output_path).resolve()

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        logging.info("Train data saved to %s", train_path)
        logging.info("Test data saved to %s", test_path)

    def _save_model(self, model: object, save_path: str) -> None:
        """
        Persist a fitted model as a .joblib file to the specified path.

        Parameters
        ----------
        model : object
            The fitted model to persist.
        save_path : str
            The file path where the model should be persisted.
        """
        try:
            logging.info("Saving model...")
            model_path = BASE_DIR / save_path
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path)
            logging.info("Model saved at %s.", model_path)
        except (FileNotFoundError, PermissionError, IOError) as e:
            logging.error("Failed to save model: %s", e)
            raise
