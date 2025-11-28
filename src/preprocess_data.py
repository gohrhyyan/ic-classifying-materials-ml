"""
Process a raw CSV dataset through cleaning, imputation, encoding, standardisation, and saving.

This module defines the ``DataPreprocessor`` class, which performs a complete
pre-processing pipeline on tabular data containing a ``label`` column.  The
pipeline imputes missing feature values with an iterative imputer backed by a
Bayesian Ridge regression model, one-hot encodes the label, standardises numeric
features, and writes the final dataset to CSV.  Fitted transformers are saved for reuse.
"""

import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).parent.parent

#TODO: can this logging be configured from outside, and made standard across modules?
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DataPreprocessor:
    """
    Handle the full data-pre-processing pipeline for a CSV file.

    The class loads a CSV file, preprocesses the target ``label`` column,
    imputes missing feature values, encodes the label, standardises numeric
    columns, and reassembles the dataset.

    Attributes
    ----------
    data : pandas.DataFrame
        The original dataset loaded from the input CSV.
    feature_data : pandas.DataFrame
        Imputed feature matrix (before standardisation).
    imputer : IterativeImputer
        Configured iterative imputer.
    scaler : StandardScaler
        Configured standard scaler.
    encoder : OneHotEncoder
        Configured one-hot encoder for the label.
    """

    def __init__(self, data_path: Path) -> None:
        """
        Initialise the preprocessor with a CSV file path and configure transformers.

        Parameters
        ----------
        data_path : Path
            Path to the input CSV file.  The file must contain a column named
            ``label`` that will be treated as the target variable.
        """
        self.base_name = data_path.with_suffix("")  # Remove file extension to get base path for outputs
        self.dataset_name = data_path.stem          # Extract dataset name from file stem
        self.data = pd.read_csv(data_path)          # Load dataset into pandas, automatically creating DataFrame

        #TODO: removie magic numbers
        # Use Bayesian Ridge regression model as base estimator for MICE (filling in missing values)
        self.imputer = IterativeImputer(
            estimator=BayesianRidge(),  # Robust regression for outlier resistance
            max_iter=100,               # More iterations for convergence (if needed)
            tol=1e-12,                  # Convergence tolerance
            initial_strategy="median",  # Start with simple median fill
            imputation_order="arabic",  # Preserve column order
            random_state=42,            # Ensure reproducibility
            verbose=0,
        )

        # StandardScaler is a preprocessing tool from scikit-learn that standardizes numeric features by transforming them to have:
        # Mean = 0
        # Standard deviation = 1 (unit variance)
        self.scaler = StandardScaler() 

        # OneHotEncoder to convert categorical labels into binary columns, let the encoder handle unknown categories safely
        self.encoder = OneHotEncoder(
            sparse_output=False,    # Return dense array
            handle_unknown="ignore",# Safe on unseen categories
            drop="first",           # Avoid multicollinearity
        )

    def preprocess_data(self, output_path: Path) -> None:
        """
        Execute the full data preprocessing pipeline.

        This method orchestrates the cleaning, imputation, encoding,
        standardisation, and saving of the preprocessed dataset.
        """
        features, labels = self._clean_data()
        scaled_features = self._scale_data(features)
        combined_data = self._combine_data(
            encoded_labels=labels, scaled_features=scaled_features
        )
        self._save_data(
            combined_data=combined_data,
            output_path=output_path,
        )

    def _clean_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Impute missing feature values and one-hot encode the target label.

        This method separates the ``label`` column from the dataset, fits and
        persists the OneHotEncoder on the label, fits and persists the
        IterativeImputer on the feature columns, and returns the imputed
        feature DataFrame and the encoded label DataFrame. Both returned
        DataFrames preserve the original row index.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            imputed_features: pd.DataFrame
            Imputed feature matrix with original column names and index.
            encoded_labels: pd.DataFrame
            One-hot encoded label columns with index matching the input data.
        """
        logging.info("Starting data cleaning preprocess...")
        # Preserve original order after transforms
        original_index = self.data.index.copy()

        if "label" not in self.data.columns:
            logging.error("Input data must contain a 'label' column.")
            raise KeyError("Input data must contain a 'label' column.")

        labels = self.data["label"].to_frame().copy()  # Keep as DataFrame for encoder
        features = self.data.drop(columns=["label"], axis=1).copy()

        logging.info("Encoding categorical variables...")
        encoded_labels = pd.DataFrame(
            self.encoder.fit_transform(labels),
            columns=self.encoder.get_feature_names_out(["label"]),
            index=original_index,
        )
        self._save_model(
            self.encoder, save_path=f"models/{self.dataset_name}_label_encoder.joblib"
        )
        logging.info("Encoding completed. Starting imputation...")

        feature_names = features.columns.tolist()
        # Fit imputer on full feature set
        feature_data = pd.DataFrame(
            self.imputer.fit_transform(features),
            columns=feature_names,
            index=original_index,
        )
        logging.info("Imputation completed.")

        self._save_model(
            self.imputer,
            save_path=f"models/{self.dataset_name}_iterative_imputer.joblib",
        )
        return feature_data, encoded_labels

    def _scale_data(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Standardise all numeric columns in the imputed feature matrix.

        Numeric columns are transformed in-place using the configured
        ``StandardScaler`` to have zero mean and unit variance, and
        persist the fitted scaler.
        """
        logging.info("Starting data standardisation preprocess...")
        # Select only numeric types to avoid scaling one-hot columns
        features = features.copy()
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        logging.debug("Numeric columns selected for scaling: %s", numeric_cols.tolist())
        features[numeric_cols] = self.scaler.fit_transform(features[numeric_cols])
        logging.info("Standardisation completed.")
        self._save_model(
            self.scaler, save_path=f"models/{self.dataset_name}_standard_scaler.joblib"
        )

        return features

    def _combine_data(
        self, encoded_labels: pd.DataFrame, scaled_features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        This method combines the imputed and standardised features with the one-hot
        encoded labels to return the full preprocessed dataset.

        Parameters
        ----------
        encoded_labels : pd.DataFrame
            One-hot encoded label columns returned by: ``clean_data``.
        scaled_features : pd.DataFrame
            Standardised feature matrix returned by: ``scale_data``.
        """
        logging.info("Combining encoded labels back to the main dataset...")
        # Axis=1 -> column-wise concatenation
        try:
            scaled_features = scaled_features.copy()
            encoded_labels = encoded_labels.copy()
            combined_data = pd.concat([scaled_features, encoded_labels], axis=1)
        except (pd.errors.MergeError, ValueError, TypeError) as e:
            logging.error("Error combining data: %s", e)
            raise

        logging.info("Combination completed.")
        return combined_data

    def _save_data(
        self,
        combined_data: pd.DataFrame,
        output_path: Path,
    ) -> None:
        """
        Saves the preprocessed dataset (features and labels) to a CSV
        file at the specified output path.

        Parameters
        ----------
        combined_data: pd.DataFrame
            Fully preprocessed dataset combining features and labels.
        output_path : Path
            Destination path for the processed CSV file.
        """
        output_file_path = BASE_DIR / output_path
        logging.info("Saving preprocessed data to %s...", output_file_path)
        combined_data.to_csv(output_file_path, index=False)
        logging.info("Data saved successfully.")

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
