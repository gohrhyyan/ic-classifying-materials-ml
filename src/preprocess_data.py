"""
Process a raw CSV dataset through encoding, and saving.

This module defines the ``DataPreprocessor`` class, which performs a complete
pre-processing pipeline on tabular data containing a ``label`` column.  The
pipeline one-hot encodes the label, and writes the dataset to CSV.
Fitted transformers are saved for reuse.
"""

import logging
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent.parent


class DataPreprocessor:
    """
    Handle the full data-pre-processing pipeline for a CSV file.

    The class loads a CSV file, preprocesses the target ``label`` column,
    one-hot encodes the label, and reassembles the dataset.

    Attributes
    ----------
    data : pandas.DataFrame
        The original dataset loaded from the input CSV.
    feature_data : pandas.DataFrame
        feature matrix.
    encoder : OneHotEncoder
        Configured one-hot encoder for the label.
    """

    def __init__(self, data_path: Path, string_labels: list[str] | None = None) -> None:
        """
        Initialise the preprocessor with a CSV file path and configure transformers.

        Parameters
        ----------
        data_path : Path
            Path to the input CSV file.  The file must contain a column named
            ``label`` that will be treated as the target variable.
        string_labels : list[str] | None
            Optional list of string labels for the target variable. If provided,
            this list will be used to configure the OneHotEncoder categories.
        """
        # Remove file extension to get base path for outputs 
        self.base_name = data_path.with_suffix( 
            ""
        )  


        # Load dataset into pandas, automatically creating DataFrame
        self.dataset_name = data_path.stem  # Extract dataset name from file stem
        self.data = pd.read_csv(
            data_path
        )
 

        # OneHotEncoder to convert categorical labels into binary columns, let the encoder handle unknown categories safely
        self.encoder = OneHotEncoder(
            categories=[string_labels]
            if string_labels is not None
            else "auto",              # Handle known categories if applicable
            sparse_output=False,      # Return dense array
            handle_unknown="ignore",  # Safe on unseen categories
            drop="first",             # Avoid multicollinearity
        )

    def preprocess_data(self, output_path: Path) -> None:
        """
        Execute the full data preprocessing pipeline.

        This method orchestrates the encoding,
        and saving of the preprocessed dataset.
        """
        features, labels = self._encode_data()
        combined_data = self._combine_data(encoded_labels=labels, features=features)
        self._save_data(
            combined_data=combined_data,
            output_path=output_path,
        )

    def _encode_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        One-hot encodes the target label.

        This method separates the ``label`` column from the dataset, fits and
        persists the OneHotEncoder on the label and returns the feature
        DataFrame and the encoded label DataFrame. Both returned
        DataFrames preserve the original row index.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            features: pd.DataFrame
            Feature matrix with original column names and index.
            encoded_labels: pd.DataFrame
            One-hot encoded label columns with index matching the input data.
        """
        logging.info("Starting data cleaning preprocess...")

        # Preserve original order after transforms
        original_index = self.data.index.copy()
        if "label" not in self.data.columns:
            logging.error("Input data must contain a 'label' column.")
            raise KeyError("Input data must contain a 'label' column.")
            
        # Keep as DataFrame for encoder
        labels = self.data["label"].to_frame().copy()  
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
        logging.info("Encoding completed.")
        return features, encoded_labels

    def _combine_data(
        self, encoded_labels: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        This method combines the features with the one-hot
        encoded labels to return the full preprocessed dataset.

        Parameters
        ----------
        encoded_labels : pd.DataFrame
            One-hot encoded label columns returned by: ``clean_data``.
        features : pd.DataFrame
            Feature matrix returned by: ``_encode_data``.
        """
        logging.info("Combining encoded labels back to the main dataset...")
        # Axis=1 -> column-wise concatenation
        try:
            features = features.copy()
            encoded_labels = encoded_labels.copy()
            combined_data = pd.concat([features, encoded_labels], axis=1)
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
