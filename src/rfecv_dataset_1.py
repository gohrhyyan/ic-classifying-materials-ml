"""
rfecv_dataset_1.py

Perform Recursive Feature Elimination with Cross-Validation (RFECV) on Dataset 1.

This module provides the ``RFECVDataset1`` class, which encapsulates the complete
RFECV workflow for the pre-processed Dataset 1 conductivity classification task.
The class loads the training and test splits, fits an ``RFECV`` selector backed by a
``RandomForestClassifier``, identifies the optimal feature subset, persists the
fitted selector, and generates diagnostic visualisations (accuracy vs. number of
features, feature importances, and confusion matrix on the held-out test set).
"""
import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

BASE_DIR = Path(__file__).parent.parent


class RFECVDataset1:
    """
    Perform Recursive Feature Elimination with Cross-Validation (RFECV) on a dataset.

    This class loads a dataset from a CSV file, applies RFECV using a RandomForestClassifier,
    and saves the selected features to a new CSV file.

    Attributes
    ----------
    train_data_path : Path
        Path to the input CSV file.
    test_data_path : Path
        Path to the input CSV file.
    output_path : Path
        Path to save outputs to the output directory.
    n_splits : int
        Number of splits for StratifiedKFold cross-validation (default 5).
    random_state : int
        Random seed for reproducibility (default 42).
    n_estimators : int
        Number of trees in the RandomForestClassifier (default 500).
    """

    def __init__(
        self,
        train_data_path: Path,
        test_data_path: Path,
        output_path: Path,
        n_splits: int = 10,
        random_state: int = 42,
        n_estimators: int = 500,
    ) -> None:
        """
        Initialise the RFECV processor with paths and hyper-parameters.

        Parameters
        ----------
        train_data_path : Path
            Path to the training CSV file.
        test_data_path : Path
            Path to the test CSV file.
        output_path : Path
            Directory for saving plots and optional reduced datasets.
        n_splits : int, optional
            Number of cross-validation folds (default 10).
        random_state : int, optional
            Seed for random number generators (default 42).
        n_estimators : int, optional
            Number of trees in the Random Forest (default 500).
        """
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.output_path = output_path

        self.n_splits = n_splits
        self.random_state = random_state
        self.n_estimators = n_estimators

    def run_rfecv(self) -> None:
        """
        Execute RFECV on the training data and persist the fitted selector.

        Loads the training split, fits ``RFECV`` with a RandomForestClassifier,
        logs the selected features and their performance, and saves the full
        fitted ``RFECV`` object for later reuse (e.g. transforming new data).

        The method stores the fitted object in ``self.rfecv``.
        """
        logging.info("Loading training data from %s", self.train_data_path)
        train_data = pd.read_csv(self.train_data_path)
        train_data = train_data.copy()
        X_train = train_data.drop(columns=["label"])
        y_train = train_data["label"].to_numpy().ravel()  # Ensure y is 1D ndarray

        estimator = RandomForestClassifier(
            random_state=self.random_state, n_estimators=self.n_estimators, n_jobs=-1
        )

        cv = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )

        self.rfecv = RFECV(
            estimator=estimator,
            step=1,
            cv=cv,
            scoring="accuracy",
            min_features_to_select=1,
            n_jobs=-1,
            verbose=1,
        )

        logging.info("Fitting RFECV (this may take a while)...")
        self.rfecv.fit(X_train, y_train)
        logging.info("RFECV fitting complete.")

        logging.info(
            "Selected %d features out of %d", self.rfecv.n_features_, X_train.shape[1]
        )
        logging.info(
            "Features selected: %s", X_train.columns[self.rfecv.support_].tolist()
        )
        logging.info(
            "Accuracy with selected features: %.4f", self.rfecv.score(X_train, y_train)
        )

        logging.info("Saving model...")
        save_path = BASE_DIR / "models" / "rfecv_model.joblib"
        joblib.dump(self.rfecv, save_path, compress=3)
        logging.info("Model saved to %s", save_path)

    def plot_accuracy_vs_features(self) -> None:
        """
        Plot cross-validated accuracy against the number of selected features.

        Generates a line plot showing mean CV accuracy with ±1 standard deviation
        shading, a vertical line at the optimal number of features, and a horizontal
        baseline for performance using all features. The figure is saved as both
        PNG (300 dpi) and SVG in the output directory.
        """

        logging.info("Generating accuracy vs features plot...")

        n_features_range = range(1, len(self.rfecv.cv_results_["mean_test_score"]) + 1)
        mean_scores = self.rfecv.cv_results_["mean_test_score"]
        std_scores = self.rfecv.cv_results_["std_test_score"]

        plt.figure(figsize=(10, 6))
        plt.plot(
            n_features_range,
            mean_scores,
            marker="o",
            linewidth=2,
            label="Mean CV Accuracy",
        )
        plt.fill_between(
            n_features_range,
            mean_scores - std_scores,
            mean_scores + std_scores,
            alpha=0.2,
            label="±1 Std Dev",
        )
        plt.axvline(
            x=self.rfecv.n_features_,
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Optimal k={self.rfecv.n_features_}",
        )
        plt.axhline(
            y=mean_scores[-1],
            color="g",
            linestyle=":",
            linewidth=2,
            label=f"Baseline (all {len(mean_scores)} features): {mean_scores[-1]:.4f}",
        )

        plt.xlabel("Number of Features")
        plt.ylabel("CV Accuracy")
        plt.title("Accuracy vs Number of Features")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/rfecv_accuracy_plot.png")
        plt.savefig(f"{self.output_path}/rfecv_accuracy_plot.svg")
        logging.info(
            "Plot saved to %s/rfecv_accuracy_plot.png and .svg", self.output_path
        )

    def plot_feature_importance(self) -> None:
        """ 
        Plot feature importances of the Random Forest estimator for the selected features.

        Bar plot ordered by descending importance. Saved as PNG and SVG in the
        output directory.
        """

        logging.info("Generating feature importance plot...")

        importances = self.rfecv.estimator_.feature_importances_
        orig_feature_names = np.array(self.rfecv.feature_names_in_)
        feature_names = orig_feature_names[self.rfecv.support_]

        indicies = np.argsort(importances)
        sorted_importances = importances[indicies]
        sorted_feature_names = feature_names[indicies]


        plt.figure(figsize=(12,6))
        plt.bar(
            range(len(sorted_importances)),
            sorted_importances,
            align="center",
            color="b",
        )
        plt.xticks(
            range(len(sorted_importances)),
            sorted_feature_names.astype(str).tolist(),
            rotation=90,
        )
        
        plt.xlabel("Feature")
        plt.ylabel("Importance")
        plt.title("Feature Importances from RFECV-selected random forest")
        plt.tight_layout()

        plt.savefig(f"{self.output_path}/rfecv_feature_importance.png")
        plt.savefig(f"{self.output_path}/rfecv_feature_importance.svg")

        logging.info(
            "Feature importance plot saved to %s/rfecv_feature_importance.png", self.output_path
        )

    def plot_confusion_matrix(self) -> None:
        """ 
        Generate and save a confusion matrix on the held-out test set using optimal features.

        The matrix uses class labels 'Non-conductive' and 'Conductive'.
        Saved as PNG and SVG in the output directory.
        """        

        logging.info("Generating confusion matrix on test data...")

        test_data = pd.read_csv(self.test_data_path)
        test_data = test_data.copy()

        X_test = test_data.drop(columns=["label"])
        y_test = test_data["label"].to_numpy().ravel()

        predictions = self.rfecv.predict(X_test)

        cm = ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions,
            display_labels = ["Non-conductive", "Conductive"],
            cmap="plasma",
            normalize=None,
        )

        cm.plot()
        plt.tight_layout()

        # Save
        plt.savefig(f"{self.output_path}/rfecv_confusion_matrix.png")
        plt.savefig(f"{self.output_path}/rfecv_confusion_matrix.svg")
        logging.info(
            "Confusion matrix saved to %s/rfecv_confusion_matrix.png", 
            self.output_path
        )
