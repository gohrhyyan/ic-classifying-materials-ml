"""
Main entry point for the IC Computing 2025 material classification project.

Pipeline overview:
  • Common steps (both datasets):
      - Load and clean raw data
      - Train/test split
  • dataset_1 → RFECV + feature selection plots
  • dataset_2 → Binary search to find the smallest training set size that achieves ≥70% CV accuracy

Usage:
    >>> python main.py dataset=dataset_1
    >>> python main.py dataset=dataset_2
    >>> python main.py --multirun dataset=dataset_1,dataset_2
"""
from pathlib import Path
from omegaconf import DictConfig
import hydra
import logging

from src.preprocess_data import DataPreprocessor
from src.data_splitter import DataSplitter
from src.rfecv_dataset_1 import RFECVDataset1
from src.handler_dataset_2 import HandlerDataset2
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@hydra.main(version_base=None, config_path="./configs", config_name="main")
def main(cfg: DictConfig):
    """
    Execute the full material classification pipeline for the specified dataset.

    Arguments
    ---------
        cfg: Hydra configuration object containing:
            - dataset: Name of the dataset ("dataset_1" or "dataset_2")
            - paths: Dictionary of relative paths (raw, preprocessed, outputs, etc.)

    Raises
    ------
        ValueError: If no dataset is specified or if an unsupported dataset is requested.
    """
    logging.info("Starting pipeline for dataset: %s", cfg.dataset)
    if cfg.dataset is None:
        raise ValueError("Dataset name must be specified in the configuration.")

    base_dir = Path(__file__).parent.resolve()
    data_dir = base_dir / cfg.paths.data_dir
    raw_data_path          = data_dir / f"{cfg.dataset}{cfg.suffix.raw}"
    preprocessed_data_path = data_dir / f"{cfg.dataset}{cfg.suffix.preprocessed}"
    train_data_path        = data_dir / f"{cfg.dataset}{cfg.suffix.train}"
    test_data_path         = data_dir / f"{cfg.dataset}{cfg.suffix.test}"
    output_dir             = base_dir / cfg.paths.outputs
    string_labels = cfg.dataset_1_labels if cfg.dataset == "dataset_1" else None #imports conductive or non-conductive labels as a list[str] from config file
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configuration for classifiers: A list of dictionaries, each defining a classifier's type (sklearn module.class path), and hyperparameters.
    task_2_classifiers = [
        # Logistic Regression: A linear for binary/multi-class classification.
        # Params: random_state for reproducibility, max_iter to prevent convergence warnings.
        {"type": "linear_model.LogisticRegression", "params": {"random_state": cfg.random_seed, "max_iter": 200}},
        # Support Vector Classifier with RBF kernel: Non-linear boundary for complex data.
        # Params: RBF kernel for non-linearity, C=1.0 for regularization strength, random_state for reproducibility.
        {"type": "svm.SVC", "params": {"kernel": "rbf", "C": 1.0, "random_state": cfg.random_seed}},
        # Random Forest: Ensemble of decision trees for robust, low-variance predictions.
        # Params: 100 trees for ensemble size, random_state for reproducibility.
        {"type": "ensemble.RandomForestClassifier", "params": {"n_estimators": 100, "random_state": cfg.random_seed}},
        # K-Nearest Neighbors: Instance-based learning using distance metrics.
        # Params: 5 nearest neighbors for local averaging.
        {"type": "neighbors.KNeighborsClassifier", "params": {"n_neighbors": 5}}
    ]
    
    # ------------------------------------------------------------------
    # 1. Load & preprocess raw data (common to both datasets)
    # ------------------------------------------------------------------
    DataPreprocessor(data_path=raw_data_path, string_labels=string_labels).preprocess_data(output_path=preprocessed_data_path)

    # ------------------------------------------------------------------
    # 2. Train / test split (common to both datasets)
    # ------------------------------------------------------------------
    DataSplitter(data_path=preprocessed_data_path).split_data(
        train_output_path=train_data_path,
        test_output_path=test_data_path,
    )

    # ------------------------------------------------------------------
    # 3. Dataset-specific analysis
    # ------------------------------------------------------------------
    if cfg.dataset == "dataset_1":
        rfecv = RFECVDataset1(
            train_data_path=train_data_path,
            test_data_path=test_data_path,
            output_path=output_dir,
            random_state=cfg.random_seed
        )
        rfecv.run_rfecv()
        rfecv.plot_accuracy_vs_features()
        rfecv.plot_feature_importance()
        rfecv.plot_confusion_matrix()
        
    elif cfg.dataset == "dataset_2":

        handler = HandlerDataset2(
            train_path=train_data_path,
            test_path=test_data_path,
            output_path=output_dir,
            classifiers=task_2_classifiers,
            random_state=cfg.random_seed
        )
        handler.run_analysis(target=0.70, sizes=10)
        
    else:
        raise ValueError("Unsupported dataset: %s" % cfg.dataset)
    
    logging.info("Pipeline completed successfully for %s!", cfg.dataset)


if __name__ == "__main__":
    main()
