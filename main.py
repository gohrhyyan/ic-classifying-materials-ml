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
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ------------------------------------------------------------------
    # 1. Load & preprocess raw data (common to both datasets)
    # ------------------------------------------------------------------
    DataPreprocessor(data_path=raw_data_path).preprocess_data(output_path=preprocessed_data_path)

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
        )
        rfecv.run_rfecv()
        rfecv.plot_accuracy_vs_features()
        # rfecv.plot_feature_importance()
        # rfecv.plot_confusion_matrix()
        
    elif cfg.dataset == "dataset_2":
        logging.info("Dataset 2 not yet implemented.") # Placeholder
        pass
    
    else:
        raise ValueError("Unsupported dataset: %s" % cfg.dataset)
    
    logging.info("Pipeline completed successfully for %s!", cfg.dataset)


if __name__ == "__main__":
    main()
