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

from src.process_data import DataProcessor
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
            - paths: Dictionary of relative paths (raw, processed, outputs, etc.)

    Raises
    ------
        ValueError: If no dataset is specified or if an unsupported dataset is requested.
    """
    logging.info("Starting pipeline for dataset: %s", cfg.dataset)
    if cfg.dataset is None:
        raise ValueError("Dataset name must be specified in the configuration.")

    BASE_DIR = Path(__file__).parent.resolve()
    
    # ------------------------------------------------------------------
    # 1. Load & preprocess raw data (common to both datasets)
    # ------------------------------------------------------------------

    raw_data_path = Path(BASE_DIR / cfg.paths.raw)
    processor = DataProcessor(data_path=raw_data_path)
    processor.process_data()

    # ------------------------------------------------------------------
    # 2. Train / test split (common to both datasets)
    # ------------------------------------------------------------------
    
    processed_data_path = Path(BASE_DIR / cfg.paths.processed)
    splitter = DataSplitter(data_path=processed_data_path)
    splitter.split_data()

    # ------------------------------------------------------------------
    # 3. Dataset-specific analysis
    # ------------------------------------------------------------------
    
    if cfg.dataset == "dataset_1":
        train_data_path = Path(BASE_DIR / cfg.paths.train_split)
        test_data_path = Path(BASE_DIR / cfg.paths.test_split)
        output_path = Path(BASE_DIR / cfg.paths.outputs)

        output_path.mkdir(parents=True, exist_ok=True)

        rfecv = RFECVDataset1(
            train_data_path=train_data_path,
            test_data_path=test_data_path,
            output_path=output_path,
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
