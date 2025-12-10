import importlib  # Used for dynamically importing scikit-learn modules at runtime.
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve
from pathlib import Path
import matplotlib.pyplot as plt

# Configuration for classifiers: A list of dictionaries, each defining a classifier's type (sklearn module.class path), and hyperparameters.
# This allows easy extension or modification of models without changing core code.
CLASSIFIERS_CONFIG = [
    # Logistic Regression: A linear for binary/multi-class classification.
    # Params: random_state for reproducibility, max_iter to prevent convergence warnings.
    {"type": "linear_model.LogisticRegression", "params": {"random_state": 42, "max_iter": 200}},
    # Support Vector Classifier with RBF kernel: Non-linear boundary for complex data.
    # Params: RBF kernel for non-linearity, C=1.0 for regularization strength, random_state for reproducibility.
    {"type": "svm.SVC", "params": {"kernel": "rbf", "C": 1.0, "random_state": 42}},
    # Random Forest: Ensemble of decision trees for robust, low-variance predictions.
    # Params: 100 trees for ensemble size, random_state for reproducibility.
    {"type": "ensemble.RandomForestClassifier", "params": {"n_estimators": 100, "random_state": 42}},
    # K-Nearest Neighbors: Instance-based learning using distance metrics.
    # Params: 5 nearest neighbors for local averaging.
    {"type": "neighbors.KNeighborsClassifier", "params": {"n_neighbors": 5}}
]

class Dataset2Processor:
    """
    Dataset 2 processor: Compare classifiers, pick best, plot learning curve,
    find min size for 70% acc. Readable prints & basic plot.
    This class encapsulates the entire workflow: data loading, model comparison via CV,
    selection of the best performer, learning curve generation to assess data efficiency,
    identification of minimal training size for a target accuracy, and visualization/saving results.
    """

    def __init__(self, train_path: Path, test_path: Path, output_path: Path):
        """
        Initialize the processor by loading training and test data from CSV files,
        separating features (X) and labels (y), setting up cross-validation splitter,
        and instantiating classifiers from the config.
        
        Args:
            train_path (Path): Path to the processed training CSV file.
            test_path (Path): Path to the processed test CSV file.
            output_path (Path): Directory path for saving plots and results.
        """
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        self.output_path = output_path
        
        self.X_train = self.train_df.drop("label", axis=1) #  Extract features (X): Drop the 'label' column from training DataFrame. Drop doesn't affect the original DataFrame, only returns a new one without the specified column.
        self.y_train = self.train_df["label"]              # Extract labels (y): The 'label' column as a Series.
        self.X_test = self.test_df.drop("label", axis=1)
        self.y_test = self.test_df["label"]

        # Set up 5-fold stratified cross-validation: Ensures each fold has roughly the same proportion of classes as the original dataset. Shuffle for randomness, random_state for reproducibility.
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Create classifier instances from the config list - using importlib to load sklearn modules, allowing flexible config.
        self.models = {}  # Dictionary to hold {cls_name: model_instance} pairs.
        for config in CLASSIFIERS_CONFIG:
            # Split the type string into module and class names (e.g., "sklearn.linear_model.LogisticRegression").
            mod_name, cls_name = config["type"].rsplit(".", 1)
            # Import the module dynamically (e.g., sklearn.linear_model).
            module = importlib.import_module(f"sklearn.{mod_name}")
            # Get the class from the module (e.g., LogisticRegression).
            cls = getattr(module, cls_name)
            # Instantiate the class with provided params and store in dict.
            self.models[cls_name] = cls(**config["params"])

        

    def run(self, target=0.70, sizes: int = 10):
        """
        Execute the full analysis pipeline:
        1. Compare all classifiers using cross-validated accuracy on full training set.
        2. Select the best-performing model.
        3. Generate and plot a learning curve for the best model across subsampled training sizes.
        4. Identify the minimal training size achieving at least the target accuracy.
        5. Save the plot and a results summary CSV.
        
        Args:
            target (float): Target cross-validation accuracy threshold (default 0.70).
            sizes (int): Number of points in the learning curve (default 10, evenly spaced from 10% to 100%).
        """
        results = {}  # Dictionary to store {model_name: mean_cv_accuracy} for comparison.
        for name, model in self.models.items():
            scores = cross_val_score(model, self.X_train, self.y_train, cv=self.cv, scoring="accuracy")  # Compute 5-fold CV accuracy scores for the model on the full training set. scoring="accuracy" uses classification accuracy as the metric.
            mean, std = scores.mean(), scores.std()                                                      # Calculate mean and standard deviation of the CV scores.
            results[name] = mean
            print(f"{name}: {mean:.3f} ± {std:.3f}")

        best_name = max(results, key=results.get)                                                        # Select the model with the highest mean CV accuracy.
        best_model = self.models[best_name]                                                              # Retrieve the corresponding model instance.
        print(f"\nBest: {best_name} ({results[best_name]:.3f})")

        print(f"\n--- Learning Curve ---")
        # Generate learning curve data:
        # - Numbers of training examples that has been used to generate the learning curve.
        # - train_scores: Training accuracy at each size (across CV folds).
        # - cv_scores: CV accuracy at each size (across CV folds).
        # - n_jobs=-1: use all cores for for parallel computation
        # - random_state for reproducibility.
        max_train_size = len(self.X_train) - (len(self.X_train) // self.cv.n_splits)
       
        train_sizes_abs, train_scores, cv_scores = learning_curve(
            best_model, self.X_train, self.y_train,
            cv=self.cv, train_sizes=np.arange(1, max_train_size+1, 1),
            scoring="accuracy", n_jobs=-1, random_state=42
        )

        # Compute mean CV accuracy across folds for each size (ignore std for simplicity here).
        cv_means = np.mean(cv_scores, axis=1)

        # Find the smallest index where mean CV accuracy meets or exceeds the target.
        # np.argmax returns the first True index in the boolean array (cv_means >= target).
        min_idx = np.argmax(cv_means >= target)
        # If no size meets the target, fall back to full dataset size.
        # Otherwise, round down to integer sample count.
        min_size = int(train_sizes_abs[min_idx]) if min_idx > 0 else len(self.X_train)
        # Retrieve the accuracy at that index.
        min_acc = cv_means[min_idx]import importlib  # Used for dynamically importing scikit-learn modules at runtime.
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve
from pathlib import Path
import matplotlib.pyplot as plt

# Configuration for classifiers: A list of dictionaries, each defining a classifier's type (sklearn module.class path), and hyperparameters.
# This allows easy extension or modification of models without changing core code.
CLASSIFIERS_CONFIG = [
    # Logistic Regression: A linear for binary/multi-class classification.
    # Params: random_state for reproducibility, max_iter to prevent convergence warnings.
    {"type": "linear_model.LogisticRegression", "params": {"random_state": 42, "max_iter": 200}},
    # Support Vector Classifier with RBF kernel: Non-linear boundary for complex data.
    # Params: RBF kernel for non-linearity, C=1.0 for regularization strength, random_state for reproducibility.
    {"type": "svm.SVC", "params": {"kernel": "rbf", "C": 1.0, "random_state": 42}},
    # Random Forest: Ensemble of decision trees for robust, low-variance predictions.
    # Params: 100 trees for ensemble size, random_state for reproducibility.
    {"type": "ensemble.RandomForestClassifier", "params": {"n_estimators": 100, "random_state": 42}},
    # K-Nearest Neighbors: Instance-based learning using distance metrics.
    # Params: 5 nearest neighbors for local averaging.
    {"type": "neighbors.KNeighborsClassifier", "params": {"n_neighbors": 5}}
]

class Dataset2Processor:
    """
    Dataset 2 processor: Compare classifiers, pick best, plot learning curve,
    find min size for 70% acc. Readable prints & basic plot.
    This class encapsulates the entire workflow: data loading, model comparison via CV,
    selection of the best performer, learning curve generation to assess data efficiency,
    identification of minimal training size for a target accuracy, and visualization/saving results.
    """

    def __init__(self, train_path: Path, test_path: Path, output_path: Path):
        """
        Initialize the processor by loading training and test data from CSV files,
        separating features (X) and labels (y), setting up cross-validation splitter,
        and instantiating classifiers from the config.
        
        Args:
            train_path (Path): Path to the processed training CSV file.
            test_path (Path): Path to the processed test CSV file.
            output_path (Path): Directory path for saving plots and results.
        """
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        self.output_path = output_path
        
        self.X_train = self.train_df.drop("label", axis=1) #  Extract features (X): Drop the 'label' column from training DataFrame. Drop doesn't affect the original DataFrame, only returns a new one without the specified column.
        self.y_train = self.train_df["label"]              # Extract labels (y): The 'label' column as a Series.
        self.X_test = self.test_df.drop("label", axis=1)
        self.y_test = self.test_df["label"]

        # Set up 5-fold stratified cross-validation: Ensures each fold has roughly the same proportion of classes as the original dataset. Shuffle for randomness, random_state for reproducibility.
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Create classifier instances from the config list - using importlib to load sklearn modules, allowing flexible config.
        self.models = {}  # Dictionary to hold {cls_name: model_instance} pairs.
        for config in CLASSIFIERS_CONFIG:
            # Split the type string into module and class names (e.g., "sklearn.linear_model.LogisticRegression").
            mod_name, cls_name = config["type"].rsplit(".", 1)
            # Import the module dynamically (e.g., sklearn.linear_model).
            module = importlib.import_module(f"sklearn.{mod_name}")
            # Get the class from the module (e.g., LogisticRegression).
            cls = getattr(module, cls_name)
            # Instantiate the class with provided params and store in dict.
            self.models[cls_name] = cls(**config["params"])

        

    def run(self, target=0.70, sizes: int = 10):
        """
        Execute the full analysis pipeline:
        1. Compare all classifiers using cross-validated accuracy on full training set.
        2. Select the best-performing model.
        3. Generate and plot a learning curve for the best model across subsampled training sizes.
        4. Identify the minimal training size achieving at least the target accuracy.
        5. Save the plot and a results summary CSV.
        
        Args:
            target (float): Target cross-validation accuracy threshold (default 0.70).
            sizes (int): Number of points in the learning curve (default 10, evenly spaced from 10% to 100%).
        """
        results = {}  # Dictionary to store {model_name: mean_cv_accuracy} for comparison.
        for name, model in self.models.items():
            scores = cross_val_score(model, self.X_train, self.y_train, cv=self.cv, scoring="accuracy")  # Compute 5-fold CV accuracy scores for the model on the full training set. scoring="accuracy" uses classification accuracy as the metric.
            mean, std = scores.mean(), scores.std()                                                      # Calculate mean and standard deviation of the CV scores.
            results[name] = mean
            print(f"{name}: {mean:.3f} ± {std:.3f}")

        best_name = max(results, key=results.get)                                                        # Select the model with the highest mean CV accuracy.
        best_model = self.models[best_name]                                                              # Retrieve the corresponding model instance.
        print(f"\nBest: {best_name} ({results[best_name]:.3f})")

        print(f"\n--- Learning Curve ---")
        # Generate learning curve data:
        # - Numbers of training examples that has been used to generate the learning curve.
        # - train_scores: Training accuracy at each size (across CV folds).
        # - cv_scores: CV accuracy at each size (across CV folds).
        # - n_jobs=-1: use all cores for for parallel computation
        # - random_state for reproducibility.
        max_train_size = len(self.X_train) - (len(self.X_train) // self.cv.n_splits)
       
        train_sizes_abs, train_scores, cv_scores = learning_curve(
            best_model, self.X_train, self.y_train,
            cv=self.cv, train_sizes=np.arange(1, max_train_size+1, 1),
            scoring="accuracy", n_jobs=-1, random_state=42
        )

        # Compute mean CV accuracy across folds for each size (ignore std for simplicity here).
        cv_means = np.mean(cv_scores, axis=1)

        # Find the smallest index where mean CV accuracy meets or exceeds the target.
        # np.argmax returns the first True index in the boolean array (cv_means >= target).
        min_idx = np.argmax(cv_means >= target)
        # If no size meets the target, fall back to full dataset size.
        # Otherwise, round down to integer sample count.
        min_size = int(train_sizes_abs[min_idx]) if min_idx > 0 else len(self.X_train)
        # Retrieve the accuracy at that index.
        min_acc = cv_means[min_idx]
        # Print the minimal size result.
        print(f"Min size for >= {target}: {min_size} samples ({min_acc:.3f} acc)")

        # Create a new figure for the learning curve plot with specified size.
        plt.figure(figsize=(8, 5))
        plt.plot(train_sizes_abs, np.mean(train_scores, axis=1), "o-", label="Train")  #Plot training accuracy curve: Mean across folds, with markers and line.
        plt.plot(train_sizes_abs, cv_means, "o-", label="CV", linewidth=2)             # Plot CV accuracy curve: Emphasized with thicker line.
        plt.axhline(target, color="r", linestyle="--", label="Target")             # Add horizontal dashed line for the target accuracy.
        plt.axvline(min_size, color="g", linestyle=":", label=f"Min: {min_size}")  # Add vertical dotted line at the minimal size.
        plt.xlabel("Training Size")
        plt.ylabel("Accuracy")
        plt.title(f"{best_name} Learning Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_path / f"{best_name}_curve.png", dpi=150)
        plt.show()

        # Create a summary DataFrame for results.
        # Note: full_acc list comprehension assumes order matches self.models keys.
        results_df = pd.DataFrame({
            "full_acc": [results.get(k, 0) for k in self.models],  # CV accuracies for all models.
            "best_min_size": min_size,  # Single value repeated for each row (could be scalar, but DataFrame broadcasts).
            "best_min_acc": min_acc    # Similarly for min accuracy.
        }, index=list(self.models.keys()))  # Use model names as row index.
        # Save the DataFrame to CSV in the output directory.
        results_df.to_csv(self.output_path / "results.csv")
        # Confirm saving in console.
        print(f"\nSaved plot & results.csv to {self.output_path}")

# Entry point for standalone script execution.
# This block runs the processor if the script is executed directly (not imported).
if __name__ == "__main__":
    # Import Path if not already (redundant here but ensures availability).
    from pathlib import Path
    # Resolve base directory: Parent of the parent of this script file (assuming structure like project/src/script.py).
    base = Path(__file__).parent.parent
    # Instantiate the processor with specific file paths:
    # - Train/test CSVs in a 'data' subfolder.
    # - Outputs in an 'outputs/dataset_2' subfolder.
    proc = Dataset2Processor(
        base / "data" / "dataset_2_preprocessed_train.csv",
        base / "data" / "dataset_2_preprocessed_test.csv",
        base / "outputs" / "dataset_2"
    )
    # Run the analysis with default target accuracy of 70% and 10 curve points.
    proc.run(target=0.70)
        # Print the minimal size result.
        print(f"Min size for >= {target}: {min_size} samples ({min_acc:.3f} acc)")

        # Create a new figure for the learning curve plot with specified size.
        plt.figure(figsize=(8, 5))
        plt.plot(train_sizes_abs, np.mean(train_scores, axis=1), "o-", label="Train")  #Plot training accuracy curve: Mean across folds, with markers and line.
        plt.plot(train_sizes_abs, cv_means, "o-", label="CV", linewidth=2)             # Plot CV accuracy curve: Emphasized with thicker line.
        plt.axhline(target, color="r", linestyle="--", label="Target")             # Add horizontal dashed line for the target accuracy.
        plt.axvline(min_size, color="g", linestyle=":", label=f"Min: {min_size}")  # Add vertical dotted line at the minimal size.
        plt.xlabel("Training Size")
        plt.ylabel("Accuracy")
        plt.title(f"{best_name} Learning Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_path / f"{best_name}_curve.png", dpi=150)
        plt.show()

        # Create a summary DataFrame for results.
        # Note: full_acc list comprehension assumes order matches self.models keys.
        results_df = pd.DataFrame({
            "full_acc": [results.get(k, 0) for k in self.models],  # CV accuracies for all models.
            "best_min_size": min_size,  # Single value repeated for each row (could be scalar, but DataFrame broadcasts).
            "best_min_acc": min_acc    # Similarly for min accuracy.
        }, index=list(self.models.keys()))  # Use model names as row index.
        # Save the DataFrame to CSV in the output directory.
        results_df.to_csv(self.output_path / "results.csv")
        # Confirm saving in console.
        print(f"\nSaved plot & results.csv to {self.output_path}")

# Entry point for standalone script execution.
# This block runs the processor if the script is executed directly (not imported).
if __name__ == "__main__":
    # Import Path if not already (redundant here but ensures availability).
    from pathlib import Path
    # Resolve base directory: Parent of the parent of this script file (assuming structure like project/src/script.py).
    base = Path(__file__).parent.parent
    # Instantiate the processor with specific file paths:
    # - Train/test CSVs in a 'data' subfolder.
    # - Outputs in an 'outputs/dataset_2' subfolder.
    proc = Dataset2Processor(
        base / "data" / "dataset_2_preprocessed_train.csv",
        base / "data" / "dataset_2_preprocessed_test.csv",
        base / "outputs" / "dataset_2"
    )
    # Run the analysis with default target accuracy of 70% and 10 curve points.
    proc.run(target=0.70)