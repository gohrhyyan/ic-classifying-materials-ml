# Import necessary libraries for dynamic module loading, numerical operations,
# data manipulation, scikit-learn model evaluation tools, file path handling,
# and plotting.
import importlib  # Used for dynamically importing scikit-learn modules at runtime.
import numpy as np  # NumPy for numerical computations, especially array operations in learning curves.
import pandas as pd  # Pandas for loading and handling CSV data as DataFrames.
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve
    # StratifiedKFold: Ensures balanced class distribution in cross-validation folds.
    # cross_val_score: Computes cross-validated scores for model evaluation.
    # learning_curve: Generates train/test sizes and scores for learning curve analysis.
from pathlib import Path  # Pathlib for cross-platform file path manipulation and resolution.
import matplotlib.pyplot as plt  # Matplotlib for creating and saving the learning curve plot.

# Configuration for classifiers: A list of dictionaries, each defining a classifier's
# name (for reference), type (sklearn module.class path), and hyperparameters.
# This allows easy extension or modification of models without changing core code.
CLASSIFIERS_CONFIG = [
    # Logistic Regression: A linear model for binary/multi-class classification.
    # Params: random_state for reproducibility, max_iter to prevent convergence warnings.
    {"name": "LR", "type": "linear_model.LogisticRegression", "params": {"random_state": 42, "max_iter": 200}},
    # Support Vector Classifier with RBF kernel: Non-linear boundary for complex data.
    # Params: RBF kernel for non-linearity, C=1.0 for regularization strength, random_state for reproducibility.
    {"name": "SVC", "type": "svm.SVC", "params": {"kernel": "rbf", "C": 1.0, "random_state": 42}},
    # Random Forest: Ensemble of decision trees for robust, low-variance predictions.
    # Params: 100 trees for ensemble size, random_state for reproducibility.
    {"name": "RF", "type": "ensemble.RandomForestClassifier", "params": {"n_estimators": 100, "random_state": 42}},
    # K-Nearest Neighbors: Instance-based learning using distance metrics.
    # Params: 5 nearest neighbors for local averaging.
    {"name": "KNN", "type": "neighbors.KNeighborsClassifier", "params": {"n_neighbors": 5}}
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
        # Load training data as a Pandas DataFrame from the specified CSV file.
        # Assumes the CSV has a 'label' column for targets and numeric features otherwise.
        self.train_df = pd.read_csv(train_path)
        # Load test data similarly.
        self.test_df = pd.read_csv(test_path)
        self.output_path = output_path
        
        # Extract features (X): Drop the 'label' column from training DataFrame.
        # Assumes all other columns are numeric features.
        self.X_train = self.train_df.drop("label", axis=1)
        # Extract labels (y): The 'label' column as a Series.
        self.y_train = self.train_df["label"]
        # Similarly for test set.
        self.X_test = self.test_df.drop("label", axis=1)
        self.y_test = self.test_df["label"]

        # Set up 5-fold stratified cross-validation: Ensures each fold has roughly
        # the same proportion of classes as the original dataset. Shuffle for randomness,
        # random_state for reproducibility.
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Dynamically create classifier instances from the config list.
        # This uses importlib to load sklearn modules on-the-fly, allowing flexible config.
        self.models = {}  # Dictionary to hold {name: model_instance} pairs.
        for config in CLASSIFIERS_CONFIG:
            # Extract the model name for keying the dictionary.
            name = config["name"]
            # Split the type string into module and class names (e.g., "sklearn.linear_model.LogisticRegression").
            mod_name, cls_name = config["type"].rsplit(".", 1)
            # Import the module dynamically (e.g., sklearn.linear_model).
            module = importlib.import_module(f"sklearn.{mod_name}")
            # Get the class from the module (e.g., LogisticRegression).
            cls = getattr(module, cls_name)
            # Instantiate the class with provided params and store in dict.
            self.models[name] = cls(**config["params"])

        # Print summary of loaded dataset sizes for user feedback.
        print(f"Loaded: {len(self.X_train)} train, {len(self.X_test)} test samples")

    def run(self, target=0.70, sizes=10):
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
        # Header for classifier comparison section in console output.
        print("\n--- Classifier Comparison (Full CV Acc) ---")
        results = {}  # Dictionary to store {model_name: mean_cv_accuracy} for comparison.
        # Iterate over each model in the dictionary.
        for name, model in self.models.items():
            # Compute 5-fold CV accuracy scores for the model on the full training set.
            # scoring="accuracy" uses classification accuracy as the metric.
            scores = cross_val_score(model, self.X_train, self.y_train, cv=self.cv, scoring="accuracy")
            # Calculate mean and standard deviation of the CV scores.
            mean, std = scores.mean(), scores.std()
            # Store the mean accuracy for later selection.
            results[name] = mean
            # Print formatted results for each model.
            print(f"{name}: {mean:.3f} ± {std:.3f}")

        # Select the model with the highest mean CV accuracy.
        best_name = max(results, key=results.get)
        # Retrieve the corresponding model instance.
        best_model = self.models[best_name]
        # Print the best model info.
        print(f"\nBest: {best_name} ({results[best_name]:.3f})")

        # Header for learning curve section.
        print(f"\n--- Learning Curve ({sizes} sizes) ---")
        # Generate learning curve data:
        # - train_sizes: Relative sizes from 0.1 to 1.0 in 'sizes' steps.
        # - train_scores: Training accuracy at each size (across CV folds).
        # - cv_scores: CV accuracy at each size (across CV folds).
        # n_jobs=-1 for parallel computation, random_state for reproducibility.
        train_sizes_rel, train_scores, cv_scores = learning_curve(
            best_model, self.X_train, self.y_train,
            cv=self.cv, train_sizes=np.linspace(0.1, 1.0, sizes),
            scoring="accuracy", n_jobs=-1, random_state=42
        )
        # Convert relative sizes to absolute sample counts.
        train_sizes = train_sizes_rel * len(self.X_train)
        # Compute mean CV accuracy across folds for each size (ignore std for simplicity here).
        cv_means = np.mean(cv_scores, axis=1)

        # Find the smallest index where mean CV accuracy meets or exceeds the target.
        # np.argmax returns the first True index in the boolean array (cv_means >= target).
        min_idx = np.argmax(cv_means >= target)
        # If no size meets the target, fall back to full dataset size.
        # Otherwise, round down to integer sample count.
        min_size = int(train_sizes[min_idx]) if min_idx > 0 else len(self.X_train)
        # Retrieve the accuracy at that index.
        min_acc = cv_means[min_idx]
        # Print the minimal size result.
        print(f"Min size for >= {target}: {min_size} samples ({min_acc:.3f} acc)")

        # Create a new figure for the learning curve plot with specified size.
        plt.figure(figsize=(8, 5))
        # Plot training accuracy curve: Mean across folds, with markers and line.
        plt.plot(train_sizes, np.mean(train_scores, axis=1), "o-", label="Train")
        # Plot CV accuracy curve: Emphasized with thicker line.
        plt.plot(train_sizes, cv_means, "o-", label="CV", linewidth=2)
        # Add horizontal dashed line for the target accuracy.
        plt.axhline(target, color="r", linestyle="--", label="Target")
        # Add vertical dotted line at the minimal size.
        plt.axvline(min_size, color="g", linestyle=":", label=f"Min: {min_size}")
        # Label x-axis as training set size.
        plt.xlabel("Training Size")
        # Label y-axis as accuracy.
        plt.ylabel("Accuracy")
        # Set plot title with the best model name.
        plt.title(f"{best_name} Learning Curve")
        # Add legend to distinguish lines.
        plt.legend()
        # Enable light grid for readability.
        plt.grid(True, alpha=0.3)
        # Save the plot as a high-resolution PNG in the output directory.
        plt.savefig(self.output_path / f"{best_name}_curve.png", dpi=150)
        # Display the plot interactively (if in a notebook/Jupyter; otherwise, may not show).
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
    proc.run(target=0.70, sizes=10)