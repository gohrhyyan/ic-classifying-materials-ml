import importlib  # Used for dynamically importing scikit-learn modules at runtime.
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve
from pathlib import Path
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class HandlerDataset2:
    """
    Dataset 2 processor: Compare classifiers, pick best, plot learning curve,
    find min size for 70% accuracy.
    This class encapsulates the entire workflow: data loading, model comparison via CV,
    selection of the best performer, learning curve generation to assess data efficiency,
    identification of minimal training size for a target accuracy, and visualization/saving results.
    """

    def __init__(self, train_path: Path, test_path: Path, output_path: Path, classifiers: list[dict], random_state: int = 42):
        logging.info(f"Initializing HandlerDataset2 with train: {train_path}, test: {test_path}, output: {output_path}")
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
        self.random_state = random_state
                
        self.X_train = self.train_df.drop("label", axis=1) # Extract features (X): Drop the 'label' column from training DataFrame. Drop doesn't affect the original DataFrame, only returns a new one without the specified column.
        self.y_train = self.train_df["label"]              # Extract labels (y): The 'label' column as a Series.
        self.X_test = self.test_df.drop("label", axis=1)
        self.y_test = self.test_df["label"]

        # Set up 5-fold stratified cross-validation: Splits all data into 5 "folds". Ensures each fold has roughly the same proportion of classes as the original dataset. Shuffles for randomness, random_state for reproducibility.
        # Each time we train a model, 4 folds are used for training and 1 for validation, rotating through all folds so each fold serves as validation once.
        # self.cv is a generator object that yields train/test indices for each fold.
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        # Create classifier instances from the config list - using importlib to load sklearn modules, allowing flexible config.
        self.models = {}  # Dictionary to hold {class_name: model_instance} pairs.
        for classifier in classifiers:
            model_name, class_name = classifier["type"].rsplit(".", 1)    # Split the type string into module and class names (e.g., "sklearn.linear_model.LogisticRegression").
            module = importlib.import_module(f"sklearn.{model_name}")     # Import the module dynamically, e.g., sklearn.linear_model, returning the module object.
            cls = getattr(module, class_name)                             # Get the class from the module (e.g., LogisticRegression), returning the class object.
            self.models[class_name] = cls(**classifier["params"])         # Instantiate the model instance with provided params and store in the models dict. 
        
        # Initialize attributes to be populated during analysis.
        self.best_model_name = None
        self.best_model = None
        self.mean_model_accuracy_results = {}
        self.min_size = None
        self.min_acc = None
        self.lc_data = None
        self.target = None

        logging.info(f"Initialization complete. Loaded {len(self.models)} classifiers.")

    def compare_classifiers(self):
        """
        Compare all classifiers using cross-validated accuracy on the full training set.
        Stores:
            self.mean_model_accuracy_results: dict of {model_name: mean_cv_accuracy}
            self.best_model_name: name of the best-performing model
            self.best_model: instance of the best-performing model
        """
        logging.info("Comparing classifiers using cross-validated accuracy...")
        # Prevent test set leakage in model selection by using only training data for cross-validation in selecting a best model.
        for name, model in self.models.items(): 
            mean_model_accuracy = cross_val_score(model, self.X_train, self.y_train, cv=self.cv, scoring="accuracy").mean()  # train and test each model n times (n = number of folds), cycling through, with each fold being used as the test set once. Compute mean accuracy across folds.
            self.mean_model_accuracy_results[name] = mean_model_accuracy                                                     # Store the mean accuracy for this model.
            logging.info(f"Model: {name}, Mean CV Accuracy: {mean_model_accuracy:.4f}")
        
        self.best_model_name = max(self.mean_model_accuracy_results, key=self.mean_model_accuracy_results.get)                     # Select the model with the highest mean CV accuracy.
        self.best_model = self.models[self.best_model_name]                                                                        # Retrieve the corresponding model instance.
        logging.info(f"Best model: {self.best_model_name} with Mean CV Accuracy: {self.mean_model_accuracy_results[self.best_model_name]:.4f}")

    def plot_classifier_comparison(self):
        """
        Plots a bar chart of all models + confusion matrices on the held-out test set.
        Outputs:
            classifier_comparison.png: Bar chart of mean CV accuracies 
            {model_name}_matrix.png: confusion matrices for each model on test set.       
        """
        logging.info("Plotting classifier comparison and confusion matrices...")
        # ---- Bar chart  ----
        fig_bar, ax_bar = plt.subplots(figsize=(3*len(self.models), 8))

        model_names = list(self.mean_model_accuracy_results.keys())                 # Get model names in order.
        model_accuracies = list(self.mean_model_accuracy_results.values())          # Get corresponding model accuracies.
        bars = ax_bar.bar(model_names, model_accuracies, color="steelblue")         # Create bars – index order matches model_names
        best_idx = model_names.index(self.best_model_name)                          # Find index of the best model.
        bars[best_idx].set_color("orange")                                          # Highlight the best model's bar in orange.

        ax_bar.set_ylim(0, 1.05)
        ax_bar.set_ylabel("Mean CV Accuracy")
        ax_bar.set_title("Classifier Comparison (5-fold CV)")

        for index, model_accuracy in enumerate(model_accuracies):                   # Annotate each bar with its accuracy value, 0.2 above the bar.
            ax_bar.text(index, model_accuracy + 0.02, f"{model_accuracy:.3f}",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")

        fig_bar.tight_layout()
        fig_bar.savefig(self.output_path / "classifier_comparison.png", dpi=200)
        plt.close(fig_bar)
        logging.info(f"Best model identified: {self.best_model_name}. Bar chart saved. Proceeding to confusion matrices.")

        # ---- Confusion matrices (one per model, trained on full train set) ----
        labels = np.unique(self.y_train)    # Get all unique class labels from training data for consistent ordering in confusion matrix.

        # output a confusion matrix for each model
        for name, model in self.models.items():         # loop through each model to plot its confusion matrix, using the full test set.
            model.fit(self.X_train, self.y_train)       # Train the model on the entire training set.
            y_pred = model.predict(self.X_test)         # Predict labels for the test set.
            test_acc = (y_pred == self.y_test).mean()   # Compute test accuracy as the proportion of correct predictions. Gives % because it's a mean of booleans. 

            # Compute confusion matrix =
            cm = confusion_matrix(self.y_test, y_pred, labels=labels)

            # Plot confusion matrix
            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            disp.plot(ax=ax_cm, cmap="Blues", values_format="d", colorbar=False)

            ax_cm.set_title(f"{name}\nTest Accuracy = {test_acc:.3f}", fontsize=13, pad=15)
            fig_cm.tight_layout()

            # Save individual high-res file
            safe_name = name.replace(" ", "_")
            fig_cm.savefig(self.output_path / f"ConfusionMatrix_{safe_name}.png",
                        dpi=300, bbox_inches='tight')
            plt.close(fig_cm)
            logging.info(f"Confusion matrix for {name} saved.")

    def compute_learning_curve(self, target: float = 0.70):
        """
        Generate learning curve for the best model and identify minimal training size
        achieving at least the target accuracy.
        Args:
            target (float): Target cross-validation accuracy threshold.
        """
        logging.info(f"Computing learning curve for {self.best_model_name} with target accuracy {target:.2f}...")
        self.target = target
        max_train_size = len(self.X_train) - (len(self.X_train) // self.cv.n_splits) # find the maximum training size for learning curve

        # preliminary runs have revealed that the target accuracy is achieved at around ~5 to 10 samples, 70% accuracy is achieved at minimum 4 samples, 
        # but as more data is added, the accuracy fluctuates significantly due to the small dataset size, and actually dips below 70% again,
        # we'll employ a finer granularity and pessimistic approach to identify the minimal size that consistently meets the target across all folds.
        # since it's not too computationally expensive, and the dataset is small, we'll use every integer training size from 1 to max_train_size (320)
        # to ensure that we're providing a reliable estimate of the minimal size needed to achieve the target accuracy,
        # we'll adopt the lowest accuracy across folds for each training size, ensuring that all folds meet the target for a given size.
        # furthermore, we'll also ensure that once the target is met at a certain size, all larger sizes also meet the target.

        # learning_curve + StratifiedKFold works as follows:
        #   - For EACH training size:
        #       - For EACH of the 5 folds:
        #            Take the FULL training indices of that fold (~80% of data)
        #            Randomly subsample 'size' samples FROM THAT FOLD'S TRAINING DATA ONLY
        #            Train model on those subsampled points
        #            Evaluate on the ENTIRE held-out validation fold (~20% of data)
        #   - This is repeated independently for all 5 folds, so 5 scores per size
        # returns: train_sizes_abs: 1D array of training sizes used, train_scores: 2D array (n_sizes x n_folds) of accuracy based on training data, cv_scores: 2D array (n_sizes x n_folds) of accuracy based on validation data. 
        train_sizes_abs, train_scores, cv_scores = learning_curve(
            self.best_model, self.X_train, self.y_train,
            cv=self.cv, train_sizes=np.arange(1, max_train_size+1, 1),
            scoring="accuracy", n_jobs=-1, random_state=self.random_state
        )

        # Identify minimal training size achieving at least the target accuracy.
        # Compute mean, min CV scores across folds for each training size.
        # For pessimistic estimate, we'll use the minimum accuracy across folds, ensuring all folds meet the target for a given size.
        cv_mins = np.min(cv_scores, axis=1)                      # Minimum CV accuracy across folds for each training size.
        cv_above_target = cv_mins >= self.target                 # returns boolean array: True if the lowest score among folds for that size >= 0.70
        self.lc_data = {
            "train_sizes_abs": train_sizes_abs,
            "train_means": np.mean(train_scores, axis=1),
            "cv_mins": cv_mins
        }
        # what's being done here:
        # We want to find the smallest training size such that from that size onward, all subsequent sizes also meet the target accuracy across all folds.
            # [::-1] reverses the array, looking from most data to least. i.e pass/fail resultss are being looked at from largest training size to smallest.
            # np.minimum.accumulate computes the cumulative minimum of the reversed boolean array. This means that once we hit a False (a size that fails), all smaller sizes will also be marked as False in the cumulative min.
            # [::-1] reverses the cumulative min array back to original order, so we can find the first index where all subsequent sizes meet the target.
            # np.argmax finds the index of the first occurrence of the maximum value (True) in the cumulative min array. This index corresponds to the smallest training size from which all larger sizes meet the target accuracy across all folds.
        if np.any(cv_above_target):
            # Target is achieved at least once → find the first stable point
            stable_from_idx = np.argmax(np.minimum.accumulate(cv_above_target[::-1])[::-1]) # Find the first index where all subsequent sizes meet the target accuracy across all folds.
            self.min_size = int(train_sizes_abs[stable_from_idx])
            self.min_acc = cv_mins[stable_from_idx]
            logging.info(f"Min stable size for >= {self.target:.0%}: {self.min_size} samples ({self.min_acc:.3f} acc)")
        else:
            # Never hits 70% even with all data
            self.min_size = None
            self.min_acc = cv_mins[-1]
            logging.warning(f"Target {self.target:.0%} NEVER reached! "
                            f"Best worst-fold accuracy: {self.min_acc:.3f} at full data.")

    def plot_learning_curve(self):
        """
        Plot learning curve of the best model with min-size annotation 
        Outputs:
            LearningCurve_{best_model_name}.png: Learning curve plot.
        """
        logging.info(f"Plotting learning curve for {self.best_model_name}...")
        if not hasattr(self, "lc_data") or self.lc_data is None:
            logging.warning("Learning curve data not computed yet. Run compute_learning_curve() first.")
            return

        data = self.lc_data
        train_sizes = data["train_sizes_abs"]
        train_means = data["train_means"]
        cv_mins = data["cv_mins"]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(train_sizes, cv_mins, "o-", color="darkorange", linewidth=2.5, label="CV Accuracy (worst fold)")

        ax.axhline(self.target, color="red", linestyle="--", label=f"Target {self.target:.0%}")

        if self.min_size is not None:
            ax.axvline(self.min_size, color="green", linestyle=":", linewidth=2,
                    label=f"Stable ≥{self.target:.0%}: {self.min_size} samples")
            ax.text(self.min_size + (train_sizes[-1] * 0.01), 0.75, f"{self.min_size}, {self.min_acc:.3f}",
                    color="green", fontweight="bold", fontsize=11,
                    bbox=dict(facecolor="white", edgecolor="green", alpha=0.8, pad=3))

        ax.set_xlabel("Training Size")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Learning Curve – {self.best_model_name}")
        ax.set_ylim(0, 1)
        ax.set_xlim(0, train_sizes[-1])
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        safe_name = self.best_model_name.replace(" ", "_")
        fig.savefig(self.output_path / f"LearningCurve_{safe_name}.png", dpi=250)
        plt.close(fig)

        logging.info(f"Learning curve saved for {self.best_model_name}.")