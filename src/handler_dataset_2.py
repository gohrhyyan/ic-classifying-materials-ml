import importlib  # Used for dynamically importing scikit-learn modules at runtime.
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve
from pathlib import Path
import matplotlib.pyplot as plt
import logging

#todo: add logging, robustness if results don't hit 70% at all.
#todo: plot confusion matrix for best model on test set.
#todo: plot comparison of all models' overall accuracies as bar chart.

class HandlerDataset2:
    """
    Dataset 2 processor: Compare classifiers, pick best, plot learning curve,
    find min size for 70% accuracy.
    This class encapsulates the entire workflow: data loading, model comparison via CV,
    selection of the best performer, learning curve generation to assess data efficiency,
    identification of minimal training size for a target accuracy, and visualization/saving results.
    """

    def __init__(self, train_path: Path, test_path: Path, output_path: Path, classifiers: list[dict], random_state: int = 42):
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
                
        self.X_train = self.train_df.drop("label", axis=1) #  Extract features (X): Drop the 'label' column from training DataFrame. Drop doesn't affect the original DataFrame, only returns a new one without the specified column.
        self.y_train = self.train_df["label"]              # Extract labels (y): The 'label' column as a Series.
        self.X_test = self.test_df.drop("label", axis=1)
        self.y_test = self.test_df["label"]

        # Set up 5-fold stratified cross-validation: Splits all data into 5 "folds". Ensures each fold has roughly the same proportion of classes as the original dataset. Shuffles for randomness, random_state for reproducibility.
        # Each time we train a model, 4 folds are used for training and 1 for validation, rotating through all folds so each fold serves as validation once.
        # self.cv is a generator object that yields train/test indices for each fold.
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        # Create classifier instances from the config list - using importlib to load sklearn modules, allowing flexible config.
        self.models = {}  # Dictionary to hold {cls_name: model_instance} pairs.
        for classifier in classifiers:
            mod_name, cls_name = classifier["type"].rsplit(".", 1)      # Split the type string into module and class names (e.g., "sklearn.linear_model.LogisticRegression").
            module = importlib.import_module(f"sklearn.{mod_name}")     # Import the module dynamically, e.g., sklearn.linear_model, returning the module object.
            cls = getattr(module, cls_name)                             # Get the class from the module (e.g., LogisticRegression), returning the class object.
            self.models[cls_name] = cls(**classifier["params"])         # Instantiate the model instance with provided params and store in the models dict. 

        

    def run_analysis(self, target: float = 0.70, sizes: int = 10):
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

        # Step 1: Compare classifiers using cross-validated accuracy.
        results = {}                                                                                                         # Dictionary to hold {model_name: mean_cv_accuracy}.
        
        for name, model in self.models.items(): 
            mean_model_accuracy = cross_val_score(model, self.X_train, self.y_train, cv=self.cv, scoring="accuracy").mean()  # train and test each model n times (n = number of folds), cycling through, with each fold being used as the test set once. Compute mean accuracy across folds.
            results[name] = mean_model_accuracy                                                                              # Store the mean accuracy for this model.

        # Step 2: Select the best-performing model based on mean CV accuracy.
        best_name = max(results, key=results.get)                                                                            # Select the model with the highest mean CV accuracy.
        best_model = self.models[best_name]                                                                                  # Retrieve the corresponding model instance.
    
        # Step 3: Generate learning curve for the best model.
        max_train_size = len(self.X_train) - (len(self.X_train) // self.cv.n_splits) # find the maximum training size for learning curve

        # preliminary runs have revealed that the target accuracy is achieved at around ~5 to 10 samples, 70% accuracy is achieved at minimum 4 samples, 
        # but as more data is added, the accuracy fluctuates significantly due to the small dataset size, and actually dips below 70% again,
        # we'll employ a finer granularity and pessimistic approach to identify the minimal size that consistently meets the target across all folds.
        # but for granularity and because the data is inherently noisy when limited, we'll use every integer training size from 1 to max_train_size (320), since the dataset is small enough.

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
            best_model, self.X_train, self.y_train,
            cv=self.cv, train_sizes=np.arange(1, max_train_size+1, 1),
            scoring="accuracy", n_jobs=-1, random_state=self.random_state
        )

        # Step 4: Identify minimal training size achieving at least the target accuracy.
        # Compute mean, min CV scores across folds for each training size.

        # For pessimistic estimate, we'll use the minimum accuracy across folds, ensuring all folds meet the target for a given size.
        cv_mins = np.min(cv_scores, axis=1)                      # Minimum CV accuracy across folds for each training size.
        cv_above_target = cv_mins >= target                      # returns boolean array: True if the lowest score among folds for that size >= 0.70

        # what's being done here:
        # We want to find the smallest training size such that from that size onward, all subsequent sizes also meet the target accuracy across all folds.
            # [::-1] reverses the array, looking from most data to least. i.e pass/fail resultss are being looked at from largest training size to smallest.
            # np.minimum.accumulate computes the cumulative minimum of the reversed boolean array. This means that once we hit a False (a size that fails), all smaller sizes will also be marked as False in the cumulative min.
            # [::-1] reverses the cumulative min array back to original order, so we can find the first index where all subsequent sizes meet the target.
            # np.argmax finds the index of the first occurrence of the maximum value (True) in the cumulative min array. This index corresponds to the smallest training size from which all larger sizes meet the target accuracy across all folds.
        stable_from_idx = np.argmax(np.minimum.accumulate((cv_above_target)[::-1])[::-1])  # Find the first index where all subsequent sizes meet the target accuracy across all folds.
        min_size = train_sizes_abs[stable_from_idx]
        min_acc = cv_mins[stable_from_idx]              
        
        # Print the minimal size result.
        print(f"Min size for >= {target}: {min_size} samples ({min_acc:.3f} acc)")

        # Create a new figure for the learning curve plot with specified size.
        plt.figure(figsize=(8, 5))
        plt.plot(train_sizes_abs, np.mean(train_scores, axis=1), "o-", label="Train")  #Plot training accuracy curve: Mean across folds, with markers and line.
        plt.plot(train_sizes_abs, cv_mins, "o-", label="CV", linewidth=2)             # Plot CV accuracy curve: Emphasized with thicker line.
        plt.axhline(target, color="r", linestyle="--", label="Target")             # Add horizontal dashed line for the target accuracy.
        plt.axvline(min_size, color="g", linestyle=":", label=f"Min: {min_size}")  # Add vertical dotted line at the minimal size.
        plt.xlabel("Training Size")
        plt.ylabel("Accuracy")
        plt.title(f"{best_name} Learning Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_path / f"{best_name}_curve.png", dpi=150)

        # Create a summary DataFrame for results.
        # Note: full_acc list comprehension assumes order matches self.models keys.
        results_df = pd.DataFrame({
            "full_acc": [results.get(k, 0) for k in self.models],  # CV accuracies for all models.
            "best_min_size": min_size,  # Single value repeated for each row (could be scalar, but DataFrame broadcasts).
            "best_min_acc": min_acc    # Similarly for min accuracy.
        }, index=list(self.models.keys()))  # Use model names as row index.
        results_df.to_csv(self.output_path / "results.csv") # Save the DataFrame to CSV in the output directory.
        print(f"\nSaved plot & results.csv to {self.output_path}") # Confirm saving in console.