"""
Machine Learning for Smarter Crop Planning
COMP 3009 Final Project - Part 3: Scaled Implementation

This file implements Section 5+: Scaling to larger datasets with comprehensive
diagnostics and model evaluation.

Features:
- Multi-crop classification (configurable)
- Cross-validation for model stability
- Comprehensive diagnostic checks
- Feature distribution analysis
- Detailed performance metrics

Sections covered: 5+
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score


# ============================================================================
# CONFIGURATION
# ============================================================================


class Config:
    """Configuration parameters for the scaled implementation."""

    # Data source
    FILENAME = "data/1400row_crop_dataset.csv"

    # Crop selection: specify rows per crop
    # Adjust to test different scenarios
    CROP_ROWS = {"rice": 50, "maize": 40, "chickpea": 35}

    # Train/test split
    TRAIN_SPLIT = 0.70
    RANDOM_SEED = 42

    # Output options
    SHOW_DETAILED_RESULTS = True
    SHOW_FEATURE_DISTRIBUTIONS = True
    RUN_SANITY_CHECKS = True

    # Feature columns
    FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "pH", "rainfall"]


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================


def load_and_prepare_data(config):
    """
    Loads dataset and prepares train/test splits for specified crops.

    Args:
        config: Config object with parameters

    Returns:
        Tuple of (train_df, test_df, full_df)
    """
    # Load data
    df = pd.read_csv(config.FILENAME)
    print(f"* Loaded {len(df)} rows from {config.FILENAME}")
    print(f"* Available crops: {', '.join(df['label'].unique())}\n")

    # Build train/test datasets
    train_dataset = pd.DataFrame()
    test_dataset = pd.DataFrame()

    print("Per-crop data splits:")
    print("-" * 60)

    for crop, total_rows in config.CROP_ROWS.items():
        crop_rows = df[df["label"] == crop]
        available = len(crop_rows)

        # Sample specified number of rows
        if len(crop_rows) >= total_rows:
            crop_data = crop_rows.sample(n=total_rows, random_state=config.RANDOM_SEED)
        else:
            crop_data = crop_rows.sample(frac=1, random_state=config.RANDOM_SEED)

        # Split into train/test
        crop_train, crop_test = train_test_split(
            crop_data,
            train_size=config.TRAIN_SPLIT,
            shuffle=True,
            random_state=config.RANDOM_SEED,
        )

        train_dataset = pd.concat([train_dataset, crop_train], ignore_index=True)
        test_dataset = pd.concat([test_dataset, crop_test], ignore_index=True)

        print(
            f"{crop:15} | available: {available:4} | sampled: {len(crop_data):3} | "
            f"train: {len(crop_train):3} | test: {len(crop_test):3}"
        )

    print("-" * 60)
    print(
        f"{'TOTAL':15} | train: {len(train_dataset):3} | test: {len(test_dataset):3}\n"
    )

    return train_dataset, test_dataset, df


# ============================================================================
# MODEL TRAINING & EVALUATION
# ============================================================================


def train_model_with_cv(X_train, y_train, cv_folds=5):
    """
    Trains Gaussian Naive Bayes with cross-validation.

    Args:
        X_train: Training features
        y_train: Training labels
        cv_folds: Number of cross-validation folds

    Returns:
        Tuple of (trained_model, cv_scores)
    """
    model = GaussianNB()

    # Cross-validation
    print(f"Running {cv_folds}-fold cross-validation...")
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=cv_folds, scoring="accuracy"
    )
    print(f"Cross-validation scores: {[f'{s:.3f}' for s in cv_scores]}")
    print(f"Mean CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}\n")

    # Train on full training set
    model.fit(X_train, y_train)

    return model, cv_scores


def evaluate_model(model, X_test, y_test, encoder):
    """
    Evaluates model performance on test set.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels (encoded)
        encoder: LabelEncoder for decoding

    Returns:
        Tuple of (predictions, accuracy)
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    correct = (y_test == y_pred).sum()

    print("=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    print(f"Correct Predictions: {correct}/{len(y_test)}\n")

    # Per-crop accuracy
    print("Per-Crop Performance:")
    print("-" * 60)

    y_test_decoded = encoder.inverse_transform(y_test)
    y_pred_decoded = encoder.inverse_transform(y_pred)

    for crop in encoder.classes_:
        mask = y_test_decoded == crop
        if mask.sum() > 0:
            crop_acc = accuracy_score(y_test[mask], y_pred[mask])
            crop_correct = (y_test[mask] == y_pred[mask]).sum()
            crop_total = mask.sum()
            print(
                f"{crop:15}: {crop_correct:2}/{crop_total:2} correct ({crop_acc * 100:5.1f}%)"
            )

    return y_pred, accuracy


# ============================================================================
# DETAILED RESULTS & VISUALIZATION
# ============================================================================


def show_sample_predictions(X_test, y_test, y_pred, encoder, num_samples=15):
    """
    Displays detailed prediction results for sample test cases.

    Args:
        X_test: Test features
        y_test: True labels (encoded)
        y_pred: Predicted labels (encoded)
        encoder: LabelEncoder
        num_samples: Number of samples to display
    """
    print("\n" + "=" * 60)
    print(f"SAMPLE PREDICTIONS (First {num_samples} Test Cases)")
    print("=" * 60)

    num_samples = min(num_samples, len(y_test))

    actual_labels = encoder.inverse_transform(y_test[:num_samples])
    predicted_labels = encoder.inverse_transform(y_pred[:num_samples])

    results = pd.DataFrame()
    results["N"] = X_test.iloc[:num_samples]["N"].round(0).astype(int).values
    results["P"] = X_test.iloc[:num_samples]["P"].round(0).astype(int).values
    results["K"] = X_test.iloc[:num_samples]["K"].round(0).astype(int).values
    results["Temp"] = X_test.iloc[:num_samples]["temperature"].round(1).values
    results["Humid"] = X_test.iloc[:num_samples]["humidity"].round(0).astype(int).values
    results["pH"] = X_test.iloc[:num_samples]["pH"].round(1).values
    results["Rain"] = X_test.iloc[:num_samples]["rainfall"].round(0).astype(int).values
    results["Actual"] = actual_labels
    results["Predicted"] = predicted_labels
    results["✓"] = [
        "✓" if a == p else "✗" for a, p in zip(actual_labels, predicted_labels)
    ]

    print(results.to_string(index=False))


def plot_feature_distributions(df, crops_to_plot, feature_cols):
    """
    Plots feature distributions for selected crops.

    Args:
        df: Full DataFrame
        crops_to_plot: List of crop names to plot
        feature_cols: List of feature column names
    """
    print("\n" + "=" * 60)
    print("FEATURE DISTRIBUTIONS")
    print("=" * 60)

    # Filter data for selected crops
    df_viz = df[df["label"].isin(crops_to_plot)]

    # Color palette
    palette = sns.color_palette("husl", len(crops_to_plot))
    colors = dict(zip(crops_to_plot, palette))

    # Create subplot grid
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()

    for idx, feature in enumerate(feature_cols):
        ax = axes[idx]

        # Plot distribution for each crop
        for crop in crops_to_plot:
            crop_data = df_viz[df_viz["label"] == crop][feature]
            sns.kdeplot(
                data=crop_data,
                label=crop.capitalize(),
                fill=True,
                alpha=0.3,
                color=colors[crop],
                linewidth=2,
                ax=ax,
            )

        ax.set_title(f"{feature.upper()} Distribution", fontsize=11, fontweight="bold")
        ax.set_xlabel(f"{feature.capitalize()} Value", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    # Hide unused subplots
    for idx in range(len(feature_cols), len(axes)):
        axes[idx].set_visible(False)

    crop_names = ", ".join([c.capitalize() for c in crops_to_plot])
    plt.suptitle(
        f"Feature Distributions: {crop_names}", fontsize=14, fontweight="bold", y=0.995
    )
    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\nFeature Statistics by Crop:")
    print("=" * 70)
    for crop in crops_to_plot:
        print(f"\n{crop.upper()}:")
        crop_data = df_viz[df_viz["label"] == crop][feature_cols]
        print(crop_data.describe().loc[["mean", "std"]].round(2))


# ============================================================================
# SANITY CHECKS & DIAGNOSTICS
# ============================================================================


def run_diagnostic_checks(
    df, X_train, y_train, X_test, y_test, crop_rows, feature_cols
):
    """
    Runs comprehensive diagnostic checks on the model and data.

    Args:
        df: Full DataFrame
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        crop_rows: Dictionary of crop selections
        feature_cols: List of feature columns
    """
    print("\n" + "=" * 60)
    print("DIAGNOSTIC CHECKS")
    print("=" * 60)

    # Check 1: Feature separation
    print("\n1. Feature Separation Analysis (First 3 Features):")
    for col in feature_cols[:3]:
        print(f"\n   {col}:")
        for crop in crop_rows.keys():
            crop_mean = df[df["label"] == crop][col].mean()
            crop_std = df[df["label"] == crop][col].std()
            print(f"      {crop:12}: μ={crop_mean:7.2f}, σ={crop_std:6.2f}")

    # Check 2: Random baseline
    print("\n2. Random Baseline Test:")
    y_shuffled = np.random.permutation(y_train)
    model_random = GaussianNB()
    model_random.fit(X_train, y_shuffled)
    random_accuracy = model_random.score(X_test, y_test)
    expected_random = 100 / len(crop_rows)
    print(f"   Random label accuracy: {random_accuracy * 100:.1f}%")
    print(f"   Expected (1/{len(crop_rows)} classes): ~{expected_random:.1f}%")

    if random_accuracy > 0.4:
        print("   ⚠️  WARNING: Random baseline is suspiciously high!")

    # Check 3: Data leakage
    print("\n3. Data Leakage Check:")
    train_set = set(map(tuple, X_train.values))
    test_set = set(map(tuple, X_test.values))
    overlap = train_set.intersection(test_set)
    print(f"   Duplicate feature vectors: {len(overlap)}")

    if len(overlap) > 0:
        print(f"   ⚠️  WARNING: {len(overlap)} test samples match training samples!")
    else:
        print("   ✓ No data leakage detected")


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Main execution function."""
    config = Config()

    print("=" * 70)
    print("MULTI-CROP NAIVE BAYES WITH COMPREHENSIVE DIAGNOSTICS")
    print("Section 5+: Scaled Implementation")
    print("=" * 70)
    print()

    # Load and prepare data
    train_dataset, test_dataset, full_df = load_and_prepare_data(config)

    # Feature preparation
    X_train = train_dataset[config.FEATURE_COLS]
    y_train = train_dataset["label"]
    X_test = test_dataset[config.FEATURE_COLS]
    y_test = test_dataset["label"]

    # Encode labels
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)

    print(
        f"Label encoding: {dict(zip(encoder.classes_, range(len(encoder.classes_))))}\n"
    )

    # Train model with cross-validation
    model, cv_scores = train_model_with_cv(X_train, y_train_encoded)

    # Evaluate model
    y_pred, accuracy = evaluate_model(model, X_test, y_test_encoded, encoder)

    # Show detailed results
    if config.SHOW_DETAILED_RESULTS:
        show_sample_predictions(X_test, y_test_encoded, y_pred, encoder)

    # Plot feature distributions
    if config.SHOW_FEATURE_DISTRIBUTIONS:
        plot_feature_distributions(
            full_df, list(config.CROP_ROWS.keys()), config.FEATURE_COLS
        )

    # Run sanity checks
    if config.RUN_SANITY_CHECKS:
        run_diagnostic_checks(
            full_df,
            X_train,
            y_train_encoded,
            X_test,
            y_test_encoded,
            config.CROP_ROWS,
            config.FEATURE_COLS,
        )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Final Test Accuracy: {accuracy * 100:.2f}%")
    print(
        f"Mean CV Accuracy: {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%"
    )
    print()
    print("KEY FINDINGS:")
    print("- High accuracy reflects clear feature separation in dataset")
    print("- Cross-validation shows model stability across different splits")
    print("- Diagnostic checks help identify potential data quality issues")
    print("- Feature distributions reveal which variables distinguish crops")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
