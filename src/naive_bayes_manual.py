"""
Machine Learning for Smarter Crop Planning
COMP 3009 Final Project - Part 1: Manual Naive Bayes Implementation

This file contains the building blocks for Naive Bayes classification:
- Data loading and preprocessing
- Feature discretization (binning)
- Prior probability calculation
- Conditional probability calculation
- Posterior probability calculation
"""

import pandas as pd
import numpy as np


# ============================================================================
# HELPER FUNCTIONS FOR FEATURE DISCRETIZATION
# ============================================================================


def compute_quantile_cuts(data_frame, cut_pts=(0.33, 0.66)):
    """
    Extracts unique crop types from the label column.

    Args:
        data_frame: DataFrame with continuous features
        cut_pts: Tuple of quantile points for binning (default: terciles)

    Returns:
        Dictionary of cut points for each numeric column
    """
    # get all numeric columns (in our case, all except label)
    numeric_cols = data_frame.columns[:-1]
    # dictionary to hold cutoffs
    cutoffs = {}

    # compute cuts for each numeric column
    for column in numeric_cols:
        # compute quantiles for each cut point
        cuts = [float(data_frame[column].quantile(pt)) for pt in cut_pts]
        # store cuts as values in dictionary
        cutoffs[column] = cuts

    # return the cutoffs dictionary
    return cutoffs


def apply_bins(df, cutoffs, label_map=None, default_labels=("low", "med", "high")):
    """
    Apply binning to continuous features based on cut points.

    Args:
        df: DataFrame with continuous features
        cutoffs: Dictionary of cut points from compute_quantile_cuts()
        label_map: Optional dictionary mapping column names to custom labels
        default_labels: Default labels if label_map not provided

    Returns:
        DataFrame with additional binned columns (e.g., 'N_bin', 'P_bin')
    """
    df_binned = df.copy()

    for col, cuts in cutoffs.items():
        # build bins - ex: [-inf, cut1, cut2, ... cutn, +inf]
        bins = [float("-inf")] + list(cuts) + [float("inf")]

        # get custom labels if available
        labels = label_map.get(col, default_labels) if label_map else default_labels

        # label count must match bins-1
        if len(labels) != len(bins) - 1:
            raise ValueError(
                f"Column '{col}' has {len(bins) - 1} bins but {len(labels)} labels."
            )

        # apply binning
        df_binned[f"{col}_bin"] = pd.cut(df_binned[col], bins=bins, labels=labels)

    # return data frame with new binning columns
    return df_binned


# ============================================================================
# HELPER FUNCTIONS FOR NAIVE BAYES CALCULATIONS
# ============================================================================


def get_crop_types(data_frame):
    """
    Extracts unique crop types from the label column.

    Returns:
        Array of unique crop names
    """
    return data_frame["label"].unique()


def filter_by_crop(data_frame, crop):
    """
    Filter dataframe to only rows matching a specific crop type.

    Args:
        data_frame: DataFrame with 'label' column
        crop: Crop name to filter by

    Returns:
        DataFrame containing only rows where label == crop
    """
    return data_frame[data_frame["label"] == crop]


# ============================================================================
# PRIOR PROBABILITY CALCULATION
# ============================================================================


def get_priors(data_frame):
    """
    Computes prior probabilities for each class in the 'label' column.

    Args:
        data_frame: DataFrame with 'label' column

    Returns:
        Dictionary like {'p_rice': 0.6, 'p_maize': 0.4}, for each unique label
    """
    # get total number of records
    total = len(data_frame)
    # get unique crop types
    crop_types = get_crop_types(data_frame)
    # make a dictionary to hold priors
    priors = {}

    # for each crop type
    for crop in crop_types:
        # use helper to filter
        crop_data = filter_by_crop(data_frame, crop)
        # calculate prior probability
        p_crop = len(crop_data) / total
        priors[f"p_{crop}"] = p_crop

    # return priors dictionary
    return priors


# ============================================================================
# CONDITIONAL PROBABILITY CALCULATION
# ============================================================================


def get_conditionals(df_binned, test_conditions):
    """
    Computes conditional probabilities for specific feature values given each crop type.

    Args:
        df_binned: DataFrame with binned columns
        test_conditions: Dictionary of test features
                        {'N_bin': 'high', 'P_bin': 'low', 'K_bin': 'med', ...}

    Returns:
        Dictionary of conditionals for each crop:
        {
            'rice': {
                'N_bin': 0.444,
                'P_bin': 0.444,
                ...
            },
            'maize': {
                'N_bin': 0.167,
                ...
            }
        }
    """
    # use helper functions
    crop_types = get_crop_types(df_binned)
    # make a dict to hold conditionals
    conditionals = {}

    # for each crop type
    for crop in crop_types:
        # use helper to filter by crop
        crop_data = filter_by_crop(df_binned, crop)
        # then compute total rows for that crop
        total_crop_rows = len(crop_data)
        # make a dict within conditionals for this crop
        conditionals[crop] = {}

        # for each feature in the test conditions
        for feature_col, test_value in test_conditions.items():
            # count how many times the test value appears for this crop
            count = len(crop_data[crop_data[feature_col] == test_value])

            # compute P(feature=value | crop)
            conditionals[crop][feature_col] = count / total_crop_rows

    return conditionals


def conditionals_display(conditionals, test_conditions):
    """
    Pretty-prints conditional probabilities in readable format.

    Args:
        conditionals: Dictionary from get_conditionals()
        test_conditions: Dictionary of test features
    """
    print("Conditional Probabilities for Test Case:")
    print("_" * 50)

    for crop, probs in conditionals.items():
        print(f"\n{crop.upper()}:")
        print("-" * 30)
        for feature, prob in probs.items():
            # extract feature name and remove '_bin' suffix for cleaner display
            feature_name = feature.replace("_bin", "")
            test_value = test_conditions[feature]
            print(f"  P({feature_name}={test_value} | {crop}) = {prob:.3f}")

    print("\n" + "=" * 50)


# ============================================================================
# POSTERIOR PROBABILITY CALCULATION
# ============================================================================


def get_posteriors(priors, conditionals):
    """
    Computes unnormalized posterior probabilities for each crop.

    Args:
        priors: Dictionary from get_priors()
        conditionals: Dictionary from get_conditionals()

    Returns:
        Dictionary of unnormalized posteriors for each crop
    """
    # make dict to hold posteriors
    posteriors = {}

    for prior_key, prior_probability in priors.items():
        crop_name = prior_key.replace("p_", "")
        # initial posterior value as the first term in the posterior equation
        posterior = prior_probability

        for feature_col, cond_prob in conditionals[crop_name].items():
            posterior *= cond_prob

        posteriors[crop_name] = posterior

    return posteriors


def normalize_posteriors(posteriors):
    """
    Normalizes posterior probabilities to sum to 1.

    Args:
        posteriors: Dictionary of unnormalized posteriors

    Returns:
        Dictionary of normalized posteriors
    """
    total = sum(posteriors.values())
    return {crop: prob / total for crop, prob in posteriors.items()}


def predict_crop(normalized_posteriors):
    """
    Predicts the most likely crop based on normalized posteriors.

    Args:
        normalized_posteriors: Dictionary from normalize_posteriors()

    Returns:
        Tuple of (predicted_crop, probability)
    """
    predicted_crop = max(normalized_posteriors, key=normalized_posteriors.get)
    probability = normalized_posteriors[predicted_crop]
    return predicted_crop, probability


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("NAIVE BAYES MANUAL IMPLEMENTATION - CROP RECOMMENDATION")
    print("=" * 70)
    print()

    # Load data
    print("Loading data...")
    data_frame = pd.read_excel("data/2-crop-tiny-dataset.xlsx")
    print(f"Loaded {len(data_frame)} samples")
    print(f"Crops: {', '.join(get_crop_types(data_frame))}")
    print()

    # Display sample data
    print("Sample data:")
    print(data_frame.head())
    print()

    # Discretize features
    print("Discretizing features...")
    label_map = {
        "N": ("low", "med", "high"),
        "P": ("low", "med", "high"),
        "K": ("low", "med", "high"),
        "temperature": ("cool", "temperate", "warm"),
        "humidity": ("low", "med", "high"),
        "ph": ("acidic", "neutral", "alkaline"),
        "rainfall": ("low", "med", "high"),
    }

    cuts = compute_quantile_cuts(data_frame)
    df_binned = apply_bins(data_frame, cuts, label_map)
    print("Discretization complete!")
    print()

    # Display binned data
    print("Binned data (first 5 rows):")
    print(df_binned.head())
    print()

    # Calculate priors
    print("-" * 70)
    print("STEP 1: PRIOR PROBABILITIES")
    print("-" * 70)
    priors = get_priors(data_frame)
    for crop, prob in priors.items():
        print(f"{crop}: {prob:.3f}")
    print()

    # Define test case
    print("-" * 70)
    print("STEP 2: TEST CASE")
    print("-" * 70)
    test_case = {
        "N_bin": "high",
        "P_bin": "low",
        "K_bin": "med",
        "temperature_bin": "cool",
        "humidity_bin": "med",
        "ph_bin": "acidic",
        "rainfall_bin": "med",
    }

    print("Test conditions:")
    for feature, value in test_case.items():
        print(f"  {feature.replace('_bin', '')}: {value}")
    print()

    # Calculate conditionals
    print("-" * 70)
    print("STEP 3: CONDITIONAL PROBABILITIES")
    print("-" * 70)
    conditionals = get_conditionals(df_binned, test_case)
    conditionals_display(conditionals, test_case)
    print()

    # Calculate posteriors
    print("-" * 70)
    print("STEP 4: POSTERIOR PROBABILITIES (Unnormalized)")
    print("-" * 70)
    posteriors = get_posteriors(priors, conditionals)
    for crop, prob in posteriors.items():
        print(f"P({crop} | conditions) ∝ {prob:.6f}")
    print()

    # Normalize posteriors
    print("-" * 70)
    print("STEP 5: NORMALIZED POSTERIORS")
    print("-" * 70)
    normalized = normalize_posteriors(posteriors)
    for crop, prob in normalized.items():
        print(f"P({crop} | conditions) = {prob:.3f} ({prob * 100:.1f}%)")
    print()

    # Final prediction
    print("-" * 70)
    print("FINAL PREDICTION")
    print("-" * 70)
    predicted_crop, confidence = predict_crop(normalized)
    print(f"Predicted crop: {predicted_crop.upper()}")
    print(f"Confidence: {confidence * 100:.1f}%")
    print()

    print("=" * 70)
    print("Analysis complete!")
    print("=" * 70)
