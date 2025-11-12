# Machine Learning for Smarter Crop Planning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-green)](https://scikit-learn.org/)

> A Naive Bayes classification system for agricultural crop recommendation based on soil nutrients and environmental conditions.

##### Final Project for **COMP 3009: Applied Math for Data Science and AI**  
University of Denver  
**Team**: Debasis Pani, Noah Sprenger, Sahithi Challapalli

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Key Findings](#key-findings)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)

---

## Overview

This project demonstrates the application of **Naive Bayes classification** to agricultural decision-making. Given soil nutrient data (N, P, K) and environmental conditions (temperature, humidity, pH, rainfall), our model predicts the optimal crop for those growing conditions.

### Problem Statement

Modern agriculture faces the challenge of optimizing crop selection for varying conditions. Poor choices lead to:
- Reduced yields and economic losses
- Inefficient resource utilization
- Environmental degradation

### Solution

We apply probabilistic machine learning (Naive Bayes) to:
- Learn crop preferences from historical data
- Predict optimal crops for new conditions
- Provide interpretable probability distributions
- Scale efficiently to large multi-crop datasets

### Hypothesis

**Naive Bayes can accurately predict optimal crop types based on soil and environmental features, providing a simple yet effective tool for agricultural decision-making.**

---

## Features

- **Manual Computation**: Step-by-step Bayesian probability calculations
- **Feature Discretization**: Intelligent binning of continuous variables
- **Python Implementation**: Scalable scikit-learn GaussianNB classifier
- **Multi-Crop Support**: Handles 2+ crop types
- **Comprehensive Diagnostics**: Cross-validation, sanity checks, and performance analysis
- **Data Visualization**: Feature distribution plots and confusion matrices
- **Production-Ready**: Configurable parameters and modular code structure

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Jupyter Notebook or JupyterLab

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ML_crop_planning.git
   cd ML_crop_planning
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook crop_rec_models.ipynb
   ```

---

## Usage

### Quick Start

1. Open `crop_rec_models.ipynb` in Jupyter
2. Run all cells sequentially (`Cell > Run All`)
3. Review outputs and visualizations
4. Modify configuration parameters to experiment

### Configuration Options

The notebook includes configurable parameters for experimentation:

```python
# In the scaled implementation cells:
CROP_ROWS = {"rice": 50, "maize": 40, "chickpea": 35}  # Samples per crop
TRAIN_SPLIT = 0.70  # Training data percentage
SHOW_DETAILED_RESULTS = True  # Display prediction tables
RUN_SANITY_CHECKS = True  # Run diagnostic tests
```

### Example Prediction

```python
# Test conditions: high N, low P, medium K, cool temp, medium humidity, acidic pH, medium rainfall
test_conditions = {
    "N_bin": "high",
    "P_bin": "low", 
    "K_bin": "med",
    "temperature_bin": "cool",
    "humidity_bin": "med",
    "ph_bin": "acidic",
    "rainfall_bin": "med"
}

# Model predicts: Rice (97.8% probability)
```

---

## Project Structure

```
ML_crop_planning/
├── crop_rec_models.ipynb          # Main Jupyter notebook
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── Crop dataset updated.csv        # Full 1,400-sample dataset (22 crops)
├── 2-crop-tiny-dataset.xlsx       # Small 15-sample dataset (2 crops)
├── 1400row_crop_dataset.csv       # Alternative large dataset
└── ML_crop_planning_deck.rtf      # Presentation slides
```

---

## Methodology

### 1. Data Preprocessing

**Feature Discretization**: Convert continuous features to categorical bins using tercile-based thresholds
- Nitrogen (N): low / med / high
- Phosphorus (P): low / med / high  
- Potassium (K): low / med / high
- Temperature: cool / temperate / warm
- Humidity: low / med / high
- pH: acidic / neutral / alkaline
- Rainfall: low / med / high

### 2. Naive Bayes Classification

**Mathematical Foundation** - Bayes' Theorem:

$$
P(\text{Crop} \mid \text{Conditions}) = \frac{P(\text{Conditions} \mid \text{Crop}) \times P(\text{Crop})}{P(\text{Conditions})}
$$

**Algorithm Steps**:
1. Calculate **priors**: $P(\text{Crop})$ from training data
2. Calculate **likelihoods**: $P(\text{Feature} \mid \text{Crop})$ for each feature
3. Compute **posteriors**: Multiply prior by all likelihoods
4. **Normalize**: Scale probabilities to sum to 1
5. **Predict**: Choose crop with highest posterior probability

### 3. Implementation Approaches

**Manual Computation** (Section 3)
- 15-sample dataset (9 rice, 6 maize)
- Hand-calculated priors, likelihoods, posteriors
- Educational demonstration of Bayesian inference

**Python Implementation** (Section 4)
- Programmatic calculation using pandas
- Modular functions for priors and conditionals
- Matches manual results for validation

**Scaled Implementation** (Sections 5-6)
- Gaussian Naive Bayes (scikit-learn)
- Multi-crop support (3-22 crops)
- Cross-validation and diagnostics
- Handles continuous features without binning

---

## Results

### Performance Summary

| Configuration | Training Samples | Test Samples | Accuracy | Notes |
|---------------|------------------|--------------|----------|-------|
| **Small Dataset** | 12 (2 crops) | 3 | 100% | Clear feature separation |
| **Medium Dataset** | 88 (3 crops) | 37 | 98% | High performance maintained |
| **Cross-Validation** | 88 (3 crops) | 5-fold | 90%±20% | Shows stability |

### Per-Crop Performance (3-Crop Model)

| Crop | Test Samples | Correct | Accuracy |
|------|--------------|---------|----------|
| Rice | 15 | 15 | 100% |
| Maize | 12 | 11 | 91.7% |
| Chickpea | 10 | 10 | 100% |

### Sample Predictions

```
 N   P   K  Temp  Humid  pH  Rain  Actual    Predicted  ✓
 90  42  43  20.9     82 6.5   203  rice      rice      ✓
 71  54  16  22.6     64 5.7    88  maize     maize     ✓
 40  67  80  20.9     82 7.0   200  chickpea  chickpea  ✓
```

---

## Key Findings

### Why Such High Accuracy?

Our diagnostic analysis revealed an important insight: **the dataset exhibits near-perfect feature separation**.

**Feature Separation Example - Potassium (K):**
- **Rice**: K ≈ 40 (±3)
- **Maize**: K ≈ 20 (±3)
- **Chickpea**: K ≈ 80 (±3)

With ~20-point gaps and only ~3-point standard deviations, there's essentially **zero overlap** between crops. This creates 5-6 sigma separation—crops occupy completely distinct regions in feature space.

### Real-World Implications

**Expected Performance with Real Data**: 70-85% accuracy (vs. our 98%+)

Real agricultural applications would face:
- Significant feature overlap between viable crops
- Measurement noise from soil testing
- Regional variations in optimal conditions
- Multiple crops viable for same conditions

### When Naive Bayes Works Well

- Features have distinct distributions per class  
- Feature independence assumption approximately holds  
- Large enough dataset for stable probability estimates  
- Computational efficiency matters  

### Limitations

- **Naive independence assumption**: Ignores feature correlations (temp × humidity)  
- **Zero-frequency problem**: Unseen combinations get zero probability  
- **Gaussian assumption**: May not hold for all features  
- **Linear boundaries**: Can't capture complex non-linear patterns  

### Critical Lesson

**High accuracy doesn't always indicate a sophisticated model**—it may simply reflect trivially separable data. Always validate that your model is solving a genuinely difficult problem through:
- Baseline comparisons (random guessing)
- Sanity checks (feature distribution analysis)
- Cross-validation
- Real-world testing

---

## Future Work

### Data Improvements
- [ ] Use real field data with measurement noise
- [ ] Include economic factors (crop prices, input costs)
- [ ] Add temporal features (planting season, climate trends)
- [ ] Incorporate soil texture and drainage data

### Model Enhancements
- [ ] Create interaction features (temp × humidity)
- [ ] Ensemble methods (combine with Decision Trees, Random Forest)
- [ ] Incorporate real-time sensor data
- [ ] Add confidence intervals for predictions

### Production Deployment
- [ ] Build user-friendly web interface for farmers
- [ ] Integrate with soil testing services
- [ ] Provide ranked recommendations with confidence scores
- [ ] Mobile app for field use

---

## References

1. **Swetha, P., & Senthilkumar, J. (2025)**. Development of a Naive Bayes-based Framework for Optimizing Crop Recommendation System and Enhancing Agricultural Yield Prediction. *2025 4th International Conference on Sentiment Analysis and Deep Learning (ICSADL)*, Bhimdatta, Nepal, pp. 1262-1267. doi: [10.1109/ICSADL65848.2025.10933086](https://doi.org/10.1109/ICSADL65848.2025.10933086)

2. **Ingle, Atharva (2021)**. Crop Recommendation Dataset. *Kaggle*. Retrieved from [https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)

3. **Scikit-learn Documentation**. [Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)

4. **Murphy, K. P. (2012)**. *Machine Learning: A Probabilistic Perspective*. MIT Press.

5. **Russell, S., & Norvig, P. (2020)**. *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.

---

## Team

- **Debasis Pani**
- **Noah Sprenger**
- **Sahithi Challapalli**

**Course**: COMP 3009: Applied Mathematics for Data Science and AI  
**Instructor**: Dr. Ahmed Abdeen Hamed
**Institution**: University of Denver  
**Term**: 2024-2025

---

## 📄 License

This project is licensed for educational purposes as part of COMP 3009 coursework.

---

## 🙏 Acknowledgments

- Course instructor for guidance on probabilistic modeling
- Atharva Ingle for making the Crop Recommendation Dataset publicly available on Kaggle
- Swetha & Senthilkumar for foundational research on crop recommendation systems
- Open-source community for excellent ML tools (scikit-learn, pandas, matplotlib)

---

## 📧 Contact

For questions or collaboration:
- Open an issue on GitHub
- Contact team members through university email

---

**If you find this project useful, please consider starring the repository!**
