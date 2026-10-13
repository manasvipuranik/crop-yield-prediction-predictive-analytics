# Crop Yield Prediction Using Predictive Analytics

**Course:** Predictive Analytics (CA3)
**Project type:** Regression
**Notebook:** `Predictive_Analytics_CA3_Final.ipynb`
**Goal:** Predict crop yield from agricultural and climate-related information using k-Nearest Neighbours and a tree-ensemble model, both implemented from scratch in Python.

---

## Table of Contents

1. [Problem Identification](#1-problem-identification)
2. [Dataset](#2-dataset)
3. [Project Workflow](#3-project-workflow)
4. [Data Preprocessing](#4-data-preprocessing)
5. [Exploratory Data Analysis](#5-exploratory-data-analysis)
6. [Data Transformation](#6-data-transformation)
7. [Predictive Modelling](#7-predictive-modelling)
8. [Model Evaluation](#8-model-evaluation)
9. [Results and Discussion](#9-results-and-discussion)
10. [Feature Importance and Interpretation](#10-feature-importance-and-interpretation)
11. [Interactive Prediction Tool](#11-interactive-prediction-tool)
12. [Limitations](#12-limitations)
13. [Future Work](#13-future-work)
14. [Background](#14-background)
15. [Repository Structure](#15-repository-structure)
16. [How to Run](#16-how-to-run)
17. [Requirements](#17-requirements)

---

## 1. Problem Identification

Crop yield depends on several factors, including the crop being cultivated, the geographical area, the year, rainfall, pesticide use, and average temperature. Estimating yield from these factors can support agricultural planning and comparison.

**Problem statement:** Develop and evaluate predictive regression models that estimate crop yield (`hg/ha_yield`) from the available agricultural and climate features.

### Objectives

- Understand and prepare the crop-yield dataset.
- Explore patterns in crop yield and its relationship with the available features.
- Build predictive models using k-Nearest Neighbours (k-NN) and a tree-ensemble approach.
- Evaluate and compare predictions using MAE, RMSE, and R².
- Interpret the results and demonstrate a prediction for user-provided inputs.

### Target and Input Features

| Role | Variable | Description |
|---|---|---|
| **Target** | `hg/ha_yield` | Crop yield in hectograms per hectare |
| Feature | `Area` | Country / area |
| Feature | `Item` | Crop type |
| Feature | `Year` | Year of observation |
| Feature | `average_rain_fall_mm_per_year` | Annual rainfall (mm) |
| Feature | `pesticides_tonnes` | Pesticide use (tonnes) |
| Feature | `avg_temp` | Average temperature (°C) |

Because the target is numerical, this is a **regression** task rather than a classification task.

---

## 2. Dataset

- **File:** `yield_df.csv`
- **Raw size:** 28,242 rows × 8 columns (including an unnamed index column)
- **Time span:** 1990 – 2013
- **Missing values:** none
- **Duplicate rows:** 2,310 (removed)
- **Rows after cleaning:** 25,932

### Summary Statistics (after cleaning)

| Statistic | Year | hg/ha_yield | Rainfall (mm/yr) | Pesticides (t) | Avg. temp (°C) |
|---|---|---|---|---|---|
| Mean | 2001.56 | 76,978.13 | 1,146.84 | 34,782.57 | 20.42 |
| Std. dev. | 7.06 | 84,647.61 | 717.39 | 57,228.57 | 6.44 |
| Min | 1990 | 50 | 51 | 0.04 | 1.30 |
| 25% | 1995 | 19,996 | 593 | 1,597 | 16.58 |
| Median | 2001 | 39,530 | 1,083 | 15,373 | 21.42 |
| 75% | 2008 | 104,168 | 1,668 | 45,620 | 26.06 |
| Max | 2013 | 501,412 | 3,240 | 367,778 | 30.65 |

> **Note:** The notebook loads the data from Google Drive (`/content/drive/MyDrive/yield_df.csv`). Keep the file at that location when running in Google Colab, or update the path in the loading cell if running elsewhere.

---

## 3. Project Workflow

The notebook follows a structured predictive-modelling process:

1. **Problem identification**: define the target, features, and objectives.
2. **Data understanding**: inspect shape, columns, data types, missing values, and duplicates.
3. **Data preprocessing**: drop the index column and remove duplicates.
4. **Exploratory data analysis**: visualise yield by crop, country, year, and correlations.
5. **Data transformation**: encode categories, split into train/test sets, and standardise.
6. **Predictive modelling**: build k-NN and a bagged tree ensemble.
7. **Model evaluation**: compare models using MAE, RMSE, and R², plus several diagnostic graphs.
8. **Data interpretation**: estimate feature importance via permutation.
9. **Prediction**: an interactive tool to predict yield for user-supplied inputs.

---

## 4. Data Preprocessing

- Dropped the unnecessary `Unnamed: 0` index column.
- Checked for missing values (none found) and dropped any rows with missing data.
- Removed **2,310 duplicate records**, leaving **25,932** rows.
- Created a helper column `yield_t_ha` (tonnes per hectare, where 1 tonne = 10,000 hg) for easier-to-read graphs. Modelling still uses the original `hg/ha_yield` target.

---

## 5. Exploratory Data Analysis

The EDA explores average yield by crop and country, the yearly yield trend, and the correlation between yield and each numerical factor. Four graphs are produced:

| Graph | Description |
|---|---|
| Average Yield by Crop | Horizontal bar chart of mean yield (t/ha) for each crop |
| Top 10 Countries by Average Yield | Countries with the highest mean yield |
| Average Crop Yield Over the Years | Line chart of mean yield per year |
| Factor Correlation with Yield | Correlation of Year, Rainfall, Pesticides, and Temperature with yield |

Correlation indicates association; it does not by itself prove that a factor causes a change in yield.

---

## 6. Data Transformation

1. **Categorical encoding:** `Area` and `Item` are converted to integer codes using sorted category lists.
2. **Train-test split:** 80% training / 20% testing using a random permutation with `np.random.seed(42)`, giving **20,745 training samples**.
3. **Test subset:** To keep the pure-Python models fast, evaluation uses the **first 1,000 test rows**.
4. **Standardisation:** Features are scaled using the **training set** mean and standard deviation only, which avoids data leakage from the test set.

---

## 7. Predictive Modelling

Both models are implemented manually with plain Python loops, with no ready-made machine-learning library, so every step is visible.

### 7.1 k-Nearest Neighbours (k-NN) Regression

- Computes the squared Euclidean distance between a query point and every training record.
- Sorts by distance and predicts the **average yield of the k nearest neighbours**.
- **Choosing k:** A validation set carved out of the training data (85% sub-training / 300 validation rows) is used to test `k ∈ {1, 3, 5, 7, 10, 15, 20, 30}`. The value of k with the lowest validation RMSE is selected.
- **Selected value: k = 1.**

### 7.2 Tree Ensemble (Bagged Decision Trees, "Random Forest")

- **Split rule:** For every feature, 9 evenly spaced cut points between the minimum and maximum are tried. The split with the lowest weighted MSE of the two child groups is chosen.
- **Stopping rules:** A node becomes a leaf (average yield) when the maximum depth is reached or it has fewer than 10 samples.
- **Ensemble:** Each tree is trained on a bootstrap sample (random rows with replacement); predictions are averaged across trees.

| Hyperparameter | Value |
|---|---|
| Number of trees | 15 |
| Maximum depth | 8 |
| Bootstrap sample size per tree | 3,000 |
| Random seed | 42 |

> **Implementation note:** This is more precisely a **bagged decision-tree ensemble**. It does not use random feature selection at each split, which is part of the standard Random Forest algorithm.

---

## 8. Model Evaluation

Models are compared using three standard regression metrics:

- **MAE (Mean Absolute Error):** average absolute prediction error; lower is better.
- **RMSE (Root Mean Squared Error):** gives greater weight to large errors; lower is better.
- **R² (coefficient of determination):** proportion of variance explained; closer to 1 is better.

Additional diagnostic graphs produced:

| Graph | Purpose |
|---|---|
| Choosing k for k-NN | Validation RMSE for each candidate k |
| k-NN vs Random Forest | Side-by-side bars of R², MAE, and RMSE |
| Actual vs Predicted | Scatter plots against the perfect-prediction line |
| Distribution of Prediction Errors | Overlaid histograms of errors in t/ha |
| Prediction Accuracy Within Tolerance | Share of predictions within 10%, 20%, and 30% of the actual yield |
| Prediction Error by Crop | Mean absolute error for each crop and model |

---

## 9. Results and Discussion

### Test-Set Performance (1,000 test records, units: hg/ha)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **k-NN Regression (k = 1)** | **5,007.24** | **16,582.45** | **0.9610** |
| Tree Ensemble (15 trees, depth 8) | 13,187.79 | 22,630.21 | 0.9273 |

### Key Findings

1. **Best model: k-NN** (R² = 0.961), with an average error of about **0.50 tonnes/ha**.
2. **k = 1** was selected as the best value on the validation set.
3. **Within 20% of the real yield:** k-NN **88%** of predictions vs. tree ensemble **59%**.
4. **Most important feature:** **Crop** for both models.
5. The k-NN model has lower MAE, lower RMSE, and higher R² than the tree ensemble on this test subset.

These results describe this test subset and do not guarantee the same performance on new regions or future years. Re-run the notebook from top to bottom to reproduce the exact figures.

---

## 10. Feature Importance and Interpretation

Feature importance is estimated using **permutation importance**:

1. A random sample of 200 test records is selected.
2. Each feature is shuffled one at a time, which breaks its relationship with the target.
3. The **increase in RMSE** (in tonnes/ha) relative to the unshuffled baseline is recorded.

A larger increase in error means the model relies more heavily on that feature. Results are plotted side by side for k-NN and the tree ensemble, and crop type is the most influential feature in both.

---

## 11. Interactive Prediction Tool

The final section lets a user predict yield for their own inputs. It prompts for:

| Input | Validation |
|---|---|
| Country | Must match a country in the dataset (case-insensitive) |
| Crop | Must match a crop in the dataset (case-insensitive) |
| Year | Within the range seen in training (1990 – 2013) |
| Rainfall (mm/year) | Within the training range |
| Pesticides (tonnes) | Within the training range |
| Average temperature (°C) | Within the training range |

Invalid entries are rejected and the user is asked again. Inputs are scaled with the same training mean and standard deviation, then passed to both models. The tool outputs:

- k-NN prediction (t/ha)
- Tree ensemble prediction (t/ha)
- Average of both models
- Difference between the two models

A bar chart of the three predictions is also displayed.

---

## 12. Limitations

- **Categorical encoding:** Country and crop are represented by integer codes, which can imply an artificial order or distance between categories. This particularly affects distance-based methods such as k-NN.
- **Random split:** Records for the same country and crop in different years may appear in both training and test sets. For forecasting future years, a time-based split is more appropriate.
- **Test subset:** Only the first 1,000 test rows are used for evaluation to keep the custom Python models manageable.
- **Selected k = 1:** A single-neighbour model is highly sensitive to near-duplicate records and noise, so the strong k-NN score should be interpreted with this in mind.
- **Association, not causation:** The models identify patterns in the data and do not prove that rainfall, pesticide use, or temperature causes a particular yield.
- **Simplified ensemble:** The tree ensemble is a manually implemented bagging approach, not a full standard Random Forest, and uses limited cut points per feature.
- **Generalisation:** Results may not carry over to new regions or years outside 1990 – 2013.

---

## 13. Future Work

- Use one-hot encoding, or a distance measure designed for mixed categorical and numerical data.
- Compare against established library models (e.g. scikit-learn implementations).
- Evaluate with time-based or region-based splits.
- Tune model parameters systematically (e.g. cross-validation, grid search).
- Evaluate on the full test set rather than a 1,000-row subset.
- Test on additional or more recent data.

---

## 14. Background

Predictive modelling is commonly treated as a structured process: understanding the problem and data, preparing data, building models, evaluating results, and communicating findings. This project follows that approach through data inspection, preprocessing, exploratory analysis, model development, evaluation, and interpretation.

k-Nearest Neighbours is a distance-based method that predicts a numerical target from nearby observations. Its performance depends on feature scaling and on how categorical variables are represented. Tree-based methods model nonlinear relationships by splitting observations into groups, and combining trees through bootstrap sampling and averaging can reduce variability compared with relying on a single tree.

The notebook provides an educational implementation of these ideas, and its results should be read in light of the category encoding, the limited test subset, and the random train-test split.

---

## 15. Repository Structure

```
.
├── Predictive_Analytics_CA3_Final.ipynb   # Main notebook (analysis, models, evaluation, prediction)
├── yield_df.csv                           # Dataset (add to repo or place in Google Drive)
└── README.md                              # Project documentation
```

---

## 16. How to Run

### Option A: Google Colab (as originally written)

1. Upload `yield_df.csv` to the root of your Google Drive (`MyDrive`).
2. Open `Predictive_Analytics_CA3_Final.ipynb` in Google Colab.
3. Run all cells from top to bottom (`Runtime → Run all`) and approve the Drive mount prompt.
4. Enter values when the prediction cell asks for input.

### Option B: Local Jupyter

1. Clone the repository and place `yield_df.csv` alongside the notebook.
2. In the notebook, **remove or skip** the Google Drive mount cell:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Change the data-loading path to:
   ```python
   df = pd.read_csv("yield_df.csv")
   ```
4. Install the requirements (below) and launch Jupyter:
   ```bash
   pip install pandas numpy matplotlib notebook
   jupyter notebook
   ```
5. Run all cells from top to bottom.

> **Runtime note:** The k-NN and tree-ensemble code uses pure Python loops, so the k-selection, forest training, and permutation-importance cells can take several minutes.

---

## 17. Requirements

- Python 3.8+
- `pandas`
- `numpy`
- `matplotlib`
- Jupyter Notebook or Google Colab
