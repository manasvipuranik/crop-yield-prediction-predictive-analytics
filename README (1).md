# Crop Yield Prediction using the KDD Process

**Course:** Predictive Analytics
**Task type:** Regression
**Models:** k-Nearest Neighbours (k-NN) and Random Forest, both implemented **from scratch with maths** (NumPy only)
**Notebook:** `Crop_Yield_KDD_Easy.ipynb`

---

## Table of contents

1. [Problem statement](#1-problem-statement)
2. [Dataset](#2-dataset)
3. [Project structure](#3-project-structure)
4. [Requirements and how to run](#4-requirements-and-how-to-run)
5. [The KDD process used in this project](#5-the-kdd-process-used-in-this-project)
6. [Data understanding and preprocessing](#6-data-understanding-and-preprocessing)
7. [Predictive modelling approach](#7-predictive-modelling-approach)
8. [Model development (the maths)](#8-model-development-the-maths)
9. [Evaluation of the models](#9-evaluation-of-the-models)
10. [Results and interpretation](#10-results-and-interpretation)
11. [Predicting a new yield](#11-predicting-a-new-yield)
12. [Conclusion](#12-conclusion)
13. [Limitations and future work](#13-limitations-and-future-work)
14. [References](#14-references)

---

## 1. Problem statement

Crop yield depends on many factors: the crop grown, the country, the year, rainfall, pesticide use and temperature. Being able to estimate yield from these factors supports agricultural planning and comparison between regions and crops.

> **Problem statement:** Build and evaluate predictive regression models that estimate crop yield (`hg/ha_yield`) from country, crop, year, annual rainfall, pesticide use and average temperature.

### Objectives

- Understand and prepare the crop-yield data.
- Explore patterns in yield and its relationship with the available features.
- Build k-NN and Random Forest regressors using mathematics, without ready-made model libraries.
- Evaluate and compare the models with MAE, RMSE and R².
- Test whether adding the rainfall and pesticide datasets improves predictions.
- Interpret the results and predict yield for user-provided inputs.

### Target and input features

| Role | Name | Description |
|---|---|---|
| Target | `hg/ha_yield` | Crop yield in hectograms per hectare (1 tonne = 10,000 hg) |
| Feature | `Area` | Country |
| Feature | `Item` | Crop |
| Feature | `Year` | Year (1990 to 2013) |
| Feature | `average_rain_fall_mm_per_year` | Average annual rainfall in mm |
| Feature | `pesticides_tonnes` | Total pesticides used, in tonnes |
| Feature | `avg_temp` | Average temperature in °C |

---

## 2. Dataset

Three CSV files are used.

| File | Columns used | Notes |
|---|---|---|
| `yield_df.csv` | `Area`, `Item`, `Year`, `hg/ha_yield`, `avg_temp` | Main file: yield and temperature |
| `rainfall.csv` | `Area`, `Year`, `average_rain_fall_mm_per_year` | Column names contain stray spaces, which the code strips. Some values are missing |
| `pesticides.csv` | `Area`, `Year`, `Value` (renamed `pesticides_tonnes`) | Only rows where `Item == "Pesticides (total)"` are kept |

The files are joined with an **inner join on Country (`Area`) + `Year`**. Rows with no rainfall or pesticide data are therefore dropped, so the merged table is smaller than the yield file.

After cleaning, the dataset has **25,932 records** covering **1990 to 2013** and **10 crops** (potatoes, cassava, sweet potatoes, yams, plantains and others, rice (paddy), maize, wheat, sorghum, soybeans).

Value ranges seen in the data:

| Feature | Range |
|---|---|
| Year | 1990 to 2013 |
| Rainfall | 51 to 3,240 mm per year |
| Pesticides | 0 to 367,778 tonnes |
| Temperature | 1 to 31 °C |

> **Data source:** *[add the source and download link for the three files here]*

---

## 3. Project structure

Suggested repository layout:

```
.
├── README.md
├── Crop_Yield_KDD_Easy.ipynb      # full analysis, models and evaluation
├── data/
│   ├── yield_df.csv
│   ├── rainfall.csv
│   └── pesticides.csv
├── images/                        # graphs exported from the notebook
└── presentation/                  # slide deck export (PDF or PPTX), optional
```

---

## 4. Requirements and how to run

### Requirements

- Python 3.8+
- `numpy`
- `pandas`
- `matplotlib`
- Jupyter Notebook, JupyterLab or Google Colab

```bash
pip install numpy pandas matplotlib notebook
```

No machine-learning library (such as scikit-learn) is used. k-NN and Random Forest are written with plain NumPy maths.

### Run in Google Colab

1. Upload `yield_df.csv`, `rainfall.csv` and `pesticides.csv` to Google Drive, in `MyDrive`.
2. Open `Crop_Yield_KDD_Easy.ipynb` in Colab.
3. Run all cells from top to bottom. The first cell mounts Drive and sets `FOLDER = "/content/drive/MyDrive"`.

### Run locally

1. Put the three CSV files in the same folder as the notebook (or change `FOLDER` in the first cell).
2. Start Jupyter and run all cells:

```bash
jupyter notebook Crop_Yield_KDD_Easy.ipynb
```

When the notebook is not running in Colab, `FOLDER` automatically falls back to `"."` (the current folder).

### Interactive prediction

The last section of the notebook uses `input()` prompts. Run it and type a country, crop, year, rainfall, pesticides and temperature. Country and crop names are not case sensitive. Each number is checked against the range seen in the training data.

### Reproducibility

Random seeds are fixed (`np.random.seed(42)` for the split and the forest, and `RandomState(0)` for feature importance), so a fresh run gives the same split and results.

---

## 5. The KDD process used in this project

The notebook and slides follow the five steps of **Knowledge Discovery in Databases (KDD)**.

| KDD step | What is done in this project |
|---|---|
| **1. Selection** | Load the three files, select the useful columns and join them on Country + Year |
| **2. Pre-processing** | Remove missing values and duplicates, check outliers with the IQR rule |
| **3. Transformation** | Encode country and crop as numbers, split 80/20 into train and test, standardise features |
| **4. Data mining** | Build k-NN and Random Forest regressors from scratch |
| **5. Interpretation / evaluation** | MAE, RMSE, R², graphs, feature importance, feature-set comparison and the discovered knowledge |

---

## 6. Data understanding and preprocessing

### Understanding

The notebook checks the shape, data types, summary statistics, missing values and duplicates. The target is numeric, so this is a **regression** problem.

### Cleaning (KDD step 2)

- Remove rows with missing values.
- Remove duplicate rows.
- Check outliers in the yield with the IQR rule: values outside `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]` are flagged. They are **kept**, because they are real high-yield crops such as potatoes and cassava.

### Exploratory analysis

Three graphs are produced:

1. **Average yield by crop.** Potatoes and cassava give the highest yields. Soybeans and sorghum give the lowest.
2. **Average yield over the years.** Yield rises steadily, from about 6.7 to about 9 tonnes per hectare.
3. **Correlation with yield.** All correlations are weak. Temperature is slightly negative, while year and pesticides are slightly positive.

Correlation shows association only. It does not prove that a factor causes a change in yield.

### Transformation (KDD step 3)

1. **Encoding.** Country and crop names are replaced by integer codes.
2. **Train/test split.** A random 80/20 split gives **20,745 training rows** and **5,187 test rows**.
3. **Standardisation (z-score)**, using the mean and standard deviation of the **training data only**, so the test data stays unseen:

   z = (x − μ) / σ

---

## 7. Predictive modelling approach

- **Problem type:** regression (predict a number, not a category).
- **k-NN:** similar inputs give similar yields. Predict by averaging the yields of the *k* most similar training records.
- **Random Forest:** many decision trees, each trained on a random sample of the data. Predict by averaging the trees' predictions.
- **Fair comparison:** both models use the same split, the same scaling and the same metrics.
- **From scratch:** both models are coded with NumPy maths. No ready-made model library is used.

---

## 8. Model development (the maths)

### 8.1 k-NN regression

**Euclidean distance** between a new point **a** and a training point **b** with *m* features:

d(a, b) = √( Σⱼ (aⱼ − bⱼ)² )

**Prediction:** take the *k* training points with the smallest distances (the neighbours Nₖ) and average their yields:

ŷ = (1 / k) · Σ yᵢ, for i in Nₖ

**Choosing k.** A validation set is cut from the training data (the last 15% of the training rows, using 400 of them). The values k ∈ {1, 3, 5, 7, 10, 15, 20, 30} are tried, and the k with the lowest validation RMSE is kept. **Best k = 1.**

### 8.2 Random Forest regression

**A single decision tree.** At every node, a feature *j* and threshold *t* split the data into Left (xⱼ ≤ t) and Right (xⱼ > t). The best split has the lowest weighted mean squared error:

MSE(S) = (1 / |S|) · Σ (yᵢ − ȳₛ)²

Cost = ( |L| · MSE(L) + |R| · MSE(R) ) / |S|

Candidate thresholds are the 10th, 20th, ... 90th percentiles of each feature. A node becomes a **leaf** (predicting the mean yield of its samples) when the maximum depth is reached, it has too few samples, or all its yields are identical.

**Making a forest.**

1. **Bootstrap sampling:** each tree is trained on rows drawn with replacement from the training data.
2. **Random feature subsets:** at every split only a random subset of the features is considered, which makes the trees differ from each other.
3. **Averaging:** with *T* trees, the final prediction is

   ŷ = (1 / T) · Σ ŷₜ

**Hyperparameters used**

| Parameter | Value |
|---|---|
| Number of trees | 20 |
| Maximum depth | 8 |
| Minimum samples to split | 10 |
| Features tried per split | 3 of 6 |
| Candidate thresholds per feature | 9 (percentiles 10 to 90) |

---

## 9. Evaluation of the models

With actual values yᵢ, predictions ŷᵢ, mean ȳ and *n* test rows:

| Metric | Formula | Reading |
|---|---|---|
| **MAE** | (1/n) · Σ \|yᵢ − ŷᵢ\| | Average error. Lower is better |
| **RMSE** | √( (1/n) · Σ (yᵢ − ŷᵢ)² ) | Punishes large errors. Lower is better |
| **R²** | 1 − Σ(yᵢ − ŷᵢ)² / Σ(yᵢ − ȳ)² | Share of variation explained. Closer to 1 is better |

All metrics are computed on the full test set (5,187 rows).

---

## 10. Results and interpretation

### 10.1 Model comparison (all three datasets)

| Model | MAE (hg/ha) | RMSE (hg/ha) | MAE (t/ha) | RMSE (t/ha) | R² |
|---|---|---|---|---|---|
| **k-NN (k = 1)** | 4,714 | 15,355 | 0.47 | 1.54 | **0.967** |
| Random Forest | 16,230 | 26,807 | 1.62 | 2.68 | 0.900 |

k-NN has the best score on every metric.

### 10.2 Graphs in the notebook

- **Model comparison:** R², MAE and RMSE side by side.
- **Actual vs predicted:** k-NN points sit close to the perfect-prediction line. Random Forest tends to under-predict the highest yields.
- **Error distribution:** k-NN errors are concentrated tightly around zero. Random Forest errors are more spread out.
- **Choosing k:** validation RMSE for each k.

### 10.3 Feature importance (permutation importance)

One feature at a time is shuffled, and the increase in RMSE is measured. A larger increase means the model relied more on that feature.

- **Crop** is the most important feature for **both** models.
- **Year** is the least important feature for both models.

### 10.4 Do the extra datasets help?

The same two models are re-run on four feature sets, with the same rows, split and k.

| Feature set | Features | k-NN R² | Random Forest R² |
|---|---|---|---|
| A. Yield only | Country, Crop, Year, Temperature | 0.782 | 0.819 |
| B. Yield + Rainfall | A + Rainfall | 0.956 | 0.850 |
| C. Yield + Pesticides | A + Pesticides | 0.883 | 0.839 |
| D. All 3 datasets | A + Rainfall + Pesticides | **0.967** | 0.900 |

Best R² rises from **0.819** (yield only) to **0.967** (all three datasets). Adding the rainfall and pesticide files clearly improved the predictions.

### 10.5 Knowledge discovered

1. Best model: k-NN with k = 1 (R² = 0.967).
2. Average error of the best model: 0.47 tonnes per hectare.
3. The most important feature for both models is the crop.
4. Rainfall and pesticide data improve prediction accuracy.

---

## 11. Predicting a new yield

Example from the notebook:

| Input | Value |
|---|---|
| Country | Argentina |
| Crop | Wheat |
| Year | 2012 |
| Rainfall | 876 mm |
| Pesticides | 45,698 tonnes |
| Temperature | 5 °C |

| Model | Predicted yield |
|---|---|
| k-NN | 3.58 t/ha |
| Random Forest | 5.55 t/ha |
| Average of both | 4.57 t/ha |
| Difference | 1.97 t/ha |

The new input is scaled with the **training** mean and standard deviation before it is passed to the models.

---

## 12. Conclusion

Following the KDD steps turned three raw files into a model that predicts crop yield. A from-scratch k-NN regressor reached R² = 0.967 with an average error of 0.47 t/ha, ahead of the from-scratch Random Forest (R² = 0.900). Crop type is the strongest driver of yield in both models, and adding rainfall and pesticide data raised the best R² from 0.819 to 0.967.

---


