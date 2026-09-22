# Heart Disease UCI - Exploratory Data Analysis

Series 1 of a 3-week data analysis project. This week covers importing and
inspecting a dataset, basic filtering/grouping, an initial ML exploration,
and one visualization.

## Project Goal

Explore the classic **Heart Disease UCI** dataset to understand which
patient attributes (age, sex, chest pain type, cholesterol, max heart rate,
etc.) are associated with a heart disease diagnosis, and take a first pass
at predicting diagnosis from those attributes with a simple ML model.

## Data Source

- **Dataset**: Heart Disease UCI (Cleveland database subset), 303 patient
  records, 14 attributes.
- **Original source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease),
  also widely mirrored on Kaggle (e.g. `navjotkaushal/heart-disease-uci-dataset`).
- **File used**: `heart.csv`, pulled from a public GitHub mirror of the
  cleaned CSV: https://github.com/sharmaroshan/Heart-UCI-Dataset

| Column | Meaning |
|---|---|
| age | age in years |
| sex | 1 = male, 0 = female |
| cp | chest pain type (0-3) |
| trestbps | resting blood pressure (mm Hg) |
| chol | serum cholesterol (mg/dl) |
| fbs | fasting blood sugar > 120 mg/dl (1 = true) |
| restecg | resting ECG results (0-2) |
| thalach | max heart rate achieved |
| exang | exercise-induced angina (1 = yes) |
| oldpeak | ST depression induced by exercise |
| slope | slope of the peak exercise ST segment |
| ca | number of major vessels colored by fluoroscopy (0-4) |
| thal | thalassemia indicator (0-3) |
| target | 1 = heart disease present, 0 = absent |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python analysis.py
```

The script reads `heart.csv` from the project root, so run it from inside
this folder.

## Analysis Steps

1. **Import** - load `heart.csv` with Pandas.
2. **Inspect** - `.head()`, `.info()`, `.describe()`, and checks for missing
   values and duplicate rows.
3. **Filter & group** - filter patients over 50 who were diagnosed with
   heart disease; group by `sex` and by chest pain type (`cp`) to compare
   average age, cholesterol, max heart rate, and disease rate.
4. **ML exploration** - a `LogisticRegression` classifier (scikit-learn)
   trained on all 13 features to predict `target`, with an 80/20
   train/test split and standardized inputs.
5. **Visualization** - a scatter plot of age vs. max heart rate achieved,
   colored by diagnosis.
6. **Bonus: Pandas vs Polars** - the same read + de-duplicate + group-by
   pipeline timed in both libraries.

## Outcomes / Findings

- The dataset is clean: **no missing values**, and only **1 duplicate row**
  (dropped before analysis, leaving 302 rows).
- **32.8%** of patients are over 50 and diagnosed with heart disease.
- Grouped by sex: women in this sample have a notably higher disease rate
  (**75%**) than men (**45%**), despite similar average age.
- Grouped by chest pain type: patients with **atypical angina** or
  **non-anginal pain** have much higher disease rates (~80%) than those
  with **typical angina** (27%) - counter-intuitive at first glance, but
  consistent with this being a well-known quirk of the dataset (classic
  "typical angina" is actually the least common presentation among
  diagnosed cases here).
- The logistic regression model reached **~79% accuracy** on the held-out
  test set. The most influential features (by standardized coefficient
  magnitude) were chest pain type (`cp`), `sex`, ST depression (`oldpeak`),
  max heart rate (`thalach`), and `thal`.
- Polars completed the equivalent read/dedupe/group-by pipeline roughly
  **3-4x faster** than Pandas in a 50-iteration timing loop. The gap is
  modest here because the dataset is tiny (302 rows); Polars' multithreaded
  Rust engine tends to show a bigger advantage as data size grows.

See `age_vs_heartrate_by_diagnosis.png` for the visualization.

## Files

- `analysis.py` - full analysis script (run this)
- `heart.csv` - dataset
- `age_vs_heartrate_by_diagnosis.png` - output visualization
- `requirements.txt` - Python dependencies
