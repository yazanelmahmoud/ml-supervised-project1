# Project 1 — Adult Income (Census)

Binary classification: predict income **<=50K** vs **>50K**. Class imbalance ~3:1.  
**Metrics:** Accuracy, F1, PR-AUC (average_precision).  
**Algorithms:** Decision Tree, kNN, SVM, Neural Network (sklearn + PyTorch).

---

## Data & Config (don’t forget)

- **Data file:** `adult.csv` (path in `config.DATA_PATH`).
- **Target column:** `config.TARGET_COLUMN` = `"class"` (values `"<=50K"` / `">50K"`).
- **Reproducibility:** `config.RANDOM_SEED = 42`, `config.TEST_SIZE = 0.2`.
- **Loading:** `data_loading.load_adult(path=...)` — loads CSV, treats `?` as NaN, removes exact duplicate rows (keeps 45,175 rows).
- **Train/test:** Stratified split; **always use `preprocessing.get_dataset()`** for consistent preprocessed train/test (do not roll your own split/preprocess elsewhere).

---

## Preprocessing (important)

Implemented in **`preprocessing.py`**. EDA-driven; see `EDA_SUMMARY.txt` for rationale.

### 1. Drops (before encoding)

- **`education`** — redundant with `education-num` (1:1 mapping); keep ordinal `education-num`.
- **`fnlwgt`** — sampling weight; near-zero correlation with target (-0.007); dropped to reduce noise.

### 2. Target encoding

- **`encode_target(y)`:** `<=50K` → 0, `>50K` → 1 (int32). Handles strip/lower.

### 3. Feature encoding

- **Numeric (kept):** `age`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`.
- **One-hot (drop first):** `workclass`, `marital-status`, `occupation`, `relationship`, `race`, `sex`.  
  `OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")`.
- **Target encoding (fit on train only):** `native-country` (41 levels, sparse). Each category → mean target on train; unseen → global mean. Prevents leakage.

### 4. Scaling

- **StandardScaler** fit on **numeric features only** (on training set), then transform train and test.

### 5. Feature matrix order

Final **X** is: `[numeric_scaled | one_hot_cols | target_encoded_native_country]`, float32.

### 6. Entry points (use these)

- **`get_dataset(path=None, test_size=None, random_state=None)`**  
  Load → stratified train/test split → preprocess.  
  **Returns:** `X_train, y_train, X_test, y_test` (use everywhere: notebooks, model scripts).

- **`get_preprocessed_train_test(X_train_raw, y_train_raw, X_test_raw, y_test_raw)`**  
  When you already have raw train/test (e.g. from a custom split): fit on train, transform both.

- **`prepare_X_y(df, target_encodings=None, scaler=None, ohe=None, fit=True)`**  
  Low-level: one DataFrame in → X, y, and (scaler, ohe, target_encodings).  
  `fit=True` for train (fit encoders/scaler), `fit=False` for test (pass fitted objects).

**Critical:** For test (or any data that didn’t fit the encoders), always pass the same `scaler`, `ohe`, and `target_encodings` that were fitted on the training set.

---

## File roles

| File | Role |
|------|------|
| `config.py` | RANDOM_SEED, DATA_PATH, TARGET_COLUMN, TEST_SIZE, METRICS_NAMES, OUTPUT_DIR |
| `data_loading.py` | `load_adult()`, `get_target_and_features()` |
| `preprocessing.py` | Drops, encodings, scaling, **`get_dataset()`** |
| `eda.py` | EDA scripts; plots → `outputs/` |
| `EDA_SUMMARY.txt` | EDA findings and preprocessing rationale |
| `evaluation.py` | Metrics (accuracy, F1, PR-AUC) |
| `models_*.py` | DT, kNN, SVM, NN (sklearn + PyTorch) |

---

## Quick start for modeling

```python
from preprocessing import get_dataset

X_train, y_train, X_test, y_test = get_dataset()
# Then fit on X_train, y_train; predict on X_test; evaluate with evaluation.py metrics.
```

No manual scaling or encoding elsewhere — preprocessing is the single source of truth.
