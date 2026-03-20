"""
Colab Code Templates
=====================
Reusable Python code snippets that get injected into Colab cells.
These are the "building blocks" that agents compose into complete workflows.

Each template is a string that may contain {placeholders} for dynamic values.
Templates output structured JSON on their last line for machine parsing.
"""

# ===================================================================
# EDA Templates
# ===================================================================

EDA_TEMPLATES = {

"load_and_inspect": '''
import pandas as pd
import numpy as np
import os, json, glob

# Find data files
data_dir = "/content/data"
csv_files = glob.glob(os.path.join(data_dir, "**/*.csv"), recursive=True)
parquet_files = glob.glob(os.path.join(data_dir, "**/*.parquet"), recursive=True)

print(f"Found files: {csv_files + parquet_files}")

# Load train/test
train_df = None
test_df = None
for f in csv_files + parquet_files:
    name = os.path.basename(f).lower()
    if "train" in name:
        train_df = pd.read_csv(f) if f.endswith(".csv") else pd.read_parquet(f)
    elif "test" in name:
        test_df = pd.read_csv(f) if f.endswith(".csv") else pd.read_parquet(f)

if train_df is None and csv_files:
    train_df = pd.read_csv(csv_files[0])

# Classify columns
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = train_df.select_dtypes(include=["object", "category"]).columns.tolist()

# Detect text columns (categorical with high cardinality and long strings)
text_cols = []
for col in categorical_cols:
    if train_df[col].nunique() > 100:
        avg_len = train_df[col].dropna().astype(str).str.len().mean()
        if avg_len > 50:
            text_cols.append(col)

print("\\n=== DATA INFO ===")
print(f"Train shape: {train_df.shape}")
if test_df is not None:
    print(f"Test shape: {test_df.shape}")
print(f"\\nDtypes:\\n{train_df.dtypes.value_counts()}")
print(f"\\nFirst 5 rows:\\n{train_df.head()}")

result = {
    "n_rows": len(train_df),
    "n_cols": len(train_df.columns),
    "n_train": len(train_df),
    "n_test": len(test_df) if test_df is not None else 0,
    "columns": train_df.columns.tolist(),
    "numeric_cols": numeric_cols,
    "categorical_cols": [c for c in categorical_cols if c not in text_cols],
    "text_cols": text_cols,
    "dtypes": train_df.dtypes.astype(str).to_dict(),
}
print(json.dumps(result))
''',

"missing_values": '''
import json
missing = train_df.isnull().sum()
missing_pct = (missing / len(train_df) * 100).round(2)
missing_info = missing_pct[missing_pct > 0].sort_values(ascending=False).to_dict()

print("\\n=== MISSING VALUES ===")
if missing_info:
    for col, pct in missing_info.items():
        bar = "█" * int(pct // 2)
        print(f"  {col:<30} {pct:>6.1f}%  {bar}")
else:
    print("  No missing values!")

print(json.dumps(missing_info))
''',

"target_distribution": '''
import json
import matplotlib.pyplot as plt

target_col = "{target}"

# Auto-detect target if not specified
if target_col == "" or target_col not in train_df.columns:
    # Heuristic: last column, or common target names
    candidates = ["target", "label", "y", "class", "outcome", "survived",
                   "SalePrice", "price", "default", "fraud"]
    for c in candidates:
        if c in train_df.columns:
            target_col = c
            break
    if target_col == "" or target_col not in train_df.columns:
        target_col = train_df.columns[-1]

y = train_df[target_col]

# Determine task type
if y.dtype in ["object", "category"] or y.nunique() <= 20:
    task_type = "binary_classification" if y.nunique() == 2 else "multiclass_classification"
    dist = y.value_counts(normalize=True).to_dict()
    imbalance_ratio = y.value_counts().max() / y.value_counts().min()

    fig, ax = plt.subplots(figsize=(8, 4))
    y.value_counts().plot(kind="bar", ax=ax)
    ax.set_title(f"Target Distribution: {target_col}")
    plt.tight_layout()
    plt.savefig("/content/plots/target_distribution.png", dpi=100)
    plt.show()

    result = {
        "column": target_col,
        "task_type": task_type,
        "n_classes": int(y.nunique()),
        "distribution": {str(k): round(v, 4) for k, v in dist.items()},
        "imbalance_ratio": round(float(imbalance_ratio), 2),
    }
else:
    task_type = "regression"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    y.hist(bins=50, ax=axes[0])
    axes[0].set_title(f"Distribution of {target_col}")
    import numpy as np
    np.log1p(y).hist(bins=50, ax=axes[1])
    axes[1].set_title(f"Log Distribution of {target_col}")
    plt.tight_layout()
    plt.savefig("/content/plots/target_distribution.png", dpi=100)
    plt.show()

    result = {
        "column": target_col,
        "task_type": task_type,
        "mean": round(float(y.mean()), 4),
        "median": round(float(y.median()), 4),
        "std": round(float(y.std()), 4),
        "skew": round(float(y.skew()), 4),
        "min": round(float(y.min()), 4),
        "max": round(float(y.max()), 4),
    }

print(json.dumps(result))
''',

"numerical_distributions": '''
import matplotlib.pyplot as plt
import seaborn as sns

num_cols = train_df.select_dtypes(include=["number"]).columns.tolist()
# Exclude target and ID columns
num_cols = [c for c in num_cols if c.lower() not in ["id", "index"]]

n = min(len(num_cols), 20)  # Plot at most 20
if n > 0:
    cols_to_plot = num_cols[:n]
    n_rows = (n + 3) // 4
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 3 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n == 1 else axes.flatten()

    for i, col in enumerate(cols_to_plot):
        train_df[col].hist(bins=30, ax=axes[i], alpha=0.7)
        axes[i].set_title(col, fontsize=9)
        axes[i].tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Numerical Feature Distributions", y=1.01)
    plt.tight_layout()
    plt.savefig("/content/plots/numerical_distributions.png", dpi=100)
    plt.show()

# Statistics summary
stats = train_df[num_cols].describe().round(3).to_dict()
print("Numerical stats computed for", len(num_cols), "columns")
''',

"categorical_analysis": '''
import matplotlib.pyplot as plt

cat_cols = train_df.select_dtypes(include=["object", "category"]).columns.tolist()
cat_cols = [c for c in cat_cols if c.lower() not in ["id", "name"]]

n = min(len(cat_cols), 12)
if n > 0:
    cols_to_plot = cat_cols[:n]
    n_rows = (n + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 3 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n == 1 else axes.flatten()

    for i, col in enumerate(cols_to_plot):
        top_vals = train_df[col].value_counts().head(10)
        top_vals.plot(kind="barh", ax=axes[i])
        axes[i].set_title(f"{col} (nunique={train_df[col].nunique()})", fontsize=9)
        axes[i].tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Categorical Feature Analysis", y=1.01)
    plt.tight_layout()
    plt.savefig("/content/plots/categorical_analysis.png", dpi=100)
    plt.show()

print(f"Analyzed {len(cat_cols)} categorical columns")
for col in cat_cols:
    print(f"  {col}: {train_df[col].nunique()} unique values")
''',

"correlations": '''
import json
import matplotlib.pyplot as plt
import seaborn as sns

num_df = train_df.select_dtypes(include=["number"])
if len(num_df.columns) > 1:
    corr = num_df.corr()

    # Plot correlation heatmap
    plt.figure(figsize=(min(20, len(corr.columns)), min(16, len(corr.columns) * 0.8)))
    mask = None
    if len(corr.columns) > 30:
        # For large matrices, show only top correlations
        target_col_name = "{target}" if "{target}" in corr.columns else corr.columns[-1]
        top_corr_cols = corr[target_col_name].abs().sort_values(ascending=False).head(20).index
        corr = corr.loc[top_corr_cols, top_corr_cols]

    sns.heatmap(corr, annot=len(corr.columns) <= 15, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, linewidths=0.5)
    plt.title("Feature Correlations")
    plt.tight_layout()
    plt.savefig("/content/plots/correlations.png", dpi=100)
    plt.show()

    # Extract target correlations
    target_corrs = {}
    for col in corr.columns:
        if col != corr.columns[-1]:
            target_corrs[col] = round(float(corr.iloc[-1][col]), 4)
    target_corrs = dict(sorted(target_corrs.items(), key=lambda x: abs(x[1]), reverse=True))
    print(json.dumps(target_corrs))
else:
    print(json.dumps({}))
''',

"feature_importance": '''
import json
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Prepare data (quick & dirty for importance estimation)
features = train_df.select_dtypes(include=["number"]).drop(
    columns=[c for c in train_df.columns if c.lower() in ["id", "index"]],
    errors="ignore"
)

# Determine target
target_col = features.columns[-1]  # Will be overridden by actual target
y = train_df[target_col]
X = features.drop(columns=[target_col], errors="ignore")

# Handle missing values simply
X = X.fillna(X.median())

# Quick RF for importance
if y.dtype in ["object", "category"] or y.nunique() <= 20:
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X, y_enc)
else:
    rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X, y)

importances = dict(zip(X.columns, rf.feature_importances_.round(4)))
importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

# Plot
top_n = min(20, len(importances))
top_feats = dict(list(importances.items())[:top_n])
plt.figure(figsize=(10, max(4, top_n * 0.3)))
plt.barh(list(top_feats.keys())[::-1], list(top_feats.values())[::-1])
plt.title("Top Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig("/content/plots/feature_importance.png", dpi=100)
plt.show()

print(json.dumps(importances))
''',
}


# ===================================================================
# Training Templates
# ===================================================================

TRAINING_TEMPLATES = {

"baseline_classification": '''
import json, time
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import {metric_func}
{model_imports}

# Load prepared data
X = pd.read_pickle("/content/data/X_train_processed.pkl")
y = pd.read_pickle("/content/data/y_train.pkl")

# Encode target if needed
if y.dtype == "object":
    le = LabelEncoder()
    y = le.fit_transform(y)

# Initialize model
model = {model_init}

# Cross-validation
start = time.time()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="{scoring}", n_jobs=-1)
elapsed = time.time() - start

result = {{
    "model_name": "{model_name}",
    "cv_mean": round(float(scores.mean()), 5),
    "cv_std": round(float(scores.std()), 5),
    "cv_scores": scores.round(5).tolist(),
    "training_time": round(elapsed, 1),
}}
print(json.dumps(result))
''',

"baseline_regression": '''
import json, time
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
{model_imports}

X = pd.read_pickle("/content/data/X_train_processed.pkl")
y = pd.read_pickle("/content/data/y_train.pkl")

model = {model_init}

start = time.time()
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="{scoring}", n_jobs=-1)
elapsed = time.time() - start

result = {{
    "model_name": "{model_name}",
    "cv_mean": round(float(scores.mean()), 5),
    "cv_std": round(float(scores.std()), 5),
    "cv_scores": scores.round(5).tolist(),
    "training_time": round(elapsed, 1),
}}
print(json.dumps(result))
''',

"optuna_tuning": '''
import json, time, optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
optuna.logging.set_verbosity(optuna.logging.WARNING)

X = pd.read_pickle("/content/data/X_train_processed.pkl")
y = pd.read_pickle("/content/data/y_train.pkl")

def objective(trial):
    {optuna_search_space}

    model = {model_init_with_params}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="{scoring}", n_jobs=-1)
    return scores.mean()

start = time.time()
study = optuna.create_study(direction="{direction}")
study.optimize(objective, n_trials={n_trials}, timeout={timeout})
elapsed = time.time() - start

result = {{
    "best_score": round(float(study.best_value), 5),
    "best_params": study.best_params,
    "n_trials_completed": len(study.trials),
    "tuning_time": round(elapsed, 1),
}}
print(json.dumps(result))
''',

"generate_submission": '''
import pandas as pd
import numpy as np
{model_imports}

# Load data
X_train = pd.read_pickle("/content/data/X_train_processed.pkl")
y_train = pd.read_pickle("/content/data/y_train.pkl")
X_test = pd.read_pickle("/content/data/X_test_processed.pkl")
sample_sub = pd.read_csv("/content/data/sample_submission.csv")

# Train final model on full training data
model = {model_init}
model.fit(X_train, y_train)

# Generate predictions
preds = model.predict(X_test)
{post_process}

# Create submission
sample_sub["{target_col}"] = preds
sample_sub.to_csv("/content/submissions/submission.csv", index=False)
print(f"Submission shape: {sample_sub.shape}")
print(sample_sub.head())
print("SUBMISSION_READY")
''',
}


# ===================================================================
# Feature Engineering Templates
# ===================================================================

FEATURE_ENGINEERING_TEMPLATES = {

"preprocess_pipeline": '''
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from category_encoders import TargetEncoder
import json

train_df = pd.read_csv("/content/data/train.csv")
test_df = pd.read_csv("/content/data/test.csv")

target_col = "{target}"
y = train_df[target_col].copy()
X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

# Drop ID columns
id_cols = [c for c in X_train.columns if c.lower() in ["id", "index", "unnamed: 0"]]
X_train = X_train.drop(columns=id_cols, errors="ignore")
X_test = X_test.drop(columns=id_cols, errors="ignore")

# Separate feature types
num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

# Fill missing - numeric
for col in num_cols:
    median_val = X_train[col].median()
    X_train[col] = X_train[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)

# Fill missing - categorical
for col in cat_cols:
    X_train[col] = X_train[col].fillna("MISSING")
    X_test[col] = X_test[col].fillna("MISSING")

# Encode categoricals
if cat_cols:
    if y.dtype in ["object", "category"] or y.nunique() <= 20:
        # Label encode for tree models
        for col in cat_cols:
            le = LabelEncoder()
            le.fit(pd.concat([X_train[col], X_test[col]]).astype(str))
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
    else:
        te = TargetEncoder(cols=cat_cols, smoothing=1.0)
        X_train[cat_cols] = te.fit_transform(X_train[cat_cols], y)
        X_test[cat_cols] = te.transform(X_test[cat_cols])

# Save processed data
X_train.to_pickle("/content/data/X_train_processed.pkl")
X_test.to_pickle("/content/data/X_test_processed.pkl")
y.to_pickle("/content/data/y_train.pkl")

print(f"Processed: X_train={X_train.shape}, X_test={X_test.shape}")
print(json.dumps({{"n_features": X_train.shape[1], "features": X_train.columns.tolist()}}))
''',

"advanced_features": '''
import pandas as pd
import numpy as np

X_train = pd.read_pickle("/content/data/X_train_processed.pkl")
X_test = pd.read_pickle("/content/data/X_test_processed.pkl")

# --- Feature interactions ---
num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
top_features = {top_features}  # Injected from feature importance

# Create interactions for top features
for i, f1 in enumerate(top_features[:5]):
    for f2 in top_features[i+1:6]:
        X_train[f"{{f1}}_x_{{f2}}"] = X_train[f1] * X_train[f2]
        X_test[f"{{f1}}_x_{{f2}}"] = X_test[f1] * X_test[f2]
        X_train[f"{{f1}}_div_{{f2}}"] = X_train[f1] / (X_train[f2] + 1e-8)
        X_test[f"{{f1}}_div_{{f2}}"] = X_test[f1] / (X_test[f2] + 1e-8)

# --- Aggregation features ---
{aggregation_code}

# Save enhanced features
X_train.to_pickle("/content/data/X_train_enhanced.pkl")
X_test.to_pickle("/content/data/X_test_enhanced.pkl")

print(f"Enhanced features: {{X_train.shape[1]}} (was {{len(num_cols)}})")
''',
}
