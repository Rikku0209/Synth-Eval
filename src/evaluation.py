from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.spatial.distance import jensenshannon, cdist
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


# ============================================================
# UTILITY METRIC
# ============================================================
# Test: TSTR (Train on Synthetic, Test on Real)
# Measures how useful synthetic data is for ML tasks
# ============================================================

def evaluate_classification(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary')
    return acc, f1


# ============================================================
# FIDELITY METRICS
# ============================================================
# Measure how similar synthetic data is to real data
# ============================================================

# ------------------------------------------------------------
# Test: DPCM (Difference in  Pearson Correlation Matrix)
# Checks if relationships between features are preserved
# ------------------------------------------------------------

def compute_dpcm(X_real, X_syn):
    corr_real = X_real.corr()
    corr_syn = X_syn.corr()
    return (corr_real - corr_syn).abs().mean().mean()


# ------------------------------------------------------------
# Test: KS Test (Kolmogorov-Smirnov Test)
# Compares distributions of each numeric column
# ------------------------------------------------------------

def compute_ks(X_real, X_syn):
    ks_results = {}
    numeric_cols = X_real.select_dtypes(include=['int64', 'float64']).columns

    for col in numeric_cols:
        stat, _ = ks_2samp(X_real[col], X_syn[col])
        ks_results[col] = stat

    return ks_results


# ------------------------------------------------------------
# Test: Wasserstein Distance (Earth Mover's Distance)
# Measures how far two distributions are from each other
# ------------------------------------------------------------

def compute_wasserstein(X_real, X_syn):
    results = {}
    numeric_cols = X_real.select_dtypes(include=['int64', 'float64']).columns

    for col in numeric_cols:
        dist = wasserstein_distance(X_real[col], X_syn[col])
        results[col] = dist

    return results


# ------------------------------------------------------------
# Test: Jensen-Shannon Divergence (JSD)
# Measures similarity between probability distributions
# Symmetric and stable version of KL divergence
# ------------------------------------------------------------

def compute_jsd(X_real, X_syn, bins=50):
    jsd_results = {}

    numeric_cols = X_real.select_dtypes(include=['int64', 'float64']).columns

    for col in numeric_cols:
        real_vals = X_real[col].dropna()
        syn_vals = X_syn[col].dropna()

        # Create common bins
        min_val = min(real_vals.min(), syn_vals.min())
        max_val = max(real_vals.max(), syn_vals.max())

        hist_real, bin_edges = np.histogram(
            real_vals, bins=bins, range=(min_val, max_val), density=True
        )
        hist_syn, _ = np.histogram(
            syn_vals, bins=bin_edges, density=True
        )

        # Avoid zero issues
        hist_real += 1e-8
        hist_syn += 1e-8

        # Normalize distributions
        hist_real /= hist_real.sum()
        hist_syn /= hist_syn.sum()

        jsd = jensenshannon(hist_real, hist_syn)
        jsd_results[col] = jsd

    return jsd_results


# ============================================================
# PRIVACY METRIC (DCR)
# ============================================================
# Test: Distance to Closest Record (DCR)
# Measures risk of memorization / data leakage
# ============================================================

def compute_dcr(real_df, syn_df, exclude_cols=None):
    exclude_cols = exclude_cols or []
    
    real = real_df.drop(columns=exclude_cols, errors="ignore").copy()
    syn = syn_df.drop(columns=exclude_cols, errors="ignore").copy()

    # Keep only numeric columns
    real_num = real.select_dtypes(include=[np.number]).copy()
    syn_num = syn.select_dtypes(include=[np.number]).copy()

    # Align common columns
    common_cols = [c for c in real_num.columns if c in syn_num.columns]
    real_num = real_num[common_cols].fillna(0)
    syn_num = syn_num[common_cols].fillna(0)

    # Scale before distance computation
    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real_num)
    syn_scaled = scaler.transform(syn_num)

    # Compute pairwise distances
    dist_matrix = cdist(syn_scaled, real_scaled, metric="euclidean")
    nearest_distances = dist_matrix.min(axis=1)

    return {
        "dcr_mean": float(nearest_distances.mean()),
        "dcr_min": float(nearest_distances.min()),
        "dcr_5th_percentile": float(np.percentile(nearest_distances, 5)),
        "nearest_distances": nearest_distances
    }


# ============================================================
# DETECTION METRIC
# ============================================================
# Test: Real vs Synthetic Classifier
# Measures how easily synthetic data can be distinguished
# ============================================================

def detection_metric(real_df, syn_df):
    real = real_df.copy()
    real['label'] = 0

    syn = syn_df.copy()
    syn['label'] = 1

    combined = pd.concat([real, syn])

    X = combined.drop('label', axis=1)
    y = combined['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    return clf.score(X_test, y_test)