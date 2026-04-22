# ============================================
# SYNTHETIC DATA EVALUATION PROJECT
# ============================================


# ============================================
# STEP 1: LOAD + PREPROCESS DATA
# ============================================

from src.preprocessing import preprocess, save_processed

df = preprocess("data/raw/adult.csv")
save_processed(df, "data/processed/cleaned_data.csv")

print("Preprocessing Done")


# ============================================
# STEP 2: SPLIT REAL DATA
# ============================================

from sklearn.model_selection import train_test_split

X = df.drop("income", axis=1)
y = df["income"]

X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Split Done")
print("Train size:", X_real_train.shape)
print("Test size:", X_real_test.shape)


# ============================================
# STEP 3: GENERATE SYNTHETIC DATA
# ============================================

from src.synthetic_generator import generate_synthetic

synthetic_data = generate_synthetic(df)
synthetic_data.to_csv("data/synthetic/synthetic_data.csv", index=False)

print("Synthetic Data Generated")


# ============================================
# STEP 4: PREPARE DATA (RAW + SCALED)
# ============================================

from sklearn.preprocessing import StandardScaler

X_syn_raw = synthetic_data.drop("income", axis=1)
y_syn = synthetic_data["income"]

scaler = StandardScaler()

X_real_train_scaled = scaler.fit_transform(X_real_train)
X_real_test_scaled = scaler.transform(X_real_test)
X_syn_scaled = scaler.transform(X_syn_raw)


# ============================================
# STEP 5: TRAIN MODELS
# ============================================

from src.model import train_models

models = train_models(
    X_real_train,
    y_real_train,
    X_real_test,
    y_real_test,
    X_syn_raw,
    y_syn,
    X_real_train_scaled,
    X_real_test_scaled,
    X_syn_scaled
)

print("Models Trained (Proper Scaling Applied)")


# ============================================
# UTILITY METRICS
# ============================================

from src.evaluation import evaluate_classification

print("\n================ UTILITY ================\n")

results = []

for name, model_pair in models.items():

    model_syn, X_test_syn = model_pair["synthetic"]
    model_real, X_test_real = model_pair["baseline"]

    y_pred_syn = model_syn.predict(X_test_syn)
    y_pred_real = model_real.predict(X_test_real)

    acc_syn, f1_syn = evaluate_classification(y_real_test, y_pred_syn)
    acc_real, f1_real = evaluate_classification(y_real_test, y_pred_real)

    results.append({
        "model": name,
        "tstr_acc": acc_syn,
        "baseline_acc": acc_real
    })

    print(f"Model: {name}")
    print(f"  TSTR     -> Accuracy: {acc_syn:.4f}, F1: {f1_syn:.4f}")
    print(f"  Baseline -> Accuracy: {acc_real:.4f}, F1: {f1_real:.4f}")
    print("-" * 50)


# ============================================
# FIDELITY METRICS
# ============================================

from src.evaluation import compute_dpcm, compute_ks, compute_wasserstein, compute_jsd

print("\n================ FIDELITY ================\n")

# DPCM
dpcm_value = compute_dpcm(X_real_train, X_syn_raw)
print(f"DPCM: {dpcm_value:.4f}")

# KS Test
print("\nKS Test (first few columns):")
ks_values = compute_ks(X_real_train, X_syn_raw)
for k, v in list(ks_values.items())[:5]:
    print(f"  {k}: {v:.4f}")

# Wasserstein
print("\nWasserstein Distance (first few columns):")
wd_values = compute_wasserstein(X_real_train, X_syn_raw)
for k, v in list(wd_values.items())[:5]:
    print(f"  {k}: {v:.4f}")

# JSD
print("\nJensen-Shannon Divergence (first few columns):")
jsd_values = compute_jsd(X_real_train, X_syn_raw)
for k, v in list(jsd_values.items())[:5]:
    print(f"  {k}: {v:.4f}")


# ============================================
# PRIVACY METRIC (DCR)
# ============================================

from src.evaluation import compute_dcr

print("\n================ PRIVACY (DCR) ================\n")

dcr_result = compute_dcr(X_real_train, X_syn_raw)

if isinstance(dcr_result, dict):
    print(f"DCR Mean: {dcr_result['dcr_mean']:.4f}")
    print(f"DCR Min: {dcr_result['dcr_min']:.4f}")
    print(f"DCR 5th Percentile: {dcr_result['dcr_5th_percentile']:.4f}")
else:
    print(f"DCR (Mean Distance): {dcr_result:.4f}")


# ============================================
# DETECTION METRIC
# ============================================

from src.evaluation import detection_metric

print("\n================ DETECTION ================\n")

det_acc = detection_metric(df, synthetic_data)
print(f"Detection Accuracy: {det_acc:.4f}")


# ============================================
# VISUALIZATION
# ============================================

import matplotlib.pyplot as plt
import numpy as np

models_names = [r["model"] for r in results]
tstr = [r["tstr_acc"] for r in results]
baseline = [r["baseline_acc"] for r in results]

x = np.arange(len(models_names))
width = 0.35

plt.figure()
plt.bar(x - width/2, tstr, width, label="TSTR")
plt.bar(x + width/2, baseline, width, label="Baseline")

plt.xticks(x, models_names)
plt.legend()

plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Comparison")

plt.savefig("results.png")

print("\nGraph saved as results.png")