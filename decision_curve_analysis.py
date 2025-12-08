import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
import streamlit as st

matplotlib.use("Agg")  # for Streamlit


# -------------------------
#   Data generation
# -------------------------
@st.cache_data
def generate_distributions(auc, prevalence=0.5, n=2000):
    d_prime = np.sqrt(2) * norm.ppf(auc)
    mu_pos = d_prime
    mu_neg = 0.0
    sigma = 1.0

    n_pos = int(n * prevalence)
    n_neg = n - n_pos

    pos = np.random.normal(loc=mu_pos, scale=sigma, size=n_pos)
    neg = np.random.normal(loc=mu_neg, scale=sigma, size=n_neg)

    return pos, neg


# -------------------------
#   DCA helpers
# -------------------------
def net_benefit(prevalence, sensitivity, specificity, pt):
    return (
        prevalence * sensitivity
        - (1.0 - prevalence) * (1.0 - specificity) * pt / (1.0 - pt)
    )


def compute_dca_curves(y_true, risk_pred, prevalence, n_points=200):
    pts = np.linspace(0.01, 0.80, n_points)

    nb_none = np.zeros_like(pts)
    nb_all = prevalence - (1.0 - prevalence) * pts / (1.0 - pts)

    nb_model = np.zeros_like(pts)
    for i, pt in enumerate(pts):
        preds = (risk_pred >= pt).astype(int)

        TP = np.sum((preds == 1) & (y_true == 1))
        FP = np.sum((preds == 1) & (y_true == 0))
        FN = np.sum((preds == 0) & (y_true == 1))
        TN = np.sum((preds == 0) & (y_true == 0))

        sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0

        nb_model[i] = net_benefit(prevalence, sens, spec, pt)

    return pts, nb_all, nb_none, nb_model


def metrics_at_threshold(y_true, risk_pred, pt):
    preds = (risk_pred >= pt).astype(int)

    TP = np.sum((preds == 1) & (y_true == 1))
    FP = np.sum((preds == 1) & (y_true == 0))
    FN = np.sum((preds == 0) & (y_true == 1))
    TN = np.sum((preds == 0) & (y_true == 0))

    sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    return sens, spec, sens, 1.0 - spec


# NEW: Confusion matrix helper
def compute_confusion_matrix(y_true, risk_pred, pt):
    preds = (risk_pred >= pt).astype(int)

    TP = int(np.sum((preds == 1) & (y_true == 1)))
    FP = int(np.sum((preds == 1) & (y_true == 0)))
    FN = int(np.sum((preds == 0) & (y_true == 1)))
    TN = int(np.sum((preds == 0) & (y_true == 0)))

    return TN, FP, FN, TP


# -------------------------
#   Plotting functions
# -------------------------
def plot_roc(y_true, risk_pred, pt, fpr_pt, tpr_pt):
    fpr, tpr, _ = roc_curve(y_true, risk_pred)
    auc_val = roc_auc_score(y_true, risk_pred)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")

    ax.scatter(fpr_pt, tpr_pt, s=60, color="red", zorder=3)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("ROC Curve")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend()
    st.pyplot(fig)


def plot_dca(pts, nb_all, nb_none, nb_model, prevalence, pt, nb_model_at_pt):
    fig, ax = plt.subplots(figsize=(4, 4))

    ax.plot(pts, nb_all, label="Treat all", linewidth=2)
    ax.plot(pts, nb_none, label="Treat none", linestyle="--")
    ax.plot(pts, nb_model, label="Model", linewidth=2)

    ax.scatter(pt, nb_model_at_pt, s=60, color="red", zorder=3)

    ax.set_ylim(-0.02, max(nb_model.max(), nb_all.max(), 0.1))
    ax.set_xlim(0, pts.max())
    ax.axhline(0, linestyle=":")
    ax.set_title(f"DCA (Prevalence = {prevalence:.2f})")
    ax.set_xlabel("Threshold probability pₜ")
    ax.set_ylabel("Net benefit")
    ax.legend()
    st.pyplot(fig)


# -------------------------
#   Streamlit UI
# -------------------------
st.set_page_config(layout="wide")
st.title("ROC + DCA Visualizer (with Confusion Matrix)")

st.sidebar.header("Settings")

auc = st.sidebar.slider("Desired AUC", 0.50, 0.99, 0.80, 0.01)
prevalence = st.sidebar.slider("Prevalence", 0.01, 0.99, 0.20, 0.01)
n_samples = st.sidebar.slider("Number of samples", 500, 10000, 3000, 500)

threshold_pt = st.sidebar.slider("Decision threshold (pₜ)", 0.01, 0.80, 0.20, 0.01)

# Generate data
pos, neg = generate_distributions(auc, prevalence, n_samples)
y_true = np.array([1] * len(pos) + [0] * len(neg))
scores = np.concatenate([pos, neg])

# Map scores → probabilities
log_reg = LogisticRegression(solver="lbfgs")
log_reg.fit(scores.reshape(-1, 1), y_true)
risk_pred = log_reg.predict_proba(scores.reshape(-1, 1))[:, 1]

# Metrics at chosen threshold
sens_pt, spec_pt, tpr_pt, fpr_pt = metrics_at_threshold(y_true, risk_pred, threshold_pt)

nb_model_at_pt = net_benefit(prevalence, sens_pt, spec_pt, threshold_pt)

# Full DCA curves
pts, nb_all, nb_none, nb_model = compute_dca_curves(y_true, risk_pred, prevalence)

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("ROC Curve")
    plot_roc(y_true, risk_pred, threshold_pt, fpr_pt, tpr_pt)

with col2:
    st.subheader("Decision Curve")
    plot_dca(pts, nb_all, nb_none, nb_model, prevalence, threshold_pt, nb_model_at_pt)

# ---- Confusion matrix ----
TN, FP, FN, TP = compute_confusion_matrix(y_true, risk_pred, threshold_pt)

st.markdown("## Confusion Matrix")
st.table({
    "": ["Predicted Negative", "Predicted Positive"],
    "True Negative": [TN, FP],
    "True Positive": [FN, TP],
})

