import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibrationDisplay
import streamlit as st

matplotlib.use("Agg")  # for Streamlit

# -------------------------
#   COLOR CONFIGURATION
# -------------------------
COLOR_M1 = "#8F0177"  # Model 1
COLOR_M2 = "#84994F"  # Model 2
COLOR_M3 = "#F87B1B"  # Model 3
COLOR_MAIN = "#9CC6DB"  # Main Model
COLOR_ACCENT = "#FF4B4B"  # Threshold lines

# -------------------------
#   Streamlit Page Config
# -------------------------
st.set_page_config(layout="wide", page_title="DCA Visualizer")


# -------------------------
#   Data Generation
# -------------------------
@st.cache_data
def generate_data_for_model(auc, prevalence, n_total):
    d_prime = np.sqrt(2) * norm.ppf(auc)
    mu_pos = d_prime
    mu_neg = 0.0
    sigma = 1.0

    n_pos = int(n_total * prevalence)
    n_neg = n_total - n_pos

    pos_scores = np.random.normal(loc=mu_pos, scale=sigma, size=n_pos)
    neg_scores = np.random.normal(loc=mu_neg, scale=sigma, size=n_neg)

    X = np.concatenate([pos_scores, neg_scores]).reshape(-1, 1)
    y = np.array([1] * len(pos_scores) + [0] * len(neg_scores))

    # Shuffle
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    lr = LogisticRegression(solver="lbfgs")
    lr.fit(X, y)

    return X, y, lr


def get_probabilities(log_reg, X, calibration_factor):
    logits = log_reg.decision_function(X)
    scaled_logits = logits * calibration_factor
    probs = 1 / (1 + np.exp(-scaled_logits))
    return probs


# -------------------------
#   DCA Helpers
# -------------------------
def net_benefit(prevalence, sens, spec, pt, test_harm=0.0):
    """
    Calculates Net Benefit, optionally subtracting test harm.
    NB = (Sens * Prev) - ((1 - Spec) * (1 - Prev) * (pt / (1 - pt))) - Test Harm
    """
    nb = (prevalence * sens) - ((1 - prevalence) * (1 - spec) * (pt / (1 - pt)))
    return nb - test_harm


def compute_dca_curve(y_true, probs, prevalence, test_harm=0.0):
    pts = np.linspace(0.01, 0.99, 100)
    nb_vals = []
    for pt in pts:
        preds = (probs >= pt).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))
        tn = np.sum((preds == 0) & (y_true == 0))

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Pass test_harm to calculation
        nb = net_benefit(prevalence, sens, spec, pt, test_harm)
        nb_vals.append(nb)
    return pts, np.array(nb_vals)


def get_treat_all_none(prevalence):
    pts = np.linspace(0.01, 0.99, 100)
    # Treat All NB does not depend on model threshold, but technically if there is test harm
    # associated with the *test*, Treat All (assuming no test is done) has 0 test harm.
    # However, usually DCA compares "Model Strategy" vs "Treat All Strategy".
    # If "Treat All" implies giving everyone the treatment without testing, Test Harm is 0.
    nb_all = prevalence - (1.0 - prevalence) * pts / (1.0 - pts)
    nb_none = np.zeros_like(pts)
    return pts, nb_all, nb_none


def calculate_metrics_at_pt(y_true, probs, pt):
    preds = (probs >= pt).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return sens, spec, tn, fp, fn, tp


# -------------------------
#   Plotting Functions
# -------------------------
def plot_combined_dca_kde(model_data, prevalence, threshold_pt, test_harm=0.0):
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(7, 8),
                                   gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.3})

    # --- Top Plot: DCA ---
    pts, nb_all, nb_none = get_treat_all_none(prevalence)
    ax1.plot(pts, nb_all, label="Treat all", color='black', linewidth=1.5, linestyle='--')
    ax1.plot(pts, nb_none, label="Treat none", color='black', linewidth=1, linestyle='-')

    # Compute DCA with Harm
    pts_model, nb_model = compute_dca_curve(
        model_data['y_true'],
        model_data['probs'],
        prevalence,
        test_harm
    )

    ax1.plot(pts_model, nb_model, color=COLOR_MAIN, linewidth=2.5, label='Model')

    ax1.axvline(threshold_pt, color=COLOR_ACCENT, linestyle="--", alpha=1)

    # Adjust Y-lim to handle potential drops due to harm
    y_min = -0.05
    if test_harm > 0:
        y_min = min(nb_model.min(), -0.05)

    ax1.set_ylim(y_min, prevalence + 0.1)
    ax1.set_xlim(0, 1)
    ax1.set_ylabel("Net Benefit")
    ax1.set_xlabel("Threshold Probability")
    ax1.set_title("Decision Curve Analysis")
    ax1.legend(loc='upper right', fontsize='small')
    ax1.grid(True, alpha=0.1)

    # --- Bottom Plot: KDE ---
    probs = model_data['probs']
    y = model_data['y_true']
    x_grid = np.linspace(0, 1, 200)

    if np.sum(y == 0) > 1:
        kde_neg = gaussian_kde(probs[y == 0])
        ax2.plot(x_grid, kde_neg(x_grid), color="#234C6A", linestyle="-", label="Negatives")
        ax2.fill_between(x_grid, kde_neg(x_grid), color="#234C6A", alpha=0.5)

    if np.sum(y == 1) > 1:
        kde_pos = gaussian_kde(probs[y == 1])
        ax2.plot(x_grid, kde_pos(x_grid), color="#94B4C1", linestyle="-", label="Positives")
        ax2.fill_between(x_grid, kde_pos(x_grid), color="#94B4C1", alpha=0.5)

    ax2.axvline(threshold_pt, color=COLOR_ACCENT, linestyle="--", linewidth=1.5, label=f"Threshold {threshold_pt:.2f}")
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Density")
    ax2.set_yticks([])
    ax2.legend(fontsize='x-small', loc='upper center')

    st.pyplot(fig)


def display_nb_calc_detailed(y_true, probs, prev, pt, test_harm=0.0, use_harm=False):
    sens, spec, tn, fp, fn, tp = calculate_metrics_at_pt(y_true, probs, pt)
    nb = net_benefit(prev, sens, spec, pt, test_harm)
    weight = pt / (1 - pt)

    term_tp = sens * prev
    term_fp = (1 - spec) * (1 - prev) * weight

    st.markdown("##### Net Benefit Formula")

    if use_harm:
        # Formula with Test Harm
        st.latex(
            r"NB = \text{Sens} \times p - (1 - \text{Spec}) \times (1 - p) \times \frac{p_t}{1 - p_t} - \text{Test Harm}")
        st.markdown("##### Calculation")
        st.latex(r'''
        NB = (%.2f \times %.2f) - ((1 - %.2f) \times (1 - %.2f) \times \frac{%.2f}{1 - %.2f}) - %.4f
        ''' % (sens, prev, spec, prev, pt, pt, test_harm))

        st.latex(r'''
        NB = %.4f - (%.2f \times %.2f \times %.2f) - %.4f
        ''' % (term_tp, (1 - spec), (1 - prev), weight, test_harm))
    else:
        # Standard Formula
        st.latex(r"NB = \text{Sens} \times p - (1 - \text{Spec}) \times (1 - p) \times \frac{p_t}{1 - p_t}")
        st.markdown("##### Calculation")
        st.latex(r'''
        NB = (%.2f \times %.2f) - ((1 - %.2f) \times (1 - %.2f) \times \frac{%.2f}{1 - %.2f})
        ''' % (sens, prev, spec, prev, pt, pt))

        st.latex(r'''
        NB = %.4f - (%.2f \times %.2f \times %.2f)
        ''' % (term_tp, (1 - spec), (1 - prev), weight))

    st.latex(r'''\mathbf{NB = %.4f}''' % nb)


def plot_calibration_multi(models_data):
    fig, ax = plt.subplots(figsize=(5, 5))
    colors = [COLOR_M1, COLOR_M2, COLOR_M3]
    for i, m in enumerate(models_data):
        CalibrationDisplay.from_predictions(
            m['y_true'], m['probs'], n_bins=10, ax=ax,
            name=m['name'], color=colors[i]
        )
    ax.set_title("Calibration")
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc="upper left")
    st.pyplot(fig)


def plot_dca_multi_compare(models_data, prevalence, threshold_pt):
    # Note: Test harm usually specific to a model, here we ignore it for comparison
    # or assume 0 for simplicity in the multi-view unless requested otherwise.
    fig, ax = plt.subplots(figsize=(5, 5))
    pts, nb_all, nb_none = get_treat_all_none(prevalence)
    ax.plot(pts, nb_all, label="Treat all", color='black', linestyle='--')
    ax.plot(pts, nb_none, label="Treat none", color='black', linestyle='-')

    colors = [COLOR_M1, COLOR_M2, COLOR_M3]
    for i, m in enumerate(models_data):
        pts_model, nb_vals = compute_dca_curve(m['y_true'], m['probs'], prevalence, test_harm=0.0)
        ax.plot(pts_model, nb_vals, color=colors[i], label=m['name'])

    ax.set_xlabel('Threshold Probability')
    ax.axvline(threshold_pt, color=COLOR_ACCENT, linestyle="--")
    ax.set_ylim(-0.05, prevalence + 0.1)
    ax.set_title(f"DCA")
    ax.legend(fontsize='small')
    st.pyplot(fig)


def plot_roc_multi(models_data):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Chance")
    colors = [COLOR_M1, COLOR_M2, COLOR_M3]
    for i, m in enumerate(models_data):
        fpr, tpr, _ = roc_curve(m['y_true'], m['probs'])
        auc_val = roc_auc_score(m['y_true'], m['probs'])
        ax.plot(fpr, tpr, color=colors[i], label=f"{m['name']} (AUC={auc_val:.2f})")
    ax.set_title("ROC")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend(loc="lower right", fontsize='small')
    st.pyplot(fig)


# -------------------------
#   Streamlit App Layout
# -------------------------
st.title("Decision Curve Analysis Visualizer")

# ==========================================
#       PAGE 1: SINGLE MODEL ANALYSIS
# ==========================================

# --- Page 1 Controls ---
# Row 1: Basic Parameters
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    n_main = st.slider("Sample Size", 500, 5000, 1000, 500, key="n_main")
with c2:
    prev_main = st.slider("Prevalence", 0.05, 0.95, 0.33, 0.01, key="prev_main")
with c3:
    pt_main = st.slider("Decision Threshold (pₜ)", 0.01, 0.99, 0.33, 0.01, key="pt_main")
with c4:
    auc_main = st.slider("Model AUC", 0.5, 0.99, 0.80, 0.01, key="auc_main")

# Row 2: Test Harm (New)
with c6:
    use_harm = st.checkbox("Include Test Harm")
with c5:
    if use_harm:
        harm_val = st.slider("Test Harm", 0.0, 0.1, 0.02, 0.005, key="harm_main")
    else:
        harm_val = 0.0

# Data Gen (Page 1)
X_main, y_main, lr_main = generate_data_for_model(auc_main, prev_main, n_main)
probs_main = get_probabilities(lr_main, X_main, 1.0)
model_main_data = {'name': 'Main Model', 'y_true': y_main, 'probs': probs_main}

# Calculate Metrics
sens, spec, tn, fp, fn, tp = calculate_metrics_at_pt(y_main, probs_main, pt_main)
nb_main = net_benefit(prev_main, sens, spec, pt_main, test_harm=harm_val)

# Layout (Page 1)
col1, col2, col3 = st.columns([1, 1.2, 0.8])

with col1:
    # Pass harm value to plot function so DCA curve adjusts
    plot_combined_dca_kde(model_main_data, prev_main, pt_main, test_harm=harm_val)

with col2:
    # Pass harm value and toggle state to display function
    display_nb_calc_detailed(y_main, probs_main, prev_main, pt_main, test_harm=harm_val, use_harm=use_harm)

with col3:
    st.markdown("**Confusion Matrix**")
    cm_df = pd.DataFrame([[tn, fp], [fn, tp]], index=["True Neg", "True Pos"], columns=["Pred Neg", "Pred Pos"])
    st.table(cm_df)

    st.markdown("**Metrics**")
    # Interpretation of Test Harm in formula: NB decreases
    metrics_df = pd.DataFrame({
        "Metric": ["Net Benefit", "Sensitivity", "Specificity", "AUC"],
        "Value": [f"{nb_main:.4f}", f"{sens:.3f}", f"{spec:.3f}", f"{auc_main:.3f}"]
    })
    st.table(metrics_df.set_index("Metric"))

# ==========================================
#       PAGE 2: MODEL COMPARISON
# ==========================================
st.markdown("---")
st.header("Relation to Discrimination and Calibration")

# --- Page 2 Global Controls ---
gc1, gc2, gc3 = st.columns(3)
with gc1:
    n_comp = st.slider("Sample Size", 500, 5000, 1000, 500, key="n_comp")
with gc2:
    prev_comp = st.slider("Prevalence", 0.05, 0.95, 0.33, 0.01, key="prev_comp")
with gc3:
    pt_comp = st.slider("Decision Threshold (pₜ)", 0.01, 0.99, 0.33, 0.01, key="pt_comp")

st.write("")

# --- Page 2 Layout: 4 Columns ---
col_dca, col_roc, col_cal, col_controls = st.columns([1, 1, 1, 0.8])

# --- Column 4: Controls (Right Side) ---
with col_controls:
    # Model 1
    st.markdown(f"<span style='color:{COLOR_M1}'><b>Model 1</b></span>", unsafe_allow_html=True)
    col_m1_1, col_m1_2 = st.columns(2)
    with col_m1_1:
        auc_m1 = st.slider("AUC", 0.5, 0.99, 0.80, 0.01, key="auc_m1")
    with col_m1_2:
        cal_m1 = st.slider("Calib", 0.1, 5.0, 0.4, 0.1, key="cal_m1")

    # Model 2
    st.markdown(f"<span style='color:{COLOR_M2}'><b>Model 2</b></span>", unsafe_allow_html=True)
    col_m2_1, col_m2_2 = st.columns(2)
    with col_m2_1:
        auc_m2 = st.slider("AUC", 0.5, 0.99, 0.80, 0.01, key="auc_m2")
    with col_m2_2:
        cal_m2 = st.slider("Calib", 0.1, 5.0, 1.0, 0.1, key="cal_m2")

    # Model 3
    st.markdown(f"<span style='color:{COLOR_M3}'><b>Model 3</b></span>", unsafe_allow_html=True)
    col_m3_1, col_m3_2 = st.columns(2)
    with col_m3_1:
        auc_m3 = st.slider("AUC", 0.5, 0.99, 0.80, 0.01, key="auc_m3")
    with col_m3_2:
        cal_m3 = st.slider("Calib", 0.1, 5.0, 2.5, 0.1, key="cal_m3")

# --- Data Generation ---
X1, y1, lr1 = generate_data_for_model(auc_m1, prev_comp, n_comp)
probs1 = get_probabilities(lr1, X1, cal_m1)

X2, y2, lr2 = generate_data_for_model(auc_m2, prev_comp, n_comp)
probs2 = get_probabilities(lr2, X2, cal_m2)

X3, y3, lr3 = generate_data_for_model(auc_m3, prev_comp, n_comp)
probs3 = get_probabilities(lr3, X3, cal_m3)

comp_models = [
    {'name': 'Model 1', 'y_true': y1, 'probs': probs1},
    {'name': 'Model 2', 'y_true': y2, 'probs': probs2},
    {'name': 'Model 3', 'y_true': y3, 'probs': probs3}
]

# --- Columns 1-3: Plots ---
with col_dca:
    plot_dca_multi_compare(comp_models, prev_comp, pt_comp)

with col_roc:
    plot_roc_multi(comp_models)

with col_cal:
    plot_calibration_multi(comp_models)