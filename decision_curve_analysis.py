import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, gaussian_kde
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibrationDisplay
import streamlit as st

matplotlib.use("Agg")  # for Streamlit

# -------------------------
#   COLOR CONFIGURATION
# -------------------------
COLOR_MAIN = "#9CC6DB"  # Main Model
COLOR_ACCENT = "#FF4B4B"  # Threshold lines
COLOR_M1_DIS = "#088F8F"   # Model 1 Discrimination
COLOR_M2_DIS = "#E86FAE"   # Model 2 Discrimination
COLOR_M1 = "#8F0177"  # Model 1
COLOR_M2 = "#84994F"  # Model 2
COLOR_M3 = "#F1A90E" # Model 3


# -------------------------
#   Streamlit Page Config
# -------------------------
st.set_page_config(layout="wide", page_title="DCA Visualizer")


# -------------------------
#   Data Generation
# -------------------------
@st.cache_data
def generate_model_data(target_auc, prevalence, n_total, variance=1.0):
    low, high = 0.0, 7.0
    best_probs = None
    best_y = None

    for _ in range(12):
        mid = (low + high) / 2
        n_pos = int(n_total * prevalence)
        n_neg = n_total - n_pos
        pos_scores = np.random.normal(loc=mid, scale=variance, size=n_pos)
        neg_scores = np.random.normal(loc=0.0, scale=1.0, size=n_neg)
        X = np.concatenate([pos_scores, neg_scores]).reshape(-1, 1)
        y = np.array([1] * n_pos + [0] * n_neg)

        model = LogisticRegression().fit(X, y)

        probs = model.predict_proba(X)[:, 1]
        current_auc = roc_auc_score(y, probs)

        if current_auc < target_auc:
            low = mid
        else:
            high = mid
        best_probs = probs
        best_y = y

    return best_y, best_probs

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
    nb_all = prevalence - (1.0 - prevalence) * (pts / (1.0 - pts))
    nb_none = np.zeros_like(pts)
    return pts, nb_all, nb_none


def calculate_metrics_at_pt(y_true, probs, pt):
    preds = (probs >= pt).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return sens, spec, tn, fp, fn, tp

def apply_calibration(probs, factor):
    p = np.clip(probs, 0.001, 0.999)
    logits = np.log(p / (1 - p))
    scaled_logits = logits * (1.0 / factor)
    return 1 / (1 + np.exp(-scaled_logits))

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
            r"NB = \text{sens} \times prev - (1 - \text{spec}) \times (1 - prev) \times \frac{p_t}{1 - p_t} - \text{Test Harm}")
        st.markdown("##### Calculation")
        st.latex(r'''
        NB = (%.2f \times %.2f) - ((1 - %.2f) \times (1 - %.2f) \times \frac{%.2f}{1 - %.2f}) - %.4f
        ''' % (sens, prev, spec, prev, pt, pt, test_harm))

        st.latex(r'''
        NB = %.4f - (%.2f \times %.2f \times %.2f) - %.4f
        ''' % (term_tp, (1 - spec), (1 - prev), weight, test_harm))
    else:
        # Standard Formula
        st.latex(r"NB = \text{sens} \times prev - (1 - \text{spec}) \times (1 - prev) \times \frac{p_t}{1 - p_t}")
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
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.1)
    ax.legend(loc="upper left", fontsize='small')
    st.pyplot(fig)


def plot_dca_multi_compare_kde(models_data, prevalence, threshold_pt):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(6, 6),
                                        gridspec_kw={'height_ratios': [2, 0.5, 0.5], 'hspace': 0.3})

    # DCA plot
    pts, nb_all, nb_none = get_treat_all_none(prevalence)
    ax1.plot(pts, nb_all, label="Treat all", color='black', linestyle='--')
    ax1.plot(pts, nb_none, label="Treat none", color='black', linestyle='-')

    colors = [COLOR_M1_DIS, COLOR_M2_DIS]
    for i, m in enumerate(models_data):
        current_harm = m.get('test_harm_sec', 0.0)
        pts_model, nb_vals = compute_dca_curve(m['y_true'], m['probs'], prevalence, current_harm)
        ax1.plot(pts_model, nb_vals, color=colors[i], label=m['name'], linewidth=2)

    ax1.set_ylim(-0.05, prevalence + 0.1) # Adjust Y-lim to handle potential drops due to harm
    ax1.set_ylabel('Net Benefit')
    ax1.axvline(threshold_pt, color=COLOR_ACCENT, linestyle="--", alpha=0.7)
    ax1.set_xlim(0, 1)
    ax1.set_title("Decision Curve Analysis")
    ax1.legend(fontsize='small')
    ax1.grid(True, alpha=0.1)

    # Distribution plots
    for ax, m, color in zip([ax2, ax3], models_data, colors):
        y = m['y_true']
        p = m['probs']

        sns.kdeplot(p[y == 0], ax=ax, fill=True, color="gray", label="Neg", alpha=0.3, bw_adjust=0.8)
        sns.kdeplot(p[y == 1], ax=ax, fill=True, color=color, label="Pos", alpha=0.5, bw_adjust=0.8)

        ax.axvline(threshold_pt, color=COLOR_ACCENT, linestyle="--", linewidth=1)
        ax.set_title("")
        ax.set_ylabel("Density")
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.legend(loc='upper right', fontsize='xx-small', frameon=False)
        ax.grid(axis='x', alpha=0.2)

    ax3.set_xlabel("Predicted Probability")
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

    ax.grid(True, alpha=0.1)
    ax.set_xlabel('Threshold Probability')
    ax.set_xlim(0, 1)
    ax.axvline(threshold_pt, color=COLOR_ACCENT, linestyle="--")
    ax.set_ylim(-0.05, prevalence + 0.1)
    ax.set_title(f"Decision Curve Analysis")
    ax.legend(fontsize='small')
    st.pyplot(fig)


def plot_roc_multi(models_data):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Chance")
    colors = [COLOR_M1_DIS, COLOR_M2_DIS]
    for i, m in enumerate(models_data):
        fpr, tpr, _ = roc_curve(m['y_true'], m['probs'])
        auc_val = roc_auc_score(m['y_true'], m['probs'])
        ax.plot(fpr, tpr, color=colors[i], label=f"{m['name']} (AUC={auc_val:.2f})")
    ax.set_title("ROC")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.grid(True, alpha=0.1)
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
# fix sample size at 5000
# fix auc at 0.80
c1, c2, c3, c4 = st.columns(4)
n_main = 5000
auc_main = 0.80

with c1:
    prev_main = st.slider("Prevalence", 0.05, 0.95, 0.33, 0.01, key="prev_main")
with c2:
    pt_main = st.slider("Decision Threshold (pₜ)", 0.01, 0.99, 0.33, 0.01, key="pt_main")

# Row 2: Test Harm
with c4:
    use_harm = st.checkbox("Include Test Harm")
with c3:
    if use_harm:
        harm_val = st.slider("Test Harm", 0.0, 0.1, 0.02, 0.005, key="harm_main")
    else:
        harm_val = 0.0

# Data Gen (Page 1)
y_main, probs_main = generate_model_data(auc_main, prev_main, n_main)
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

# ===============================================
#       PAGE 2: MODEL COMPARISON - DISCRIMINATION
# ===============================================
st.markdown("---")
st.header("Model Comparison - Discrimination")

# fixed sample size at 5000
n_sec = 5000
dc1, dc2 = st.columns(2)
with dc1:
    prev_sec = st.slider("Prevalence", 0.05, 0.95, 0.33, 0.01, key="prev_sec")
with dc2:
    pt_sec = st.slider("Decision Threshold (pₜ)", 0.01, 0.99, 0.33, 0.01, key="pt_sec")
st.write("")

col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 0.8, 0.6])
with col_ctrl3:
    # Model1
    st.markdown(f"<h3 style='color:{COLOR_M1_DIS}'>Model 1</h3>", unsafe_allow_html=True)
    use_harm_m1 = st.checkbox("Include Test Harm", key="use_harm_m1")
    if use_harm_m1:
        harm_val_m1 = st.slider("Test Harm", 0.0, 0.1, 0.02, 0.005, key="harm_val_m1")
    else:
        harm_val_m1 = 0.0
    auc_m1 = st.slider("AUC", 0.55, 0.95, 0.80, key="am1")
    var_m1 = st.slider("Heterogeneity (Std Dev)", 0.5, 2.5, 1.0, 0.1, key="vm1",
                       help="Lower variance means predictions cluster more in the middle")
    # Model 2
    st.markdown(f"<h3 style='color:{COLOR_M2_DIS}'>Model 2</h3>", unsafe_allow_html=True)
    use_harm_m2 = st.checkbox("Include Test Harm", key="use_harm_m2")
    if use_harm_m2:
        harm_val_m2 = st.slider("Test Harm", 0.0, 0.1, 0.02, 0.005, key="harm_val_m2")
    else:
        harm_val_m2 = 0.0
    auc_m2 = st.slider("AUC", 0.55, 0.95, 0.80, key="am2")
    var_m2 = st.slider("Heterogeneity (Std Dev)", 0.5, 2.5, 2.0, 0.1, key="vm2",
                       help="Higher variance means more 'confident' predictions at 0 and 1")

# Page 2: Data generation
y1_p2, p1_p2 = generate_model_data(auc_m1, prev_sec, n_sec, variance=var_m1)
y2_p2, p2_p2 = generate_model_data(auc_m2, prev_sec, n_sec, variance=var_m2)

m1_data = {'name': 'Model 1', 'y_true': y1_p2, 'probs': p1_p2, 'test_harm_sec': harm_val_m1}
m2_data = {'name': 'Model 2', 'y_true': y2_p2, 'probs': p2_p2, 'test_harm_sec': harm_val_m2}

with col_ctrl1:
    plot_dca_multi_compare_kde([m1_data, m2_data], prev_sec, pt_sec)

with col_ctrl2:
    plot_roc_multi([m1_data, m2_data])

# ============================================
#       PAGE 3: MODEL COMPARISON - CALIBRATION
# ============================================
st.markdown("---")
st.header("Model Comparison - Calibration")

# --- Page 3 Global Controls ---
gc1, gc2 = st.columns(2)
# fixed sample size at 5000, fixed AUC at 0.80
n_comp = 5000
auc_cal = 0.80

with gc1:
    prev_comp = st.slider("Prevalence", 0.05, 0.95, 0.33, 0.01, key="prev_comp")
with gc2:
    pt_comp = st.slider("Decision Threshold (pₜ)", 0.01, 0.99, 0.33, 0.01, key="pt_comp")

st.write("")

# --- Page 3 Layout: 3 Columns ---
col_dca, col_cal, col_controls = st.columns([1, 1, 0.7])

# --- Column 3: Controls (Right Side) ---
with col_controls:
    # Model 1
    st.markdown(f"<span style='color:{COLOR_M1}'><b>Model 1</b></span>", unsafe_allow_html=True)
    cal_m1 = st.slider("Calibration", 0.1, 3.0, 0.4, 0.1, key="cal_m1")

    # Model 2
    st.markdown(f"<span style='color:{COLOR_M2}'><b>Model 2</b></span>", unsafe_allow_html=True)
    cal_m2 = st.slider("Calibration", 0.1, 3.0, 1.0, 0.1, key="cal_m2")

    # Model 3
    st.markdown(f"<span style='color:{COLOR_M3}'><b>Model 3</b></span>", unsafe_allow_html=True)
    cal_m3 = st.slider("Calibration", 0.1, 3.0, 2.0, 0.1, key="cal_m3")

# --- Data Generation ---
y1, p1_raw = generate_model_data(auc_cal, prev_comp, n_comp, variance=1.0)
y2, p2_raw = generate_model_data(auc_cal, prev_comp, n_comp, variance=1.0)
y3, p3_raw = generate_model_data(auc_cal, prev_comp, n_comp, variance=1.0)

probs1 = apply_calibration(p1_raw, cal_m1)
probs2 = apply_calibration(p2_raw, cal_m2)
probs3 = apply_calibration(p3_raw, cal_m3)

comp_models = [
    {'name': 'Model 1', 'y_true': y1, 'probs': probs1},
    {'name': 'Model 2', 'y_true': y2, 'probs': probs2},
    {'name': 'Model 3', 'y_true': y3, 'probs': probs3}
]

# --- Columns 1-2: Plots ---
with col_dca:
    plot_dca_multi_compare(comp_models, prev_comp, pt_comp)

with col_cal:
    plot_calibration_multi(comp_models)