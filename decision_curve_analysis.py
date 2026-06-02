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
import plotly.graph_objects as go
import math

matplotlib.use("Agg")  # for Streamlit

## Set font ##
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Work+Sans:wght@300;400;600;700&display=swap');

        /* Die Schriftart auf Text-Elemente anwenden, aber Icons ausschließen */
        html, body, .stMarkdown, p, h1, h2, h3, h4, h5, h6, label, [data-testid="stMetricValue"] {
            font-family: 'Work Sans', sans-serif !important;
        }

        /* Spezifisch für den Expander-Titel (verhindert das Überlagern) */
        .st-ae p, .st-an p {
            font-family: 'Work Sans', sans-serif !important;
        }

        /* WICHTIG: Icons (Pfeile etc.) dürfen NICHT die Schriftart überschreiben */
        [data-testid="stExpander"] svg, 
        [data-icon], 
        .st-ae svg {
            font-family: inherit !important;
        }

        /* Falls der Expander-Header immer noch zerschossen ist, hier gezielt korrigieren */
        summary[data-testid="stExpanderHeader"] {
            font-family: 'Work Sans', sans-serif !important;
        }

        /* Das Icon im Expander schützen */
        summary[data-testid="stExpanderHeader"] svg {
            font-family: unset !important;
        }
    </style>
""", unsafe_allow_html=True)

# Inject Custom CSS
st.markdown("""
<style>
/* Change the overall app background color */
.stApp {
    background-color: rgba(0,0,0,0); /* Light gray/off-white */
}

/* Target the header so it blends seamlessly with the background */
[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}

/* 
The Bulletproof Selector:
1. Finds ANY vertical block that contains the 'custom-card' marker.
2. EXCLUDES it if it contains a nested vertical block that also has the marker.
Result: It only ever styles the absolute deepest container holding your marker.
*/
div[data-testid="stVerticalBlock"]:has(.custom-card):not(:has(div[data-testid="stVerticalBlock"] .custom-card)) {
    background-color: #FFFFFF !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15) !important;
    padding: 1.5rem !important;
    /* Ensure the shadow doesn't get clipped by Streamlit's default overflow settings */
    overflow: visible !important; 
}
</style>
""", unsafe_allow_html=True)


def st_footer(text):
    """Renders small, black footer text cleanly."""
    st.markdown("<hr style='margin: 1rem 0; border: none; border-top: 1px solid #E6E9EF;'>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='font-size: 0.85rem; color: black; margin: 0;'>{text}</p>",
        unsafe_allow_html=True
    )
# -------------------------
#   COLOR CONFIGURATION
# -------------------------
COLOR_MAIN = "#9CC6DB"  # Main Model
COLOR_ACCENT = "#FF4B4B"  # Threshold lines
COLOR_M1_DIS = "#09a7a7"   # Model 1 Discrimination
COLOR_M1_DIS_Neg = "#055f5f" # Model 1 Discrimination Negative
COLOR_M2_DIS = "#df6ba8"   # Model 2 Discrimination
COLOR_M2_DIS_Neg = "#8a4268" # Model 2 Discrimination Negative
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
    # variance meaning the ratio
    low, high = 0.0, 7.0
    best_probs = None
    best_y = None

    scale_neg = 1.0 / np.sqrt(variance)
    scale_pos = 1.0 * np.sqrt(variance)

    for _ in range(25):
        mid = (low + high) / 2
        n_pos = int(n_total * prevalence)
        n_neg = n_total - n_pos
        pos_scores = np.random.normal(loc=mid, scale=scale_pos, size=n_pos)
        neg_scores = np.random.normal(loc=0.0, scale=scale_neg, size=n_neg)
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

    st.markdown("### Net Benefit Formula")

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
    neg_distr_colors = [COLOR_M1_DIS_Neg, COLOR_M2_DIS_Neg]
    for ax, m, color, neg_distr_colors in zip([ax2, ax3], models_data, colors, neg_distr_colors):
        y = m['y_true']
        p = m['probs']

        sns.kdeplot(p[y == 0], ax=ax, fill=True, color=neg_distr_colors, label="Negative", alpha=0.3, bw_adjust=0.8)
        sns.kdeplot(p[y == 1], ax=ax, fill=True, color=color, label="Positive", alpha=0.5, bw_adjust=0.8)

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
st.markdown("Decision Curve Analysis (DCA) is a method for estimating and evaluating a model's clinical utility by quantifying clinical consequences in terms of benefits and harms and thereby estimating the net benefit (NB) of a model. The evaluation of the potential clinical utility of a model is an essential addition to the evaluation of the model's statistical predictive performance, as the latter does not account for the clinical consequences and therefore is not sufficiently informative when deciding whether to use the respective model in clinical practice. ")
st.write("")
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
    prev_main = st.slider("Prevalence of the Event", 0.05, 0.95, 0.33, 0.01, key="prev_main")

with c2:
    pt_main = st.slider("Decision Threshold (pₜ)", 0.01, 0.99, 0.33, 0.01, key="pt_main", help="**Numbers needed**: How many interventions would I do to get one True Positive? **For instance:** I would perform 20 times intervention x to treat one person for whom the intervention is beneficial (i.e. with the event) -> **Odds** of 1:20, i.e. threshold of 0.0476 (4.76%)")

    # Row 2: Test Harm
with c4:
    use_harm = st.checkbox("Include Test Harm", help="The utility of a test in DCA is per default equal to 0; however, in cases where the data collection required to inform the model involves invasive or dangerous procedures or significant financial, time or effort investment, you can **explicitly account for the test harm** in DCA.")
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
    with st.container(border=True):
        st.markdown("<span class='custom-card'></span>", unsafe_allow_html=True)
        # Pass harm value to plot function so DCA curve adjusts
        plot_combined_dca_kde(model_main_data, prev_main, pt_main, test_harm=harm_val)
        st_footer("<b>Figure 1.</b> Decision Curve Analysis and the corresponding risk distribution plot.")

with col2:
    # Pass harm value and toggle state to display function
    with st.container(border=True):
        st.markdown("<span class='custom-card'></span>", unsafe_allow_html=True)
        display_nb_calc_detailed(y_main, probs_main, prev_main, pt_main, test_harm=harm_val, use_harm=use_harm)

# Metrics
    with st.container (border= True):
        st.markdown("<span class='custom-card'></span>", unsafe_allow_html=True)
        st.markdown("### **Metrics**")
        _, content_col, _ = st.columns([0.25, 0.7, 0.05])
        with content_col:
            st.markdown("""
                        <style>
                            [data-testid="stMetricValue"] {
                            font-size: 1.8rem !important;
                            }
                            .stDataFrame [data-testid="stTable"] th,
                            .stDataFrame [data-testid="stTable"] td {
                            }
                        </style>
                    """, unsafe_allow_html=True)
            m1, m2 = st.columns(2, gap="small")
            m1.metric("Net Benefit", f"{nb_main:.4f}")
            m2.metric("AUC", f"{auc_main:.3f}")

            m3, m4 = st.columns(2)
            m3.metric("Sensitivity", f"{sens:.1%}")
            m4.metric("Specificity", f"{spec:.1%}")

with col3:
    with st.container(border=True):
        st.markdown("<span class='custom-card'></span>", unsafe_allow_html=True)
        st.markdown("### **Confusion Matrix**")
        # Adjust Scale for Graphic
        total = tn + fp + fn + tp
        def scale_to_100(value, total):
            return int(round((value / total) * 100)) if total > 0 else 0

        s_tp = scale_to_100(tp, total)
        s_fp = scale_to_100(fp, total)
        s_fn = scale_to_100(fn, total)
        s_tn = scale_to_100(tn, total)

        COLS = 7
        GAP_Y = 5

        rows_tp = math.ceil(s_tp / COLS) if s_tp > 0 else 0
        rows_fp = math.ceil(s_fp / COLS) if s_fp > 0 else 0
        rows_fn = math.ceil(s_fn / COLS) if s_fn > 0 else 0
        rows_tn = math.ceil(s_tn / COLS) if s_tn > 0 else 0

        max_rows_top = max(rows_tp, rows_fp, 1)
        max_rows_bottom = max(rows_fn, rows_tn, 1)

        def get_coords(count, offset_x, base_y, direction, section_max_rows):
            x = [offset_x + (i % COLS) for i in range(count)]
            if direction == "up":
                y = [(base_y + section_max_rows - 1) - (i // COLS) for i in range(count)]
            else:
                y = [base_y - (i // COLS) for i in range(count)]
            return x, y


        fig = go.Figure()

        quadrants = [
            (s_tp, tp, "True Positive", "#E74C3C", -2, 0, "up", max_rows_top, False),
            (s_fp, fp, "False Positive", "#3498DB", 7, 0, "up", max_rows_top, True),
            (s_fn, fn, "False Negative", "#E74C3C", -2, -GAP_Y, "down", max_rows_bottom, True),
            (s_tn, tn, "True Negative", "#3498DB", 7, -GAP_Y, "down", max_rows_bottom, False)
        ]
        for s_count, real_count, label, color, ox, oy, direct, m_rows, false_categorized in quadrants:
            if s_count > 0:
                x, y = get_coords(s_count, ox, oy, direct, m_rows)
                if false_categorized:
                    marker_style = dict(
                        size=12,
                        color='rgba(0,0,0,0)',
                        symbol="circle",
                        line=dict(
                            color=color,
                            width=2,
                            dash='dot'
                        )
                    )
                else:
                    marker_style = dict(
                        size=12,
                        color=color,
                        symbol="circle",
                        line=dict(width=0)
                    )
                fig.add_trace(go.Scatter(
                    x=x, y=y,
                    mode='markers',
                    marker=marker_style,
                    name=f"{label}",
                    hovertemplate=f"<b>{label}</b><br>Percentage: %{{text}}%<extra></extra>",
                    text=[round(real_count / total * 100, 1)] * s_count
                ))

        plotly_font = dict(family="Work Sans, sans-serif", size=13, color="black")
        plotly_font_bold = dict(family="Work Sans, sans-serif", size=14, color="black")
        #Predicted positive
        y_pos_labels = max_rows_top + 0.8
        fig.add_annotation(x=1.2, y=y_pos_labels, text=f"True Positive (n={tp})", showarrow=False, font=plotly_font)
        fig.add_annotation(x=10.2, y=y_pos_labels, text=f"False Positive (n={fp})", showarrow=False, font=plotly_font)
        fig.add_annotation(x=5.75, y=y_pos_labels + 1.5, text=f"<b>PREDICTED POSITIVE (n={tp + fp})</b>", showarrow=False,
                           font=plotly_font_bold)
        #Predicted negative
        y_neg_labels_top = -GAP_Y +1.2
        fig.add_annotation(x=1.2, y=y_neg_labels_top, text=f"False Negative (n={fn})", showarrow=False, font=plotly_font)
        fig.add_annotation(x=10.2, y=y_neg_labels_top, text=f"True Negative (n={tn})", showarrow=False, font=plotly_font)
        fig.add_annotation(x=5.75, y=y_neg_labels_top + 1.5, text=f"<b>PREDICTED NEGATIVE (n={tn + fn})</b>", showarrow=False,
                           font=plotly_font_bold)

        deepest_row = -GAP_Y - (max_rows_bottom - 1)

        fig.update_layout(
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-4, 16]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[deepest_row - 1.5, y_pos_labels + 3]),
            margin=dict(l=0, r=0, t=10, b=0),
            height=500,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        st_footer(f"<b>Figure 2.</b> Confusion Matrix as a function of prevalence and the decision threshold. One icon is approximately equivalent to {total / 100:.0f} individuals (N={total}).")



# ===============================================
#       PAGE 2: MODEL COMPARISON - DISCRIMINATION
# ===============================================
st.markdown("---")
st.header("Model Comparison - Discrimination")
st.markdown("Steyerberg et al. (2010) explain that a well-discriminating model is particularly important when resources are limited and only those who could benefit most from them, such as high-risk individuals (vs. low-risk individuals), should be allocated the resource. This is because measures of discrimination such as the AUC tell you how well your model ranks individuals with the event higher than individuals without the event. As this is a highly relevant quality in various clinical scenarios, a model's discrimination performance is taken into account in DCA (Vickers et al., 2019), as illustrated below. ")
selected_mode = st.pills(
    "Selection of Scenario:",
    ["Free Analysis", "Scenario 1a: Same AUC, different NB", "Scenario 1b: Higher AUC, lower NB", "Scenario 2: Test Harm"],
    default="Free Analysis"
)

if "last_mode" not in st.session_state:
    st.session_state.last_mode = "Free Analysis"
    st.session_state.prev_sec = 0.33
    st.session_state.pt_sec = 0.33
    st.session_state.am1 = 0.80
    st.session_state.am2 = 0.60
    st.session_state.vm1 = 1.0
    st.session_state.vm2 = 1.0
    st.session_state.use_harm_m1 = False
    st.session_state.use_harm_m2 = False

mode_changed = (selected_mode != st.session_state.last_mode)

var_m1 = 1.0
var_m2 = 1.0

if selected_mode == "Scenario 2: Test Harm":
    if mode_changed:
        st.session_state.prev_sec = 0.20
        st.session_state.pt_sec = 0.05
        st.session_state.am1 = 0.80
        st.session_state.am2 = 0.70
        st.session_state.vm1 = 1.0
        st.session_state.vm2 = 1.0
        st.session_state.use_harm_m1 = True
        st.session_state.harm_val_m1 = 0.03
        st.session_state.use_harm_m2 = False
    var_m1 = 1.0
    var_m2 = 1.0
elif selected_mode == "Scenario 1b: Higher AUC, lower NB":
    if mode_changed:
        st.session_state.prev_sec = 0.20
        st.session_state.pt_sec = 0.25
        st.session_state.am1 = 0.8
        st.session_state.am2 = 0.7
        st.session_state.use_harm_m1 = False
        st.session_state.use_harm_m2 = False
        st.session_state.vm1 = 0.5
        st.session_state.vm2 = 1.6
    var_m1 = st.session_state.get("vm1", 0.5)
    var_m2 = st.session_state.get("vm2", 1.6)
elif selected_mode == "Scenario 1a: Same AUC, different NB":
    if mode_changed:
        st.session_state.prev_sec = 0.30
        st.session_state.pt_sec = 0.27
        st.session_state.am1 = 0.80
        st.session_state.am2 = 0.80
        st.session_state.vm1 = 1.0
        st.session_state.vm2 = 2.2
        st.session_state.use_harm_m1 = False
        st.session_state.use_harm_m2 = False
    var_m1 = st.session_state.get("vm1", 1.0)
    var_m2 = st.session_state.get("vm2", 2.2)
else:
    if mode_changed:
        st.session_state.prev_sec = 0.33
        st.session_state.pt_sec = 0.33
        st.session_state.am1 = 0.80
        st.session_state.am2 = 0.60
        st.session_state.vm1 = 1.0
        st.session_state.vm2 = 1.0
        st.session_state.use_harm_m1 = False
        st.session_state.use_harm_m2 = False
    var_m1 = st.session_state.get("vm1", 1.0)
    var_m2 = st.session_state.get("vm2", 1.0)

st.session_state.last_mode = selected_mode


# fixed sample size at 5000
n_sec = 5000
if selected_mode == "Scenario 2: Test Harm":
    st.markdown("Let's imagine a scenario, in which we want to avoid missing TPs, e.g. because of severe clinical consequences. Thus, the decision threshold is set relatively low at 5% to 10%. We compare two models, model 1 is more complex and requires hard-to-obtain data, but has a higher AUC than model 2, using easy-access data. We want to evaluate the NB of both models in the respective threshold range. Note that the model classification profile is fixed at 1.0 for both models for this scenario, hence yielding symmetric ROC curves.")
if selected_mode == "Scenario 1b: Higher AUC, lower NB":
    st.markdown("")
if selected_mode == "Scenario 1a: Same AUC, different NB":
    st.markdown("")
dc1, dc2 = st.columns(2)
with dc1:
    prev_sec = st.slider("Prevalence", 0.05, 0.95, value=st.session_state.get("prev_sec", 0.33), key="prev_sec")
with dc2:
    pt_sec = st.slider("Decision Threshold (pₜ)", 0.01, 0.99, value=st.session_state.get("pt_sec", 0.33), key="pt_sec")
st.write("")

col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 1])
with col_ctrl3:
    st.markdown(f"<h3 style='color:{COLOR_M1_DIS}'>Model 1</h3>", unsafe_allow_html=True)
    use_harm_m1 = st.checkbox("Include Test Harm", key="use_harm_m1")
    if selected_mode == "Scenario 2: Test Harm":
        st.info("ℹ️ Taking into account the test harm of model 1, we might conclude that we would not subject more that 30 persons to it in order to find one true positive if the model was perfect (Vickers et al., 2019). The reciprocal of 30 is approximately 0.03.")
    if use_harm_m1:
        harm_val_m1 = st.slider("Test Harm", 0.0, 0.1, 0.02, 0.005, key="harm_val_m1")
    else:
        harm_val_m1 = 0.0
    auc_m1 = st.slider("AUC", 0.55, 0.95, value=st.session_state.get("am1", 0.80), key="am1")
    if selected_mode in ["Free Analysis", "Scenario 1a: Same AUC, different NB", "Scenario 1b: Higher AUC, lower NB"]:
        var_m1 = st.slider("Model Classification Profile", 0.4, 2.5, value=st.session_state.get("vm1", 1.0), key="vm1", help=(
    "Adjusts the Model's Classification Profile (illustrating how it distributes risk predictions).\n\n"
    "• 1.0 = The Balanced All-Rounder. The model performs more consistently across the risk spectrum.\n\n"
    "• > 1.0 = The High-Risk Specialist. The model predicts more confidently for higher-risk patients, making it more fitting for higher decision thresholds.\n\n"
    "• < 1.0 = The Low-Risk Specialist. The model predicts more confidently for lower-risk patients, making it more fitting for lower decision thresholds (e.g. easy-to-access screening)."
    )
)
    # Model 2
    st.markdown(f"<h3 style='color:{COLOR_M2_DIS}'>Model 2</h3>", unsafe_allow_html=True)
    use_harm_m2 = st.checkbox("Include Test Harm", key="use_harm_m2")
    if use_harm_m2:
        harm_val_m2 = st.slider("Test Harm", 0.0, 0.1, 0.02, 0.005, key="harm_val_m2")
    else:
        harm_val_m2 = 0.0
    auc_m2 = st.slider("AUC", 0.55, 0.95, value=st.session_state.get("am2", 0.60), key="am2")
    if selected_mode in ["Free Analysis", "Scenario 1a: Same AUC, different NB", "Scenario 1b: Higher AUC, lower NB"]:
        var_m2 = st.slider("Model Classification Profile", 0.4, 2.5, value=st.session_state.get("vm2", 1.0), key="vm2",
                       help=(
    "Adjusts the Model's Classification Profile (illustrating how it distributes risk predictions).\n\n"
    "• 1.0 = The Balanced All-Rounder. The model performs more consistently across the risk spectrum.\n\n"
    "• > 1.0 = The High-Risk Specialist. The model predicts more confidently for higher-risk patients, making it more fitting for higher decision thresholds.\n\n"
    "• < 1.0 = The Low-Risk Specialist. The model predicts more confidently for lower-risk patients, making it more fitting for lower decision thresholds (e.g. easy-to-access screening)."
    )
)

# Page 2: Data generation
y1_p2, p1_p2 = generate_model_data(auc_m1, prev_sec, n_sec, variance=var_m1)
y2_p2, p2_p2 = generate_model_data(auc_m2, prev_sec, n_sec, variance=var_m2)

m1_data = {'name': 'Model 1', 'y_true': y1_p2, 'probs': p1_p2, 'test_harm_sec': harm_val_m1}
m2_data = {'name': 'Model 2', 'y_true': y2_p2, 'probs': p2_p2, 'test_harm_sec': harm_val_m2}

with col_ctrl1:
    with st.container(border=True):
        st.markdown("<span class='custom-card'></span>", unsafe_allow_html=True)
        plot_dca_multi_compare_kde([m1_data, m2_data], prev_sec, pt_sec)
        st_footer("<b>Figure 3.</b> Decision Curve Analysis and the corresponding risk distribution plots for both models.")

with col_ctrl2:
    with st.container(border=True):
        st.markdown("<span class='custom-card'></span>", unsafe_allow_html=True)
        plot_roc_multi([m1_data, m2_data])
        st_footer("<b>Figure 4.</b> The receiver operating characteristic (ROC) curve for both models compared to each other. FPR= False Positive Rate, TPR= True Positive Rate.")

if selected_mode == "Scenario 2: Test Harm":
    st.success("💡 Considering the test harm, model 1 would be clinically harmful in the selected threshold range, even though the model performs better regarding its AUC. Model 2 would be eligible in this scenario, displaying higher NB in the selected threshold range and being superior to the default strategies, i.e. not harmful.")
if selected_mode == "Free Analysis":
    is_default = (
            st.session_state.get("prev_sec") == 0.33 and
            st.session_state.get("pt_sec") == 0.33 and
            st.session_state.get("am1") == 0.80 and
            st.session_state.get("am2") == 0.60 and
            st.session_state.get("vm1") == 1.0 and
            st.session_state.get("use_harm_m1") == False and
            st.session_state.get("use_harm_m2") == False and
            st.session_state.get("vm2") == 1.0
    )
    if is_default:
        st.success("💡 You can see that given two models with a similar model classification profile, the model with the higher discrimination performance (AUC) has NB across a wider range of threshold probabilities.")
if selected_mode == "Scenario 1a: Same AUC, different NB":
    st.write("")
    st.success(
        "💡 Notice that both models are set to the **same AUC** (e.g., 0.80). "
        "However, by changing the *Model Classification Profile*, you alter *how* they distribute risk:\n"
        "- **Model 1** acts more like an *All-Rounder*, yielding a higher NB at **lower thresholds**.\n"
        "- **Model 2** acts more like a *Specialist*, maintaining a higher NB at **higher thresholds** because it identifies a high-risk subgroup with higher certainty.\n\n"
        "**Different model qualities can be more or less beneficial or even harmful in different clinical contexts (reflected in the respective selected threshold range).**"
    )

# ============================================
#       PAGE 3: MODEL COMPARISON - CALIBRATION
# ============================================
st.markdown("---")
st.header("Model Comparison - Calibration")
st.markdown("Steyerberg et al. (2010) explain that a well-calibrated model is particularly essential if you want to inform patients about their prognosis. This is because, calibration measures how well the predicted probabilities correspond to the true fraction of positives. Van Calster & Vickers (2015) noted that for a well-calibrated model approximately x out of 100 patients with a risk score of x% should actually have the respective outcome. As this, too, is a highly relevant quality of a model in different clinical scenarios, DCA takes a model's calibration into account as well (Vickers et al., 2019), as you can see below.")
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
col_dca, col_cal, col_controls = st.columns([1, 1, 1])

# --- Column 3: Controls (Right Side) ---
with col_controls:
    # Model 1
    st.markdown(f"<h3 style='color:{COLOR_M1}'><b>Model 1</b></span>", unsafe_allow_html=True)
    cal_m1 = st.slider("Calibration", 0.1, 3.0, 0.4, 0.1, key="cal_m1")

    # Model 2
    st.markdown(f"<h3 style='color:{COLOR_M2}'><b>Model 2</b></span>", unsafe_allow_html=True)
    cal_m2 = st.slider("Calibration", 0.1, 3.0, 1.0, 0.1, key="cal_m2")

    # Model 3
    st.markdown(f"<h3 style='color:{COLOR_M3}'><b>Model 3</b></span>", unsafe_allow_html=True)
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
    with st.container(border=True):
        st.markdown("<span class='custom-card'></span>", unsafe_allow_html=True)
        plot_dca_multi_compare(comp_models, prev_comp, pt_comp)
        st_footer("<b>Figure 5.</b> Decision Curve Analysis for all three models compared to each other.")

with col_cal:
    with st.container(border=True):
        st.markdown("<span class='custom-card'></span>", unsafe_allow_html=True)
        plot_calibration_multi(comp_models)
        st_footer("<b>Figure 6.</b> Calibration plot illustrating the relation between the mean predicted probabilities and the true fraction of positives for all three models.")