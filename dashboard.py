# dashboard.py
# Airline Passenger Satisfaction — Interactive Dashboard
# Run with:  streamlit run dashboard.py
# The script must be run from the project folder (the folder that contains train.csv).

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Airline Passenger Satisfaction",
    page_icon="✈️",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING  (cached so the file is only read once per session)
# ─────────────────────────────────────────────────────────────────────────────
@st.experimental_memo
def load_data():
    """Load train.csv from the project folder and return a cleaned DataFrame."""
    # Build a path that works whether the user runs the script from the
    # project folder or from a different working directory.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "train.csv")
    df = pd.read_csv(csv_path)

    # Fill the single column that has missing values
    df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(
        df["Arrival Delay in Minutes"].median()
    )

    # Derived columns used across multiple sections
    df["Satisfied"] = df["satisfaction"] == "satisfied"
    df["Departure Delay Status"] = np.where(
        df["Departure Delay in Minutes"] == 0, "No Delay", "Delayed"
    )
    # Age bands for age-based charts
    bins = [0, 17, 29, 44, 59, 100]
    labels = ["Under 18", "18–29", "30–44", "45–59", "60+"]
    df["Age Group"] = pd.cut(df["Age"], bins=bins, labels=labels)

    return df


df = load_data()

# Convenience: list of all service-rating columns (0–5 scale)
SERVICE_COLS = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Inflight service",
    "Cleanliness",
]

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
CLR_SATISFIED = "#2563EB"      # blue
CLR_DISSATISFIED = "#EF4444"   # red
CLR_NEUTRAL = "#94A3B8"        # slate
CLR_ACCENT = "#0EA5E9"         # sky blue (single-series bars)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def pct_fmt(val: float) -> str:
    """Format a 0-100 float as '42.3 %'."""
    return f"{val:.1f} %"


def satisfaction_rate(subset: pd.DataFrame) -> float:
    """Return % satisfied in a DataFrame subset."""
    if len(subset) == 0:
        return 0.0
    return subset["Satisfied"].mean() * 100


def make_grouped_bar(
    data: pd.DataFrame,
    group_col: str,
    title: str,
    figsize=(7, 4),
) -> plt.Figure:
    """
    Grouped horizontal bar chart showing satisfied vs neutral/dissatisfied
    counts for each category in group_col.
    """
    ct = pd.crosstab(data[group_col], data["satisfaction"])
    # Ensure consistent column order
    for col in ["neutral or dissatisfied", "satisfied"]:
        if col not in ct.columns:
            ct[col] = 0
    ct = ct[["satisfied", "neutral or dissatisfied"]]

    fig, ax = plt.subplots(figsize=figsize)
    categories = ct.index.tolist()
    x = np.arange(len(categories))
    width = 0.35

    bars_sat = ax.barh(x + width / 2, ct["satisfied"], width,
                       label="Satisfied", color=CLR_SATISFIED)
    bars_dis = ax.barh(x - width / 2, ct["neutral or dissatisfied"], width,
                       label="Neutral / Dissatisfied", color=CLR_DISSATISFIED)

    ax.set_yticks(x)
    ax.set_yticklabels(categories, fontsize=9)
    ax.set_xlabel("Number of Passengers", fontsize=9)
    ax.set_title(title, fontsize=10, pad=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.legend(fontsize=8, loc="lower right")

    for bar in bars_sat:
        w = bar.get_width()
        ax.text(w + max(ct.values.flatten()) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{int(w):,}", va="center", fontsize=7)
    for bar in bars_dis:
        w = bar.get_width()
        ax.text(w + max(ct.values.flatten()) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{int(w):,}", va="center", fontsize=7)

    plt.tight_layout()
    return fig


def make_satisfaction_rate_bar(
    data: pd.DataFrame,
    group_col: str,
    title: str,
    figsize=(7, 4),
    horizontal=True,
) -> plt.Figure:
    """Bar chart of satisfaction RATE (%) per category."""
    rates = (
        data.groupby(group_col)["Satisfied"]
        .mean()
        .mul(100)
        .sort_values(ascending=True)
        .reset_index()
    )
    rates.columns = [group_col, "Satisfaction Rate (%)"]

    fig, ax = plt.subplots(figsize=figsize)
    if horizontal:
        bars = ax.barh(rates[group_col], rates["Satisfaction Rate (%)"],
                       color=CLR_ACCENT)
        ax.set_xlabel("Satisfaction Rate (%)", fontsize=9)
        ax.axvline(50, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{w:.1f}%", va="center", fontsize=8)
        ax.set_xlim(0, 105)
    else:
        bars = ax.bar(rates[group_col], rates["Satisfaction Rate (%)"],
                      color=CLR_ACCENT)
        ax.set_ylabel("Satisfaction Rate (%)", fontsize=9)
        ax.axhline(50, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"{h:.1f}%", ha="center", fontsize=8)
        ax.set_ylim(0, 105)

    ax.set_title(title, fontsize=10, pad=8)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — FILTERS
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("🔍 Filters")
st.sidebar.markdown("Use the filters below to focus the dashboard on a specific passenger group.")

# Customer Type
all_customer_types = sorted(df["Customer Type"].unique().tolist())
sel_customer_types = st.sidebar.multiselect(
    "Customer Type",
    options=all_customer_types,
    default=all_customer_types,
)

# Type of Travel
all_travel_types = sorted(df["Type of Travel"].unique().tolist())
sel_travel_types = st.sidebar.multiselect(
    "Type of Travel",
    options=all_travel_types,
    default=all_travel_types,
)

# Travel Class
all_classes = sorted(df["Class"].unique().tolist())
sel_classes = st.sidebar.multiselect(
    "Travel Class",
    options=all_classes,
    default=all_classes,
)

# Gender
all_genders = sorted(df["Gender"].unique().tolist())
sel_genders = st.sidebar.multiselect(
    "Gender",
    options=all_genders,
    default=all_genders,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Filters apply to all charts and KPI cards below.  \n"
    "Deselecting all options in a filter resets it to 'all'."
)

# Apply filters (fall back to all if user deselects everything)
mask = pd.Series(True, index=df.index)
if sel_customer_types:
    mask &= df["Customer Type"].isin(sel_customer_types)
if sel_travel_types:
    mask &= df["Type of Travel"].isin(sel_travel_types)
if sel_classes:
    mask &= df["Class"].isin(sel_classes)
if sel_genders:
    mask &= df["Gender"].isin(sel_genders)

filtered = df[mask].copy()

# ─────────────────────────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────────────────────────
st.title("✈️ Airline Passenger Satisfaction Dashboard")
st.markdown(
    "An interactive overview of satisfaction patterns across **{:,} passengers** "
    "(filtered view: **{:,} passengers**).  \n"
    "Use the sidebar filters to drill into specific passenger groups.".format(
        len(df), len(filtered)
    )
)

if len(filtered) == 0:
    st.warning("No passengers match the current filters. Please adjust the sidebar selections.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — KPI CARDS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Key Performance Indicators")

total = len(filtered)
satisfied_n = filtered["Satisfied"].sum()
dissatisfied_n = total - satisfied_n
sat_pct = satisfied_n / total * 100

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric(
    label="Total Passengers (filtered)",
    value=f"{total:,}",
)
kpi2.metric(
    label="✅ Satisfied",
    value=f"{satisfied_n:,}",
    delta=pct_fmt(sat_pct),
)
kpi3.metric(
    label="❌ Neutral / Dissatisfied",
    value=f"{dissatisfied_n:,}",
    delta=pct_fmt(100 - sat_pct),
    delta_color="inverse",
)
kpi4.metric(
    label="Satisfaction Rate",
    value=pct_fmt(sat_pct),
    delta=f"{sat_pct - 50:.1f} pp vs 50 % baseline",
    delta_color="normal",
)

st.caption(
    "KPIs reflect the currently filtered passenger group. "
    "Remove all sidebar filters to see the full dataset."
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — SATISFACTION BY PASSENGER GROUP
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("👥 Satisfaction by Passenger Group")
st.markdown(
    "The charts below show **satisfaction counts** and **satisfaction rates** "
    "for the four passenger-level dimensions. "
    "Note: observed associations — not causal claims."
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Customer Type", "Type of Travel", "Travel Class", "Gender"]
)

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(
            make_grouped_bar(filtered, "Customer Type",
                             "Satisfied vs Neutral/Dissatisfied by Customer Type"),
        )
    with c2:
        st.pyplot(
            make_satisfaction_rate_bar(filtered, "Customer Type",
                                       "Satisfaction Rate (%) by Customer Type"),
        )
    ct_rate = filtered.groupby("Customer Type")["Satisfied"].mean().mul(100).round(1)
    st.caption("Satisfaction rates: " + "  |  ".join(
        f"**{k}**: {v:.1f}%" for k, v in ct_rate.items()
    ))

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(
            make_grouped_bar(filtered, "Type of Travel",
                             "Satisfied vs Neutral/Dissatisfied by Type of Travel"),
        )
    with c2:
        st.pyplot(
            make_satisfaction_rate_bar(filtered, "Type of Travel",
                                       "Satisfaction Rate (%) by Type of Travel"),
        )
    tt_rate = filtered.groupby("Type of Travel")["Satisfied"].mean().mul(100).round(1)
    st.caption("Satisfaction rates: " + "  |  ".join(
        f"**{k}**: {v:.1f}%" for k, v in tt_rate.items()
    ))

with tab3:
    class_order = ["Business", "Eco Plus", "Eco"]
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(
            make_grouped_bar(filtered, "Class",
                             "Satisfied vs Neutral/Dissatisfied by Travel Class"),
        )
    with c2:
        st.pyplot(
            make_satisfaction_rate_bar(filtered, "Class",
                                       "Satisfaction Rate (%) by Travel Class"),
        )
    cl_rate = filtered.groupby("Class")["Satisfied"].mean().mul(100).round(1)
    st.caption("Satisfaction rates: " + "  |  ".join(
        f"**{k}**: {v:.1f}%" for k, v in cl_rate.items()
    ))

with tab4:
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(
            make_grouped_bar(filtered, "Gender",
                             "Satisfied vs Neutral/Dissatisfied by Gender"),
        )
    with c2:
        st.pyplot(
            make_satisfaction_rate_bar(filtered, "Gender",
                                       "Satisfaction Rate (%) by Gender"),
        )
    gd_rate = filtered.groupby("Gender")["Satisfied"].mean().mul(100).round(1)
    st.caption("Satisfaction rates: " + "  |  ".join(
        f"**{k}**: {v:.1f}%" for k, v in gd_rate.items()
    ))

# Age Group mini-section
st.markdown("#### Satisfaction Rate by Age Group")
fig_age = make_satisfaction_rate_bar(
    filtered, "Age Group",
    "Satisfaction Rate (%) by Age Group",
    figsize=(7, 3),
    horizontal=False,
)
col_age, _ = st.columns([2, 1])
with col_age:
    st.pyplot(fig_age)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — SERVICE RATINGS ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("⭐ Service Ratings Analysis")
st.markdown(
    "Ratings are on a **0–5 scale** (0 = not applicable, 5 = excellent).  \n"
    "The chart compares mean ratings between satisfied and neutral/dissatisfied passengers."
)

# Mean rating per service per satisfaction group
mean_ratings = (
    filtered.groupby("satisfaction")[SERVICE_COLS]
    .mean()
    .T
    .round(2)
)

# Rename columns for cleaner display
col_rename = {
    "neutral or dissatisfied": "Neutral / Dissatisfied",
    "satisfied": "Satisfied",
}
mean_ratings = mean_ratings.rename(columns=col_rename)
mean_ratings["Difference (Sat − Dis)"] = (
    mean_ratings.get("Satisfied", 0) - mean_ratings.get("Neutral / Dissatisfied", 0)
).round(2)
mean_ratings = mean_ratings.sort_values("Difference (Sat − Dis)", ascending=False)

# Chart: mean rating comparison
fig_svc, ax_svc = plt.subplots(figsize=(9, 5))
services = mean_ratings.index.tolist()
x_svc = np.arange(len(services))
w = 0.35

if "Satisfied" in mean_ratings.columns:
    ax_svc.barh(x_svc + w / 2, mean_ratings["Satisfied"], w,
                label="Satisfied", color=CLR_SATISFIED)
if "Neutral / Dissatisfied" in mean_ratings.columns:
    ax_svc.barh(x_svc - w / 2, mean_ratings["Neutral / Dissatisfied"], w,
                label="Neutral / Dissatisfied", color=CLR_DISSATISFIED)

ax_svc.set_yticks(x_svc)
ax_svc.set_yticklabels(services, fontsize=8)
ax_svc.set_xlabel("Mean Rating", fontsize=9)
ax_svc.set_title("Mean Service Ratings: Satisfied vs Neutral/Dissatisfied", fontsize=10)
ax_svc.set_xlim(0, 5.5)
ax_svc.legend(fontsize=8)
plt.tight_layout()

c_svc1, c_svc2 = st.columns([3, 2])
with c_svc1:
    st.pyplot(fig_svc)
with c_svc2:
    st.markdown("**Mean ratings by group**")
    display_cols = [c for c in ["Satisfied", "Neutral / Dissatisfied", "Difference (Sat − Dis)"]
                    if c in mean_ratings.columns]
    st.dataframe(mean_ratings[display_cols].style.format("{:.2f}"), height=420)

st.caption(
    "Services with the largest positive difference are most strongly associated "
    "with satisfaction in this dataset. Differences do not establish causation."
)

# Distribution of a selected service rating
st.markdown("#### Explore a Single Service Rating")
selected_service = st.selectbox(
    "Select a service to inspect",
    options=SERVICE_COLS,
    index=SERVICE_COLS.index("Online boarding"),
)

rating_counts = (
    filtered.groupby([selected_service, "satisfaction"])
    .size()
    .unstack(fill_value=0)
    .rename(columns=col_rename)
)

fig_dist, ax_dist = plt.subplots(figsize=(7, 3.5))
ratings_x = rating_counts.index.tolist()
bar_w = 0.35
x_r = np.arange(len(ratings_x))

if "Satisfied" in rating_counts.columns:
    ax_dist.bar(x_r + bar_w / 2, rating_counts["Satisfied"], bar_w,
                label="Satisfied", color=CLR_SATISFIED)
if "Neutral / Dissatisfied" in rating_counts.columns:
    ax_dist.bar(x_r - bar_w / 2, rating_counts["Neutral / Dissatisfied"], bar_w,
                label="Neutral / Dissatisfied", color=CLR_DISSATISFIED)

ax_dist.set_xticks(x_r)
ax_dist.set_xticklabels([str(r) for r in ratings_x], fontsize=9)
ax_dist.set_xlabel("Rating (0 = not applicable, 5 = excellent)", fontsize=9)
ax_dist.set_ylabel("Number of Passengers", fontsize=9)
ax_dist.set_title(f"Passenger Counts by '{selected_service}' Rating", fontsize=10)
ax_dist.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax_dist.legend(fontsize=8)
plt.tight_layout()

col_dist, _ = st.columns([2, 1])
with col_dist:
    st.pyplot(fig_dist)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — DELAY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("⏱️ Delay Analysis")
st.markdown(
    "Departure and arrival delays are measured in minutes.  \n"
    "The majority of flights have **zero delay**; outliers extend to over 1 500 minutes."
)

col_d1, col_d2 = st.columns(2)

# -- Satisfaction by delay status
with col_d1:
    st.markdown("**Satisfaction by Departure Delay Status**")
    delay_ct = pd.crosstab(
        filtered["Departure Delay Status"], filtered["satisfaction"]
    ).rename(columns=col_rename)

    fig_del, ax_del = plt.subplots(figsize=(5, 3))
    statuses = delay_ct.index.tolist()
    x_del = np.arange(len(statuses))
    w_del = 0.35
    if "Satisfied" in delay_ct.columns:
        ax_del.bar(x_del + w_del / 2, delay_ct["Satisfied"], w_del,
                   label="Satisfied", color=CLR_SATISFIED)
    if "Neutral / Dissatisfied" in delay_ct.columns:
        ax_del.bar(x_del - w_del / 2, delay_ct["Neutral / Dissatisfied"], w_del,
                   label="Neutral / Dissatisfied", color=CLR_DISSATISFIED)
    ax_del.set_xticks(x_del)
    ax_del.set_xticklabels(statuses, fontsize=9)
    ax_del.set_ylabel("Passengers", fontsize=9)
    ax_del.set_title("Satisfaction by Departure Delay Status", fontsize=9)
    ax_del.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax_del.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig_del)

    delay_rate = filtered.groupby("Departure Delay Status")["Satisfied"].mean().mul(100).round(1)
    st.caption("Satisfaction rates: " + "  |  ".join(
        f"**{k}**: {v:.1f}%" for k, v in delay_rate.items()
    ))

# -- Delay distribution histogram (log-scale for readability)
with col_d2:
    st.markdown("**Distribution of Departure Delay (minutes)**")
    delayed_only = filtered[filtered["Departure Delay in Minutes"] > 0][
        "Departure Delay in Minutes"
    ]

    fig_hist, ax_hist = plt.subplots(figsize=(5, 3))
    if len(delayed_only) > 0:
        ax_hist.hist(delayed_only, bins=40, color=CLR_ACCENT, edgecolor="white", linewidth=0.3)
        ax_hist.set_xlabel("Delay (minutes)", fontsize=9)
        ax_hist.set_ylabel("Number of Flights (log scale)", fontsize=9)
        ax_hist.set_title("Departure Delay Distribution (delayed flights only)", fontsize=9)
        ax_hist.set_yscale("log")
        ax_hist.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{int(v):,}")
        )
        plt.tight_layout()
    else:
        ax_hist.text(0.5, 0.5, "No delayed flights in current filter",
                     ha="center", va="center", transform=ax_hist.transAxes)
    st.pyplot(fig_hist)
    if len(delayed_only) > 0:
        st.caption(
            f"Of the **{len(filtered):,}** filtered passengers, "
            f"**{len(delayed_only):,}** ({len(delayed_only)/len(filtered)*100:.1f}%) "
            f"experienced a departure delay. Median delay: "
            f"**{int(delayed_only.median())} min**; max: **{int(delayed_only.max())} min**."
        )

# -- Mean delay by satisfaction group
st.markdown("**Mean Delays by Satisfaction Group**")
delay_stats = filtered.groupby("satisfaction")[
    ["Departure Delay in Minutes", "Arrival Delay in Minutes"]
].mean().round(1).rename(index={"satisfied": "Satisfied",
                                 "neutral or dissatisfied": "Neutral / Dissatisfied"})
delay_stats.columns = ["Mean Departure Delay (min)", "Mean Arrival Delay (min)"]
st.dataframe(delay_stats)
st.caption(
    "Differences in mean delays between groups are small in absolute terms. "
    "Association with satisfaction is observed but not causal."
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — INSIGHTS & RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("💡 Insights & Recommendations")

# Compute dynamic values for insight bullets
overall_sat_pct = df["Satisfied"].mean() * 100
biz_class_sat = df[df["Class"] == "Business"]["Satisfied"].mean() * 100
eco_sat = df[df["Class"] == "Eco"]["Satisfied"].mean() * 100
loyal_sat = df[df["Customer Type"] == "Loyal Customer"]["Satisfied"].mean() * 100
disloyal_sat = df[df["Customer Type"] == "disloyal Customer"]["Satisfied"].mean() * 100
biz_travel_sat = df[df["Type of Travel"] == "Business travel"]["Satisfied"].mean() * 100
personal_sat = df[df["Type of Travel"] == "Personal Travel"]["Satisfied"].mean() * 100
male_sat = df[df["Gender"] == "Male"]["Satisfied"].mean() * 100
female_sat = df[df["Gender"] == "Female"]["Satisfied"].mean() * 100

# Top 3 services by difference
top_services = mean_ratings["Difference (Sat − Dis)"].head(3).index.tolist() if "Difference (Sat − Dis)" in mean_ratings.columns else []

st.markdown("### 📌 Key Observations (full dataset)")

st.markdown(f"""
| Dimension | Finding |
|-----------|---------|
| **Overall Satisfaction** | **{overall_sat_pct:.1f}%** of passengers were satisfied; **{100-overall_sat_pct:.1f}%** were neutral or dissatisfied. |
| **Travel Class** | Business class passengers were satisfied at **{biz_class_sat:.1f}%**, compared to **{eco_sat:.1f}%** in Economy. A large gap exists across classes. |
| **Customer Loyalty** | Loyal customers were satisfied at **{loyal_sat:.1f}%** vs **{disloyal_sat:.1f}%** for disloyal customers. |
| **Travel Purpose** | Business travellers: **{biz_travel_sat:.1f}%** satisfied. Personal travellers: **{personal_sat:.1f}%** satisfied — a substantial difference. |
| **Gender** | Male: **{male_sat:.1f}%** satisfied; Female: **{female_sat:.1f}%** satisfied. Gender shows a small difference in this dataset. |
| **Service Ratings** | Services with the largest gap between satisfied and dissatisfied groups (filtered view): **{", ".join(top_services) if top_services else "see chart above"}**. |
""")

st.info(
    "⚠️ **Correlation ≠ Causation.** All findings above are observed associations "
    "in this dataset. They do not prove that any one factor directly causes "
    "satisfaction or dissatisfaction."
)

st.markdown("### 🎯 Practical Recommendations")

st.markdown("""
**1. Economy & Eco Plus Experience**
The satisfaction gap between Business and Economy classes is large. Targeted improvements to
seat comfort, cleanliness, and onboard service in economy cabins could close this gap.
Investigation into which specific service attributes economy passengers rate lowest is a
logical next step.

**2. Disloyal / First-Time Passenger Onboarding**
Disloyal customers show substantially lower satisfaction rates. Improving the first-flight
experience — ease of online booking, check-in, and gate information — may increase the
likelihood of passengers returning and becoming loyal.

**3. Personal Traveller Experience**
Personal travellers are markedly less satisfied than business travellers. Airlines may
consider tailoring offerings (entertainment, food, flexibility) to leisure passengers,
who may have different expectations.

**4. Service Quality Focus Areas**
Based on the service rating gap in the filtered view, prioritise the services listed in
the "Service Ratings" section above. Consistent improvements in high-gap areas are likely
to move satisfaction scores.

**5. Delay Management**
While the mean delay difference between satisfied and dissatisfied groups is modest,
minimising unexpected delays — especially long-tail events — and communicating delays
proactively can protect the passenger experience.

**6. Further Analysis**
This dashboard is descriptive. A predictive model (e.g. logistic regression or a
decision tree) could identify the strongest independent drivers of satisfaction and
quantify their relative importance.
""")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Data source: Airline Passenger Satisfaction dataset (train.csv).  "
    "Dashboard built with Streamlit · Pandas · Matplotlib.  "
    "All observations are associational and should not be interpreted as causal."
)
