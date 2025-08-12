import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# Page look & feel (dark)
# ---------------------------
st.set_page_config(page_title="Water Pollution Dashboard", layout="wide")
st.markdown(
    """
    <style>
        .main { background-color:#222; color:#f2f2f2; }
        .sidebar .sidebar-content { background-color:#333; color:#f2f2f2; }
        .title { text-align:center; font-size:2.2em; margin-top:6px; color:#f2f2f2; }
        .subtitle { text-align:center; font-size:1.2em; margin-bottom:20px; color:#ccc; }
        .eda-title { font-size:1.6em; font-weight:500; margin:6px 0 12px 0; }
        .prediction-result { padding:14px; border-radius:10px; color:white; text-align:center; margin-top:10px; font-size:1.1em; }
        .clean { background-color:#76c7c0; }
        .slightly-polluted { background-color:#ffcc00; color:#222; }
        .polluted { background-color:#ff6f61; }
    </style>
    <div class="title">💧 WATER POLLUTION OF KLANG RIVER DASHBOARD 🌊</div>
    <div class="subtitle">Analysis and Prediction </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Data loading / cleaning
# ---------------------------
@st.cache_data

def load_and_clean(path: str = "DataK.csv", fix_tc_row: int | None = 412):
    # 1) read
    df0 = pd.read_csv(path)
    df0.columns = df0.columns.str.strip()

    # 2) copy for cleaning
    df_raw = df0.copy()

    # 3) handle strings like "<1.000"
    def clean_special_values(value):
        if isinstance(value, str):
            s = value.strip()
            if s.startswith("<"):
                try:
                    return float(s.replace("<", "")) - 0.001
                except:
                    return np.nan
        try:
            return float(value)
        except:
            return np.nan

    # apply to selected columns if present
    cols_to_clean = [c for c in ["BOD", "SS", "NH3N"] if c in df_raw.columns]
    for col in cols_to_clean:
        df_raw[col] = df_raw[col].apply(clean_special_values)

    # 4) missing summary (for those cols)
    cleaned_missing_values = df_raw[cols_to_clean].isnull().sum()

    # 5) finalize df + NH3N median
    df = df_raw.copy()
    if "NH3N" in df.columns:
        df["NH3N"] = df["NH3N"].fillna(df["NH3N"].median())

    # 6) TC median override (optional)
    if "TC" in df.columns and fix_tc_row is not None and fix_tc_row in df.index:
        df.loc[fix_tc_row, "TC"] = df["TC"].median()

    
    var = [
    'DO', 'BOD', 'COD', 'SS',
    'pH', 'NH3N', 'TEMP', 'COND',
    'SAL', 'TUR', 'TC']
    
    df_selected= df[var]
    
    # 7) descriptive stats
    numeric_df = df_selected.select_dtypes(include=[np.number])
    desc_stats = numeric_df.describe().T
    desc_stats["skew"] = numeric_df.skew(numeric_only=True)
    desc_stats["kurtosis"] = numeric_df.kurtosis(numeric_only=True)
    desc_stats = desc_stats.round(2)

    # 8) helper lists (keep your existing names)
    numeric_cols = numeric_df.columns.tolist()
    feature_cols = [c for c in ["NH3N", "BOD", "COD", "DO", "pH", "SS"] if c in df.columns]

    # returns match your existing pattern + extras
    return df, numeric_cols, feature_cols, cleaned_missing_values, desc_stats


load_data = load_and_clean
df, numeric_cols, feature_cols, cleaned_missing_values, desc_stats = load_data("DataK.csv")

#st.success("Data loaded & cleaned")
#st.write(cleaned_missing_values)


# ---------------------------
# Sidebar navigation
# ---------------------------
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Home", "Water Pollution Prediction"])


# ---------------------------
# HOME (EDA)
# ---------------------------
if menu == "Home":
    
    with st.sidebar:
        st.markdown("### Filters")
        river_opt = ["All"] + sorted(df["river"].dropna().unique().tolist())
        station_opt = ["All"] + sorted(df["Station_id"].dropna().unique().tolist())
        sel_river = st.selectbox("River", river_opt)
        
        mask = pd.Series(True, index=df.index)
    if sel_river != "All":
        mask &= df["river"] == sel_river


    view = df.loc[mask].copy()
    if view.empty:
        st.warning("No rows match the current filters.")
        st.stop()


    
    st.markdown("<div class='eda-title'>Exploratory Data Analysis</div>", unsafe_allow_html=True)
    
    #Show descriptive statiscitcs 
    st.write("#### Descriptive stats for overall Klang River:")
    st.dataframe(desc_stats, use_container_width=True)

    st.write("#### Filtered data preview")
    st.dataframe(view.head(20), use_container_width=True)
    
     # Choose numeric parameter to explore
    num_choices = [c for c in numeric_cols if c != "Year"]
    default_idx = num_choices.index("NH3N") if "NH3N" in num_choices else 0
    column = st.selectbox("Select a parameter to visualize:", num_choices, index=default_idx, key="visualize_param")
    
    



    # ---------------- Row 1 ----------------
    # C1: Histogram | C2: Skewness (KDE)
    C1, C2 = st.columns(2)

    with C1:
        st.write("#### Histogram")
        st.write("")    # small vertical gap
        st.markdown("<br>", unsafe_allow_html=True)      # two breaks
        fig, ax = plt.subplots(facecolor="#333333")
        ax.hist(view[column].dropna(), bins=10, edgecolor="white")
        ax.set_title(f"Histogram of {column}", color="white")
        ax.set_xlabel(column, color="white"); ax.set_ylabel("Frequency", color="white")
        ax.tick_params(axis="x", colors="white"); ax.tick_params(axis="y", colors="white")
        st.pyplot(fig)

    with C2:
        st.write("#### Skewness")
        skewness = view[column].dropna().skew()
        st.caption(f"Skewness of {column}: **{skewness:.2f}**")
        fig, ax = plt.subplots(facecolor="#333333")
        # KDE line (uses seaborn for smooth density)
        import seaborn as sns
        sns.kdeplot(view[column].dropna(), ax=ax, fill=True)
        ax.set_title(f"KDE (Skewness line) of {column}", color="white")
        ax.set_xlabel(column, color="white"); ax.set_ylabel("Density", color="white")
        ax.tick_params(axis="x", colors="white"); ax.tick_params(axis="y", colors="white")
        st.pyplot(fig)
        st.markdown("<br>", unsafe_allow_html=True)   

    # ---------------- Row 2 ----------------
    # C3: Box plot | C4: Scatter plot
    C3, C4 = st.columns(2)

    with C3:
        st.write("#### Box Plot")
        st.write("") 
        st.markdown("<br><br>", unsafe_allow_html=True)      # two breaks
        fig, ax = plt.subplots(facecolor="#333333")
        ax.boxplot(view[column].dropna(), patch_artist=True)
        ax.set_title(f"Box Plot of {column}", color="white")
        ax.set_ylabel(column, color="white")
        ax.tick_params(axis="y", colors="white")
        ax.tick_params(axis="x", colors="white")
        st.pyplot(fig)
        st.markdown("<br>", unsafe_allow_html=True)   

    with C4:
        st.write("#### Scatter Plot")
        y_axis_choices = [c for c in num_choices if c != column]
        y_axis = st.selectbox("Select Y-axis parameter:", y_axis_choices, key="scatter_y")
        fig, ax = plt.subplots(facecolor="#333333")
        ax.scatter(view[column], view[y_axis], alpha=0.8)
        ax.set_title(f"{column} vs {y_axis}", color="white")
        ax.set_xlabel(column, color="white"); ax.set_ylabel(y_axis, color="white")
        ax.tick_params(axis="x", colors="white"); ax.tick_params(axis="y", colors="white")
        st.pyplot(fig)

    # ---------------- Row 3 ----------------
    # C5: Class distribution
    c5, _ = st.columns([1, 1])
    with c5:
        st.write("#### Pollution Class Distribution in Klang River")
        counts = view["Class"].value_counts().reindex(["Clean", "Slightly Polluted", "Polluted"]).fillna(0)
        labels = counts.index.tolist()
        sizes = counts.values
        colors = ["#76c7c0", "#ffcc00", "#ff6f61"]

        fig, ax = plt.subplots(facecolor="#333333")
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90,
            colors=colors, textprops={"color": "white"})
        ax.set_title("Klang River Pollution Class distribution", color="white")
        st.pyplot(fig)
    

# ---------------------------
# PREDICTION (trained on your CSV)
# ---------------------------
else:
    st.title("Water Pollution Prediction")
    st.caption("Gradient Boosting on fixed features with SMOTE: NH3N, BOD, COD, DO, SS.")

    from sklearn.ensemble import GradientBoostingClassifier
    from collections import Counter

    # must have SMOTE
    try:
        from imblearn.over_sampling import SMOTE
    except Exception:
        st.error("SMOTE requires 'imbalanced-learn'. Install with: pip install imbalanced-learn")
        st.stop()

    # fixed features (no feature selection)
    selected_features = ["NH3N", "BOD", "COD", "DO", "SS"]

    # sanity check
    needed = set(selected_features + ["Class"])
    if not needed.issubset(df.columns):
        st.error(f"Your CSV must include {', '.join(selected_features)} and Class.")
        st.stop()

    # --- train model on ALL available rows with SMOTE ---
    @st.cache_resource
    def train_model(df_):
        data_train = df_[selected_features + ["Class"]].dropna()
        if data_train.empty:
            raise ValueError("No complete rows for training.")

        X = data_train[selected_features].apply(pd.to_numeric, errors="coerce")
        y = data_train["Class"].astype(str)

        # choose a safe k_neighbors for SMOTE based on minority class size
        counts = Counter(y)
        min_class_count = min(counts.values())
        if min_class_count <= 1:
            # can't apply SMOTE with only 1 sample in a class
            st.warning("Not enough samples in a minority class for SMOTE. Training without SMOTE.")
            X_fit, y_fit = X, y
        else:
            k_neighbors = max(1, min(5, min_class_count - 1))
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_fit, y_fit = smote.fit_resample(X, y)

        model = GradientBoostingClassifier(random_state=42)
        model.fit(X_fit, y_fit)
        return model

    model = train_model(df)

    # ----------- UI: inputs (keep your variable names) -----------
    c1, c2, c3 = st.columns(3)
    with c1:
        do   = st.number_input("DO (mg/L)",   min_value=0.001, step=0.1,  value=8.692)
        bod  = st.number_input("BOD (mg/L)",  min_value=1.000, step=0.1,  value=3.000)
    with c2:
        cod  = st.number_input("COD (mg/L)",  min_value=1.000, step=0.1,  value=13.000)
        ss   = st.number_input("SS (mg/L)",   min_value=1.000, step=0.1,  value=3.000)
    with c3:
        nh3n = st.number_input("NH3N (mg/L)", min_value=0.001, step=0.1, value=0.370)
        

    # map inputs to a dict so we can build X_new in the right feature order
    _inputs = {"NH3N": nh3n, "BOD": bod, "COD": cod, "DO": do, "SS": ss}

    if st.button("Predict"):
        X_new = np.array([[ _inputs[f] for f in selected_features ]], dtype=float)

        pred = model.predict(X_new)[0]
        proba = model.predict_proba(X_new)[0]
        classes = list(model.classes_)
        #conf = float(proba[classes.index(pred)]) if pred in classes else None

        cls = pred
        color_class = "clean" if cls == "Clean" else ("slightly-polluted" if cls == "Slightly Polluted" else "polluted")
        #conf_txt = f" (confidence: {conf:.2f})" if conf is not None else ""
        st.markdown(
            f"""<div class="prediction-result {color_class}">
                Prediction: <b>{cls}</b>
            </div>""",
            unsafe_allow_html=True,
        )
        st.caption("Model: Gradient Boosting on NH3N, BOD, COD, DO, SS with SMOTE oversampling.")
