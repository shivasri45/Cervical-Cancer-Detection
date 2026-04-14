import logging
import io
import streamlit as st
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Cervical Cancer Detection", layout="wide", page_icon="🧬")

st.markdown("""
<style>
/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
}
[data-testid="stMetricLabel"] { color: #a0a0b8; font-size: 0.85rem; }
[data-testid="stMetricValue"] { color: #ffffff; font-weight: 700; }

/* Tab bar */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(30,30,47,0.5);
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 500;
}

/* Sidebar status */
.sidebar-status { font-size: 0.9rem; line-height: 1.8; }
.sidebar-status .done { color: #4ade80; }
.sidebar-status .pending { color: #6b7280; }

/* Primary buttons */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(99,102,241,0.4);
}

/* Expanders */
.streamlit-expanderHeader { font-weight: 600; }

/* Divider */
hr { border-color: rgba(255,255,255,0.06) !important; }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize all session-state variables to prevent resets."""
    defaults = {
        "data": None, "cleaned_data": None, "file_id": None,
        "target": None, "selected_features": [], "split": None,
        "model_name": None, "model": None, "trained_model": None,
        "imputed": False, "outliers_removed": False,
        "k_folds": 5, "cv_scores": None,
        "train_columns": None, "target_encoder": None,
        "tuned_model": None, "best_params": None,
        "outlier_count": 0, "scaler": None,
        "feature_importances": None,
        "problem_type": "Classification",
        "last_prediction": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


def _reset_downstream(*keep):
    """Reset all downstream state keys when upstream data changes."""
    keys_to_reset = {
        "split", "trained_model", "tuned_model", "best_params",
        "cv_scores", "train_columns", "target_encoder",
        "imputed", "outliers_removed", "outlier_count",
        "scaler", "feature_importances",
    }
    for k in keys_to_reset - set(keep):
        if k in ("imputed", "outliers_removed"):
            st.session_state[k] = False
        elif k == "outlier_count":
            st.session_state[k] = 0
        else:
            st.session_state[k] = None


@st.cache_data(show_spinner=False)
def load_file(uploaded_file):
    """Cache-friendly file loader supporting CSV and Excel."""
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file, na_values=["?", "NA", "N/A", ""])
        elif name.endswith((".xls", ".xlsx")):
            return pd.read_excel(uploaded_file, na_values=["?", "NA", "N/A", ""])
    except Exception as e:
        logger.error("File load failed: %s", e)
        st.error(f"Failed to load file: {e}")
        return None


def sidebar_status():
    """Render a pipeline progress tracker in the sidebar."""
    steps = [
        ("1. Patient Data", st.session_state["data"] is not None),
        ("2. EDA", st.session_state["data"] is not None),
        ("3. Data Cleaning", st.session_state["imputed"] or st.session_state["outliers_removed"]),
        ("4. Risk Factors", len(st.session_state["selected_features"]) > 0),
        ("5. Data Split", st.session_state["split"] is not None),
        ("6. Model Setup", st.session_state["model"] is not None),
        ("7. Train & Validate", st.session_state["trained_model"] is not None),
        ("8. Diagnostics", st.session_state["trained_model"] is not None),
        ("9. Tuning", st.session_state["tuned_model"] is not None),
        ("10. Screen Patient", False),
    ]
    html = '<div class="sidebar-status">'
    for label, done in steps:
        icon = "✅" if done else "⬜"
        cls = "done" if done else "pending"
        html += f'<span class="{cls}">{icon} {label}</span><br>'
    html += "</div>"
    st.sidebar.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 3. TAB FUNCTIONS
# ---------------------------------------------------------------------------

def tab_data_input():
    st.header("1️⃣ Patient Data Input")
    st.caption("Upload cervical cancer screening data (e.g., UCI Cervical Cancer Risk Factors dataset).")
    uploaded_file = st.file_uploader(
        "Upload Patient Dataset",
        type=["csv", "xls", "xlsx"],
        help="Upload CSV/Excel with patient risk factors and biopsy results."
    )

    if uploaded_file:
        if st.session_state["file_id"] != uploaded_file.file_id:
            with st.spinner("Loading dataset…"):
                df = load_file(uploaded_file)
            if df is not None:
                st.session_state["data"] = df
                st.session_state["cleaned_data"] = df.copy()
                st.session_state["file_id"] = uploaded_file.file_id
                _reset_downstream()
                logger.info("Loaded %s — %d rows × %d cols", uploaded_file.name, *df.shape)

    if st.session_state["data"] is None:
        st.info("👆 Upload a cervical cancer dataset (CSV or Excel) to begin screening analysis.")
        return

    df = st.session_state["data"]

    col1, col2 = st.columns([1, 2])
    with col1:
        target_col = st.selectbox(
            "🎯 Select Diagnosis Target", df.columns,
            index=len(df.columns) - 1,
            help="The biopsy/diagnosis column to predict (e.g., Biopsy, Hinselmann, Schiller, Citology)."
        )
        st.session_state["target"] = target_col

        if df[target_col].isnull().sum() > 0:
            before = len(df)
            df = df.dropna(subset=[target_col])
            st.session_state["data"] = df
            st.session_state["cleaned_data"] = st.session_state["cleaned_data"].dropna(subset=[target_col])
            st.warning(f"Dropped {before - len(df)} rows with missing target.")

        c1, c2 = st.columns(2)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", df.shape[1])

        # Quick data-quality badges
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        if missing_pct > 0:
            st.warning(f"⚠️ {missing_pct:.1f}% missing values")
        else:
            st.success("✅ No missing values")

        n_unique = df[target_col].nunique()
        st.caption(f"Diagnosis target has **{n_unique}** unique values")

    with col2:
        st.write("### Raw Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

    # PCA Visualization
    st.divider()
    st.subheader("Patient Data — PCA Projection")
    feature_cols = st.multiselect(
        "Select features for PCA (numeric only)",
        [c for c in df.columns if c != target_col],
        default=[c for c in df.select_dtypes(include=[np.number]).columns if c != target_col][:5]
    )

    if len(feature_cols) >= 2:
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            pca_data = df[feature_cols].select_dtypes(include=[np.number]).dropna()
            if pca_data.shape[0] > 0 and pca_data.shape[1] >= 2:
                import plotly.express as px
                scaled = StandardScaler().fit_transform(pca_data)
                components = PCA(n_components=2).fit_transform(scaled)
                fig = px.scatter(
                    x=components[:, 0], y=components[:, 1],
                    color=df.loc[pca_data.index, target_col].astype(str),
                    labels={"x": "PC 1", "y": "PC 2", "color": target_col},
                    title="2-D PCA Projection",
                    template="plotly_dark",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig.update_layout(margin=dict(t=40, b=20, l=20, r=20))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough numeric columns or rows for PCA. Impute data first in Tab 3.")
        except Exception as e:
            logger.error("PCA failed: %s", e)
            st.error(f"PCA visualization error: {e}")


def tab_eda():
    st.header("2️⃣ Exploratory Data Analysis")
    st.caption("Analyze patient risk factor distributions and data quality.")
    if st.session_state["data"] is None:
        st.info("Please upload data in Step 1.")
        return

    df = st.session_state["data"]
    import plotly.express as px

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(include="all").T, use_container_width=True)

    with col2:
        st.subheader("Missing Values")
        missing = df.isnull().sum().reset_index()
        missing.columns = ["Feature", "Missing Count"]
        missing = missing[missing["Missing Count"] > 0].sort_values("Missing Count", ascending=False)

        if not missing.empty:
            fig = px.bar(
                missing, x="Feature", y="Missing Count",
                title="Missing Values per Feature", text_auto=True,
                color="Missing Count", color_continuous_scale="Reds",
                template="plotly_dark",
            )
            fig.update_layout(margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values found!")

    # Data profiling expander
    with st.expander("📊 Data Profile", expanded=False):
        profile = pd.DataFrame({
            "Type": df.dtypes.astype(str),
            "Non-Null": df.count(),
            "Null": df.isnull().sum(),
            "Unique": df.nunique(),
            "Memory (KB)": (df.memory_usage(deep=True) / 1024).round(2),
        })
        st.dataframe(profile, use_container_width=True)

    # Distribution plots
    st.divider()
    st.subheader("Risk Factor Distributions")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        dist_col = st.selectbox("Select feature to visualize", num_cols)
        fig = px.histogram(
            df, x=dist_col, marginal="box",
            title=f"Distribution of {dist_col}", template="plotly_dark",
            color_discrete_sequence=["#6366f1"],
        )
        fig.update_layout(margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)


def tab_cleaning():
    st.header("3️⃣ Data Cleaning & Preprocessing")
    st.caption("Handle missing values and outliers in patient records.")
    if st.session_state["cleaned_data"] is None:
        st.info("Please upload data in Step 1.")
        return

    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import IsolationForest

    df_clean = st.session_state["cleaned_data"].copy()

    col1, col2 = st.columns(2)

    # ---- Imputation ----
    with col1:
        st.subheader("1. Imputation")
        impute_strategy = st.selectbox("Missing Value Strategy", ["mean", "median", "most_frequent"])
        if st.button("Apply Imputation", type="primary", key="btn_impute"):
            try:
                num_cols = df_clean.select_dtypes(include=[np.number]).columns
                cat_cols = df_clean.select_dtypes(exclude=[np.number]).columns

                if len(num_cols) > 0:
                    imp = SimpleImputer(strategy=impute_strategy)
                    df_clean[num_cols] = imp.fit_transform(df_clean[num_cols])
                if len(cat_cols) > 0:
                    cat_imp = SimpleImputer(strategy="most_frequent")
                    df_clean[cat_cols] = cat_imp.fit_transform(df_clean[cat_cols])

                st.session_state["cleaned_data"] = df_clean
                st.session_state["imputed"] = True
                logger.info("Imputation applied (%s)", impute_strategy)
            except Exception as e:
                logger.error("Imputation failed: %s", e)
                st.error(f"Imputation failed: {e}")

        if st.session_state["imputed"]:
            remaining = st.session_state["cleaned_data"].isnull().sum().sum()
            st.success(f"✅ Imputation applied! Remaining nulls: {remaining}")

    # ---- Outlier Detection ----
    with col2:
        st.subheader("2. Outlier Detection")
        outlier_method = st.selectbox("Method", ["IQR", "Isolation Forest"])
        num_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
        outlier_cols = st.multiselect(
            "Select columns for outlier detection:",
            num_cols, default=num_cols[:2] if len(num_cols) >= 2 else num_cols
        )

        if st.button("Detect & Remove Outliers", type="primary", key="btn_outlier"):
            if not outlier_cols:
                st.warning("Select at least one column.")
            elif df_clean[outlier_cols].isnull().values.any():
                st.error("Missing values detected in selected columns. Apply imputation first.")
            else:
                try:
                    X_out = df_clean[outlier_cols]
                    if outlier_method == "Isolation Forest":
                        preds = IsolationForest(
                            contamination=0.05, random_state=42
                        ).fit_predict(X_out)
                        mask = preds != -1
                    else:
                        Q1, Q3 = X_out.quantile(0.25), X_out.quantile(0.75)
                        IQR = Q3 - Q1
                        mask = ~((X_out < (Q1 - 1.5 * IQR)) | (X_out > (Q3 + 1.5 * IQR))).any(axis=1)

                    removed = int((~mask).sum())
                    st.session_state["cleaned_data"] = df_clean[mask].reset_index(drop=True).copy()
                    st.session_state["outliers_removed"] = True
                    st.session_state["outlier_count"] = removed
                    logger.info("Removed %d outliers via %s", removed, outlier_method)
                except Exception as e:
                    logger.error("Outlier removal failed: %s", e)
                    st.error(f"Outlier removal failed: {e}")

        if st.session_state["outliers_removed"]:
            st.success(f"✅ Removed {st.session_state['outlier_count']} outliers!")

    # ---- Preview & download ----
    with st.expander("Preview Cleaned Data"):
        st.dataframe(st.session_state["cleaned_data"].head(10), use_container_width=True)

    csv = st.session_state["cleaned_data"].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Cleaned Data",
        data=csv, file_name="cleaned_data.csv",
        mime="text/csv", use_container_width=True,
    )


def tab_feature_selection():
    st.header("4️⃣ Risk Factor Selection")
    st.caption("Identify the most relevant risk factors for cervical cancer prediction.")
    if st.session_state["cleaned_data"] is None:
        st.info("Please upload data in Step 1.")
        return

    import plotly.express as px
    from sklearn.preprocessing import LabelEncoder

    # Work on a COPY so we never mutate cleaned_data
    df_fs = st.session_state["cleaned_data"].copy()
    target = st.session_state["target"]

    if target is None:
        st.warning("Please select a target in Tab 1.")
        return

    st.subheader("Risk Factor Correlation with Diagnosis")

    # Encode target temporarily if categorical
    if df_fs[target].dtype == "O" or df_fs[target].dtype.name == "category":
        df_fs[target] = LabelEncoder().fit_transform(df_fs[target].astype(str))

    num_df = df_fs.select_dtypes(include=[np.number])
    if target not in num_df.columns:
        st.error("Target could not be processed for correlation. Check data types.")
        return

    corr = num_df.corr()[target].drop(target, errors="ignore").sort_values(ascending=False)

    threshold = st.slider("Absolute Correlation Threshold", 0.0, 1.0, 0.05, step=0.01)
    selected_features = corr[abs(corr) > threshold].index.tolist()
    st.session_state["selected_features"] = selected_features

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.bar(
            x=corr.index, y=corr.values,
            labels={"x": "Feature", "y": f"Correlation with {target}"},
            title="Feature Correlations",
            template="plotly_dark",
            color=abs(corr.values),
            color_continuous_scale="Viridis",
        )
        fig.update_layout(margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("Features Selected", len(selected_features))
        if selected_features:
            st.write(selected_features)
        else:
            st.warning("No features pass the threshold. Try lowering it.")


def tab_data_split():
    st.header("5️⃣ Data Split")
    st.caption("Split patient records into training and testing sets.")
    problem_type = st.session_state["problem_type"]

    if st.session_state["cleaned_data"] is None or not st.session_state.get("selected_features"):
        st.info("Please complete Feature Selection first (Tab 4).")
        return

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    test_size = st.slider("Test Size (%)", 10, 50, 20, step=5) / 100
    use_scaling = st.checkbox("Apply StandardScaler to features", value=False,
                              help="Recommended for SVM and Logistic/Linear Regression.")

    if st.button("Split Data", type="primary", key="btn_split"):
        try:
            X = st.session_state["cleaned_data"][st.session_state["selected_features"]].copy()
            y = st.session_state["cleaned_data"][st.session_state["target"]].copy()

            # Validate
            if len(X) < 10:
                st.error("Dataset too small (< 10 rows). Upload more data.")
                return
            if problem_type == "Classification" and y.nunique() < 2:
                st.error("Target has fewer than 2 classes. Cannot classify.")
                return
            if problem_type == "Classification" and y.nunique() > 50:
                st.warning("⚠️ Target has 50+ classes. Consider regression or grouping classes.")

            # Encode target
            if problem_type == "Classification" and y.dtype == "O":
                le = LabelEncoder()
                y = pd.Series(le.fit_transform(y.astype(str)), name=y.name)
                st.session_state["target_encoder"] = le
            else:
                st.session_state["target_encoder"] = None

            # Dummify features BEFORE split
            X = pd.get_dummies(X, drop_first=True)
            st.session_state["train_columns"] = list(X.columns)

            # Optional scaling
            if use_scaling:
                scaler = StandardScaler()
                X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
                st.session_state["scaler"] = scaler
            else:
                st.session_state["scaler"] = None

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            st.session_state["split"] = (X_train, X_test, y_train, y_test)

            # Reset downstream models
            for k in ["trained_model", "tuned_model", "best_params", "cv_scores", "feature_importances"]:
                st.session_state[k] = None

            logger.info("Data split: train=%d, test=%d", len(X_train), len(X_test))
        except Exception as e:
            logger.error("Data split failed: %s", e)
            st.error(f"Data split failed: {e}")

    if st.session_state["split"] is not None:
        X_train, X_test, _, _ = st.session_state["split"]
        st.success("✅ Data split successfully!")
        c1, c2, c3 = st.columns(3)
        c1.metric("Training Samples", f"{len(X_train):,}")
        c2.metric("Test Samples", f"{len(X_test):,}")
        c3.metric("Features", X_train.shape[1])


def tab_model_setup():
    st.header("6️⃣ Model Setup")
    st.caption("Choose a classification algorithm for cervical cancer detection.")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression

    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    }

    name = st.selectbox("Choose a Classification Algorithm", list(models.keys()))
    st.session_state["model_name"] = name

    if name == "SVM":
        kernel = st.selectbox("SVM Kernel", ["linear", "poly", "rbf", "sigmoid"])
        st.session_state["model"] = SVC(kernel=kernel, probability=True, random_state=42)
    else:
        st.session_state["model"] = models[name]

    st.info(f"🔧 Selected model: **{name}**")

    # Model info card
    with st.expander("ℹ️ Model Details"):
        info = {
            "Random Forest": "Ensemble of decision trees. Excellent for medical screening — handles complex interactions between risk factors and is robust to noisy data.",
            "SVM": "Support Vector Machine. Effective for high-dimensional medical datasets. Use with feature scaling for best results.",
            "Logistic Regression": "Interpretable linear model. Provides clear odds ratios for each risk factor — useful for clinical reporting.",
        }
        st.write(info.get(name, ""))


def tab_train_evaluate():
    st.header("7️⃣ Train & Validate")
    st.caption("Train the model on patient data and validate with cross-validation.")
    problem_type = st.session_state["problem_type"]

    from sklearn.model_selection import cross_val_score
    from sklearn.base import clone

    k_folds = st.number_input(
        "K-Fold Cross Validation", min_value=2, max_value=10,
        value=st.session_state["k_folds"]
    )
    st.session_state["k_folds"] = k_folds

    if st.button("Train Model", type="primary", key="btn_train"):
        if st.session_state.get("split") is None:
            st.error("Please split the data in Tab 5 first.")
            return
        if st.session_state.get("model") is None:
            st.error("Please select a model in Tab 6 first.")
            return

        X_train, X_test, y_train, y_test = st.session_state["split"]

        if X_train.isnull().values.any():
            st.error("🚨 Missing values in training data. Apply Imputation in Tab 3.")
            return

        try:
            model = clone(st.session_state["model"])
            scoring = "accuracy" if problem_type == "Classification" else "neg_mean_squared_error"

            progress = st.progress(0, text="Running cross-validation...")
            scores = cross_val_score(model, X_train, y_train, cv=int(k_folds), scoring=scoring)
            progress.progress(50, text="Fitting model on full training set...")

            model.fit(X_train, y_train)
            progress.progress(100, text="Training complete!")

            st.session_state["trained_model"] = model
            st.session_state["cv_scores"] = scores

            # Store feature importances for tree-based models
            if hasattr(model, "feature_importances_"):
                st.session_state["feature_importances"] = pd.Series(
                    model.feature_importances_, index=X_train.columns
                ).sort_values(ascending=False)

            logger.info("Model trained: %s, CV mean=%.4f", st.session_state["model_name"], np.mean(scores))
        except Exception as e:
            logger.error("Training failed: %s", e)
            st.error(f"Training failed: {e}")

    if st.session_state.get("trained_model") is not None:
        st.success("✅ Model Trained Successfully!")
        scores = st.session_state.get("cv_scores")
        if scores is not None and len(scores) > 0:
            metric_name = "Accuracy" if problem_type == "Classification" else "Neg MSE"
            st.write(
                f"**Cross-Validation {metric_name} (CV={st.session_state['k_folds']}):** "
                f"{np.mean(scores):.4f} ± {np.std(scores):.4f}"
            )

        # Feature importance chart
        fi = st.session_state.get("feature_importances")
        if fi is not None:
            import plotly.express as px
            with st.expander("📊 Feature Importances", expanded=True):
                fig = px.bar(
                    x=fi.values[:15], y=fi.index[:15],
                    orientation="h",
                    labels={"x": "Importance", "y": "Feature"},
                    title="Top 15 Feature Importances",
                    template="plotly_dark",
                    color=fi.values[:15],
                    color_continuous_scale="Viridis",
                )
                fig.update_layout(yaxis=dict(autorange="reversed"), margin=dict(t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)


def tab_metrics():
    st.header("8️⃣ Diagnostic Performance")
    st.caption("Evaluate sensitivity, specificity, and overall diagnostic accuracy.")
    problem_type = st.session_state["problem_type"]

    if st.session_state.get("trained_model") is None:
        st.info("Train the model in Tab 7 to view metrics.")
        return
    if st.session_state.get("split") is None:
        st.warning("Split data is missing. Please re-run Tab 5.")
        return

    from sklearn.metrics import (
        accuracy_score, mean_squared_error, classification_report,
        confusion_matrix, r2_score, mean_absolute_error,
    )
    import plotly.express as px
    import plotly.figure_factory as ff

    X_train, X_test, y_train, y_test = st.session_state["split"]
    model = st.session_state["trained_model"]

    try:
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return

    if problem_type == "Classification":
        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)

        c1, c2, c3 = st.columns(3)
        c1.metric("Train Accuracy", f"{train_acc:.4f}")
        c2.metric("Test Accuracy", f"{test_acc:.4f}")
        c3.metric("Overfit Gap", f"{(train_acc - test_acc):.4f}")

        if train_acc - test_acc > 0.15:
            st.warning("⚠️ High variance — model may be **overfitting** to training patients.")
        elif test_acc < 0.6:
            st.warning("⚠️ Low accuracy — model may be **underfitting**. Consider more risk factors or different model.")
        else:
            st.success("✅ Model shows good diagnostic generalization!")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Classification Report")
            report = classification_report(y_test, test_preds, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

        with col2:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, test_preds)
            labels = sorted(np.unique(y_test))
            # Decode labels if encoder exists
            le = st.session_state.get("target_encoder")
            if le is not None:
                try:
                    labels = le.inverse_transform(labels).tolist()
                except Exception:
                    labels = [str(l) for l in labels]
            else:
                labels = [str(l) for l in labels]

            fig = px.imshow(
                cm, text_auto=True, x=labels, y=labels,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                title="Confusion Matrix", color_continuous_scale="Blues",
                template="plotly_dark",
            )
            fig.update_layout(margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

    else:  # Regression
        train_mse = mean_squared_error(y_train, train_preds)
        test_mse = mean_squared_error(y_test, test_preds)
        train_r2 = r2_score(y_train, train_preds)
        test_r2 = r2_score(y_test, test_preds)
        test_mae = mean_absolute_error(y_test, test_preds)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Train MSE", f"{train_mse:.4f}")
        c2.metric("Test MSE", f"{test_mse:.4f}")
        c3.metric("Test R²", f"{test_r2:.4f}")
        c4.metric("Test MAE", f"{test_mae:.4f}")

        # Actual vs Predicted scatter
        st.subheader("Actual vs Predicted")
        scatter_df = pd.DataFrame({"Actual": y_test, "Predicted": test_preds})
        fig = px.scatter(
            scatter_df, x="Actual", y="Predicted",
            title="Actual vs Predicted", template="plotly_dark",
            color_discrete_sequence=["#8b5cf6"],
        )
        # Add perfect prediction line
        min_val = min(scatter_df["Actual"].min(), scatter_df["Predicted"].min())
        max_val = max(scatter_df["Actual"].max(), scatter_df["Predicted"].max())
        fig.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                      line=dict(color="#4ade80", dash="dash"))
        fig.update_layout(margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)


def tab_tuning():
    st.header("9️⃣ Hyperparameter Tuning")
    st.caption("Optimize model parameters for better diagnostic performance.")
    problem_type = st.session_state["problem_type"]

    if st.session_state.get("trained_model") is None:
        st.info("Train a base model first in Tab 7.")
        return

    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from sklearn.metrics import accuracy_score, mean_squared_error
    from sklearn.base import clone

    tune_method = st.radio("Search Method", ["GridSearchCV", "RandomizedSearchCV"], horizontal=True)

    if st.button("Start Tuning", type="primary", key="btn_tune"):
        if st.session_state.get("split") is None:
            st.error("Please split data first (Tab 5).")
            return

        X_train, X_test, y_train, y_test = st.session_state["split"]
        model_name = st.session_state["model_name"]

        param_grid = {}
        if model_name == "Random Forest":
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
            }
        elif model_name == "SVM":
            param_grid = {
                "C": [0.1, 1, 10],
                "gamma": ["scale", "auto"],
            }
        elif model_name == "Logistic Regression":
            param_grid = {
                "C": [0.01, 0.1, 1, 10],
                "solver": ["lbfgs", "liblinear"],
            }
        elif model_name == "Linear Regression":
            param_grid = {
                "fit_intercept": [True, False],
            }

        if not param_grid:
            st.info("No hyperparameter grid defined for this model.")
            return

        try:
            base_model = clone(st.session_state["model"])
            scoring = "accuracy" if problem_type == "Classification" else "neg_mean_squared_error"

            with st.spinner("🔍 Tuning in progress… This may take a minute."):
                if tune_method == "GridSearchCV":
                    search = GridSearchCV(base_model, param_grid, cv=3, scoring=scoring, n_jobs=-1)
                else:
                    search = RandomizedSearchCV(
                        base_model, param_grid, cv=3, n_iter=min(10, len(param_grid)),
                        scoring=scoring, random_state=42, n_jobs=-1,
                    )
                search.fit(X_train, y_train)

            st.session_state["tuned_model"] = search.best_estimator_
            st.session_state["best_params"] = search.best_params_
            logger.info("Tuning complete: %s", search.best_params_)
        except Exception as e:
            logger.error("Tuning failed: %s", e)
            st.error(f"Tuning failed: {e}")

    if st.session_state.get("tuned_model") is not None:
        st.success("✅ Tuning Complete!")
        st.write("**Best Parameters:**", st.session_state["best_params"])

        X_test = st.session_state["split"][1]
        y_test = st.session_state["split"][3]
        tuned_preds = st.session_state["tuned_model"].predict(X_test)

        if problem_type == "Classification":
            st.metric("Tuned Test Accuracy", f"{accuracy_score(y_test, tuned_preds):.4f}")
        else:
            st.metric("Tuned Test MSE", f"{mean_squared_error(y_test, tuned_preds):.4f}")


def tab_predict():
    st.header("🔟 Patient Screening")
    st.caption("Enter patient risk factors to predict cervical cancer risk.")
    problem_type = st.session_state["problem_type"]

    model = st.session_state.get("tuned_model") or st.session_state.get("trained_model")

    if model is None or st.session_state.get("split") is None:
        st.info("🚨 Please train a model in Tab 7 to unlock Patient Screening!")
        return

    train_columns = st.session_state.get("train_columns")
    if train_columns is None:
        st.error("Training column metadata missing. Re-run Tab 5.")
        return

    st.markdown("### 🩺 Enter patient risk factors below")

    original_data = st.session_state["cleaned_data"]
    features = st.session_state["selected_features"]

    # Validate that features still exist in original_data
    missing_feats = [f for f in features if f not in original_data.columns]
    if missing_feats:
        st.error(f"Features {missing_feats} not found in cleaned data. Re-run Tab 4 (Feature Selection).")
        return

    user_inputs = {}
    cols = st.columns(3)

    for i, feature in enumerate(features):
        col = cols[i % 3]
        with col:
            unique_vals = original_data[feature].dropna().unique().tolist()

            # Determine if feature is numeric or categorical
            is_numeric = pd.api.types.is_numeric_dtype(original_data[feature])

            # Few unique values → selectbox (categorical-like)
            if len(unique_vals) <= 10:
                # Safe sort: handle mixed types by converting to string for sort
                try:
                    unique_vals = sorted(unique_vals)
                except TypeError:
                    unique_vals = sorted(unique_vals, key=str)
                user_inputs[feature] = st.selectbox(
                    f"Select {feature}", unique_vals, key=f"pred_{feature}"
                )

            # Numeric with many unique values → slider
            elif is_numeric:
                min_val = float(original_data[feature].min())
                max_val = float(original_data[feature].max())
                mean_val = float(original_data[feature].mean())

                if min_val == max_val:
                    st.text_input(
                        f"{feature} (constant)", value=str(min_val),
                        disabled=True, key=f"pred_{feature}"
                    )
                    user_inputs[feature] = min_val
                else:
                    # Clamp mean to valid range to avoid Streamlit errors
                    mean_val = max(min_val, min(max_val, mean_val))
                    step = (max_val - min_val) / 100.0

                    user_inputs[feature] = st.slider(
                        f"Adjust {feature}",
                        min_value=min_val, max_value=max_val,
                        value=mean_val, step=step,
                        key=f"pred_{feature}"
                    )

            # High-cardinality categorical → selectbox
            else:
                try:
                    unique_vals = sorted(unique_vals, key=str)
                except Exception:
                    pass
                user_inputs[feature] = st.selectbox(
                    f"Select {feature}", unique_vals, key=f"pred_{feature}"
                )

    st.divider()

    if st.button("🩺 Screen Patient", type="primary", use_container_width=True, key="btn_predict"):
        try:
            # Build input DataFrame from user selections
            input_df = pd.DataFrame([user_inputs])

            # Apply dummy encoding — use drop_first=True to match training
            input_encoded = pd.get_dummies(input_df, drop_first=True)

            # Align to the exact training columns; missing dummies → 0
            input_aligned = input_encoded.reindex(columns=train_columns, fill_value=0)

            # Ensure all columns are numeric (dummies should be 0/1)
            input_aligned = input_aligned.astype(float)

            # Apply scaler if used during training
            scaler = st.session_state.get("scaler")
            if scaler is not None:
                input_aligned = pd.DataFrame(
                    scaler.transform(input_aligned),
                    columns=input_aligned.columns
                )

            prediction = model.predict(input_aligned)[0]

            # ---- Store results in session_state so they persist across reruns ----
            result = {"prediction": prediction, "input_aligned": input_aligned}

            if problem_type == "Classification":
                target_encoder = st.session_state.get("target_encoder")
                if target_encoder is not None:
                    result["predicted_class"] = target_encoder.inverse_transform([int(prediction)])[0]
                else:
                    result["predicted_class"] = (
                        int(prediction) if isinstance(prediction, (float, np.floating))
                        else prediction
                    )

                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(input_aligned)[0]
                    result["proba"] = proba
                    result["confidence"] = float(max(proba)) * 100

                    if target_encoder is not None:
                        result["class_labels"] = target_encoder.classes_.tolist()
                    else:
                        result["class_labels"] = [str(c) for c in range(len(proba))]

            st.session_state["last_prediction"] = result
            logger.info("Screening prediction: %s", prediction)


        except Exception as e:
            logger.error("Prediction failed: %s", e)
            st.error(f"Prediction failed: {e}")
            st.session_state["last_prediction"] = None

    # ---- Display persisted prediction results (survives reruns) ----
    result = st.session_state.get("last_prediction")
    if result is not None:
        st.markdown("<br>", unsafe_allow_html=True)

        if problem_type == "Classification":
            predicted_class = result.get("predicted_class", result["prediction"])
            # Map 0/1 to clinical labels
            if predicted_class in (0, "0"):
                st.success("### ✅ Screening Result: **Low Risk (Negative)**")
            elif predicted_class in (1, "1"):
                st.error("### ⚠️ Screening Result: **High Risk (Positive)**")
                st.warning("This patient should be referred for further clinical evaluation. This is a screening tool, not a diagnosis.")
            else:
                st.info(f"### 🎯 Screening Result: **{predicted_class}**")

            if "confidence" in result:
                conf = result['confidence']
                if conf < 60:
                    st.warning("⚠️ Low confidence — consider additional diagnostic tests.")
                st.info(f"🧠 **Confidence:** {result['confidence']:.1f}%")

            if "proba" in result and "class_labels" in result:
                import plotly.express as px
                prob_df = pd.DataFrame({
                    "Class": result["class_labels"],
                    "Probability": result["proba"]
                })
                fig = px.bar(
                    prob_df, x="Class", y="Probability",
                    title="Class Probabilities", template="plotly_dark",
                    color="Probability", color_continuous_scale="Viridis",
                    text_auto=".2%",
                )
                fig.update_layout(margin=dict(t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)
        else:
            prediction = result["prediction"]
            st.success(f"### 📈 Predicted Value: **{prediction:,.2f}**")


def tab_export():
    st.header("📦 Export & Download")
    st.caption("Download the trained screening model and prediction results.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Export Trained Model")
        model = st.session_state.get("tuned_model") or st.session_state.get("trained_model")
        if model is not None:
            import pickle
            buf = io.BytesIO()
            pickle.dump(model, buf)
            buf.seek(0)
            st.download_button(
                "⬇️ Download Model (.pkl)",
                data=buf, file_name="trained_model.pkl",
                mime="application/octet-stream",
            )
        else:
            st.info("Train a model first to export it.")

    with col2:
        st.subheader("Export Predictions")
        if st.session_state.get("split") is not None and model is not None:
            X_test = st.session_state["split"][1]
            y_test = st.session_state["split"][3]
            preds = model.predict(X_test)
            result_df = X_test.copy()
            result_df["Actual"] = y_test.values if hasattr(y_test, "values") else y_test
            result_df["Predicted"] = preds
            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Predictions (.csv)",
                data=csv, file_name="predictions.csv",
                mime="text/csv",
            )
        else:
            st.info("Train and evaluate a model first.")


# ---------------------------------------------------------------------------
# 4. MAIN APPLICATION LAYOUT
# ---------------------------------------------------------------------------
st.title("🧬 Cervical Cancer Detection System")
st.caption("ML-powered screening pipeline for cervical cancer risk assessment using patient risk factors.")

# Sidebar — locked to Classification for cancer detection
st.sidebar.header("🩺 About")
st.sidebar.markdown(
    "This tool uses machine learning to predict cervical cancer risk "
    "based on patient demographics, habits, and medical history.\n"
)
problem_type = "Classification"
st.session_state["problem_type"] = problem_type

st.sidebar.divider()
st.sidebar.header("📋 Pipeline Progress")
sidebar_status()

# Tabs
tabs = st.tabs([
    "1. Patient Data", "2. EDA", "3. Cleaning", "4. Risk Factors",
    "5. Split", "6. Model", "7. Train", "8. Diagnostics",
    "9. Tuning", "10. Screen", "📦 Export"
])

with tabs[0]:  tab_data_input()
with tabs[1]:  tab_eda()
with tabs[2]:  tab_cleaning()
with tabs[3]:  tab_feature_selection()
with tabs[4]:  tab_data_split()
with tabs[5]:  tab_model_setup()
with tabs[6]:  tab_train_evaluate()
with tabs[7]:  tab_metrics()
with tabs[8]:  tab_tuning()
with tabs[9]:  tab_predict()
with tabs[10]: tab_export()
