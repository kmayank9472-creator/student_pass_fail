import io
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Student Pass/Fail Predictor", page_icon="🎓", layout="wide")
st.title("🎓 Student Pass/Fail Predictor")
st.write("Train a logistic regression model from student study, attendance, and midterm data.")

FEATURES = ["study_hours_per_week", "attendance_pct", "midterm_score"]
TARGET = "pass"
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BASE_DIR / "Pass-Fail Data.csv"
NOTEBOOK = BASE_DIR / "Student_pass_fail_Classification.ipynb"

@st.cache_data
def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw))

uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
try:
    if uploaded is not None:
        df = read_csv_bytes(uploaded.getvalue())
        data_source = uploaded.name
    elif DEFAULT_CSV.exists():
        df = pd.read_csv(DEFAULT_CSV)
        data_source = DEFAULT_CSV.name
    else:
        st.error("CSV file missing. Upload a CSV with the required columns.")
        st.stop()
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

missing_columns = [col for col in FEATURES + [TARGET] if col not in df.columns]
if missing_columns:
    st.error("CSV is missing required columns: " + ", ".join(missing_columns))
    st.caption("Required: study_hours_per_week, attendance_pct, midterm_score, pass")
    st.stop()

try:
    df = df[FEATURES + [TARGET]].apply(pd.to_numeric, errors="raise").dropna()
except (ValueError, TypeError):
    st.error("Required columns must contain numeric values, with pass coded as 0 or 1.")
    st.stop()

if not set(df[TARGET].unique()).issubset({0, 1}):
    st.error("The 'pass' column must use 0 for fail and 1 for pass.")
    st.stop()
if df[TARGET].nunique() != 2 or df[TARGET].value_counts().min() < 2 or len(df) < 8:
    st.error("Need at least 8 usable rows and at least 2 examples of each class to train and evaluate.")
    st.stop()

st.caption(f"Using **{data_source}** · {len(df)} usable records")
with st.expander("Preview data and basic checks", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)
    a, b, c = st.columns(3)
    a.metric("Rows", len(df))
    b.metric("Missing values", int(df.isna().sum().sum()))
    c.metric("Duplicate rows", int(df.duplicated().sum()))
    st.write("Class counts (0 = fail, 1 = pass)")
    st.bar_chart(df[TARGET].value_counts().sort_index())

X = df[FEATURES]
y = df[TARGET].astype(int)
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression())
    ])
    model.fit(X_train, y_train)
except ValueError as exc:
    st.error(f"Could not train the model: {exc}")
    st.stop()

pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)
st.subheader("Model evaluation")
st.metric("Test accuracy", f"{accuracy:.1%}")
cm = confusion_matrix(y_test, pred, labels=[0, 1])
st.write("Confusion matrix (rows = actual, columns = predicted; order: fail, pass)")
st.dataframe(pd.DataFrame(cm, index=["Actual fail", "Actual pass"], columns=["Predicted fail", "Predicted pass"]))
report = classification_report(y_test, pred, labels=[0, 1], target_names=["Fail", "Pass"], output_dict=True, zero_division=0)
st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

st.subheader("Predict for a student")
col1, col2, col3 = st.columns(3)
with col1:
    hours = st.number_input("Study hours per week", min_value=0.0, max_value=168.0, value=6.0, step=0.5)
with col2:
    attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=78.0, step=1.0)
with col3:
    score = st.number_input("Midterm score", min_value=0.0, max_value=100.0, value=64.0, step=1.0)

student = pd.DataFrame([[hours, attendance, score]], columns=FEATURES)
prediction = int(model.predict(student)[0])
probabilities = model.predict_proba(student)[0]
pass_probability = float(probabilities[list(model.classes_).index(1)])
if prediction == 1:
    st.success(f"Predicted result: PASS · pass probability {pass_probability:.1%}")
else:
    st.warning(f"Predicted result: FAIL · pass probability {pass_probability:.1%}")

st.sidebar.download_button(
    "Download CSV data", data=df.to_csv(index=False).encode("utf-8"),
    file_name="Pass-Fail Data.csv", mime="text/csv"
)
if NOTEBOOK.exists():
    st.sidebar.download_button(
        "Download original notebook", data=NOTEBOOK.read_bytes(),
        file_name=NOTEBOOK.name, mime="application/x-ipynb+json"
    )
