
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "models" / "accident_severity_model.joblib"
METADATA_FILE = BASE_DIR / "models" / "metadata.json"
METRICS_FILE = BASE_DIR / "outputs" / "evaluation_metrics.json"

st.set_page_config(
    page_title="Road Accident Severity Prediction",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 AI-Based Road Accident Severity Prediction")

st.write(
    "Enter road, weather, time, and other available details to estimate "
    "the severity category using a machine learning model trained on "
    "historical accident records."
)

st.warning(
    "This application predicts accident severity for records similar to "
    "the training data. It does not predict whether an accident will occur. "
    "Do not use this prediction as an emergency or road-safety decision."
)

# Check that the trained model and metadata exist.
if not MODEL_FILE.exists() or not METADATA_FILE.exists():
    st.error("The trained model or metadata file was not found.")
    st.info(
        "Run `python train_model.py` in your project folder first."
    )
    st.stop()

# Load model and metadata.
try:
    model = joblib.load(MODEL_FILE)

    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        metadata = json.load(file)

except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()

feature_columns = metadata.get("feature_columns", [])
numeric_features = set(metadata.get("numeric_columns", []))
categorical_features = set(metadata.get("categorical_columns", []))
class_labels = metadata.get("target_classes", [])

if not feature_columns:
    st.error("No feature columns were found in metadata.json.")
    st.info("Run `python train_model.py` again.")
    st.stop()

if not class_labels:
    st.error("No target class labels were found in metadata.json.")
    st.info("Run `python train_model.py` again.")
    st.stop()

st.caption(
    f"Model selected during training: "
    f"{metadata.get('best_model', 'Trained classifier')}"
)

st.subheader("Enter accident-related details")
st.write("Complete the fields below and click the prediction button.")

input_values = {}

with st.form("prediction_form"):
    left, right = st.columns(2)

    for index, feature in enumerate(feature_columns):
        container = left if index % 2 == 0 else right

        with container:
            if feature in numeric_features:
                input_values[feature] = st.number_input(
                    feature.replace("_", " "),
                    value=0.0,
                    format="%.3f",
                    help=(
                        "Enter a numeric value using the same units "
                        "and coding as the training dataset."
                    )
                )

            else:
                # Use actual values from the training dataset if available.
                options = metadata.get("categorical_options", {}).get(
                    feature, []
                )

                if options and len(options) <= 50:
                    input_values[feature] = st.selectbox(
                        feature.replace("_", " "),
                        options=options
                    )
                else:
                    input_values[feature] = st.text_input(
                        feature.replace("_", " "),
                        value="",
                        help=(
                            "Enter the value in the same format as "
                            "the training dataset."
                        )
                    )

    submitted = st.form_submit_button(
        "Predict Accident Severity",
        type="primary",
        use_container_width=True
    )

if submitted:
    input_df = pd.DataFrame(
        [[input_values[col] for col in feature_columns]],
        columns=feature_columns
    )

    # Convert numerical inputs to numbers.
    for col in numeric_features:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(
                input_df[col], errors="coerce"
            )

    # Convert empty categorical fields to missing values.
    for col in categorical_features:
        if col in input_df.columns:
            input_df[col] = input_df[col].replace("", pd.NA)

    try:
        prediction = int(model.predict(input_df)[0])

        # The model target was encoded as 0, 1, 2, 3 during training.
        if 0 <= prediction < len(class_labels):
            predicted_label = class_labels[prediction]
        else:
            predicted_label = str(prediction)

        st.subheader("Prediction Result")
        st.success(f"Predicted severity class: {predicted_label}")

        st.info(
            "This is a predicted severity category, not a prediction "
            "of whether an accident will happen. Check the dataset "
            "documentation to interpret the severity code."
        )

        # Show model scores, if supported.
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            model_classes = model.named_steps["classifier"].classes_

            score_rows = []

            for class_id, probability in zip(
                model_classes, probabilities
            ):
                class_id = int(class_id)

                if 0 <= class_id < len(class_labels):
                    label = class_labels[class_id]
                else:
                    label = str(class_id)

                score_rows.append({
                    "Severity class": label,
                    "Model score (%)": round(
                        float(probability) * 100, 2
                    )
                })

            score_df = pd.DataFrame(score_rows)
            score_df = score_df.sort_values(
                "Model score (%)", ascending=False
            )

            st.subheader("Model class scores")
            st.dataframe(
                score_df.reset_index(drop=True),
                use_container_width=True
            )

            st.caption(
                "These scores describe the model's class estimates. "
                "They are not the probability of an accident occurring."
            )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.write(
            "Check the input values. If the problem continues, "
            "send me the complete error message."
        )

# Display evaluation results from the saved training metrics.
if METRICS_FILE.exists():
    with st.expander("View model evaluation results"):
        try:
            with open(METRICS_FILE, "r", encoding="utf-8") as file:
                metrics = json.load(file)

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Accuracy",
                f"{metrics.get('accuracy', 0):.1%}"
            )
            col2.metric(
                "Macro Precision",
                f"{metrics.get('precision_macro', 0):.1%}"
            )
            col3.metric(
                "Macro Recall",
                f"{metrics.get('recall_macro', 0):.1%}"
            )
            col4.metric(
                "Macro F1-score",
                f"{metrics.get('f1_macro', 0):.1%}"
            )

        except Exception as error:
            st.info(f"Evaluation metrics are unavailable: {error}")

# Display the saved confusion matrix if available.
confusion_matrix_file = OUTPUT_DIR = BASE_DIR / "outputs" / "confusion_matrix.png"

if confusion_matrix_file.exists():
    with st.expander("View confusion matrix"):
        st.image(
            str(confusion_matrix_file),
            caption="Confusion matrix for the test dataset",
            use_container_width=True
        )
