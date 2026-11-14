"""Small Streamlit interface for the exported Amazon review sentiment model."""

from pathlib import Path

import joblib
import streamlit as st


ARTIFACT_PATH = Path(__file__).with_name("sentiment_pipeline.joblib")


@st.cache_resource
def load_model():
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {ARTIFACT_PATH}. Run the export cell in "
            "amazon_review_sentiment.ipynb first."
        )
    artifact = joblib.load(ARTIFACT_PATH)
    return artifact["pipeline"], artifact.get("model_name", "Sentiment model")


st.set_page_config(page_title="Amazon Review Sentiment", page_icon="🛍️")
st.title("Amazon Review Sentiment")
st.write(
    "Enter a product review to estimate whether its sentiment is positive or negative. "
    "This demo uses the model exported from the notebook."
)

try:
    pipeline, model_name = load_model()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

st.caption(f"Loaded model: {model_name}")
review = st.text_area(
    "Review text",
    placeholder="The product arrived quickly and works exactly as expected.",
    height=150,
)

if st.button("Analyze sentiment", type="primary", disabled=not review.strip()):
    prediction = int(pipeline.predict([review])[0])
    sentiment = "Positive" if prediction == 1 else "Negative"
    probability = float(pipeline.predict_proba([review])[0, 1])
    confidence = probability if prediction == 1 else 1 - probability

    st.subheader(sentiment)
    st.metric("Model confidence", f"{confidence:.1%}")
    st.progress(probability, text=f"Estimated positive probability: {probability:.1%}")
    st.caption("Model confidence is an estimate, not a guarantee. Neutral sentiment is not modeled.")
