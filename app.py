import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Product review Sentiment Classifier", page_icon="🛍️")


@st.cache_resource
def load_classifier():
    return pipeline("sentiment-analysis")


classifier = load_classifier()

st.title("🛍️ Product Review Classifier")
st.write(
    " Enter a product review to identify whether this sentiment is Positive or Negative"
)

review = st.text_area(
    "Product review",
    placeholder="Example: The headphone have excelent sound and battery life",
    height=150,
)

if st.button("Analyze sentiment"):
    if not review.strip():
        st.warning("Please enter a review")
    else:
        with st.spinner("Analyzing review..."):
            result = classifier(review, truncation=True)[0]

    label = result["label"]
    confidence = result["score"] * 100

    if label == "POSITIVE":
        st.success("Sentiment: Positive")
    else:
        st.error("Sentiment: Negative")

    st.write(f"Confidence : **{confidence:.2f}%**")
