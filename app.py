import streamlit as st
from transformers import pipeline

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

st.set_page_config(
    page_title="product-review-sentiment-classifier", page_icon="🛍️", layout="centered"
)


@st.cache_resource
def load_classifier():
    return pipeline("sentiment-analysis", model=MODEL_NAME, tokenizer=MODEL_NAME)


def set_review(example_text):
    st.session_state.review = example_text


if "review" not in st.session_state:
    st.session_state.review = ""

st.title("🛍️ Product Review Sentiment Classifier")
st.write("Analzyse product review and identity sentiment Positive or Negative")
st.subheader("Try Example")

col1, col2, col3 = st.columns(3)

with col1:
    st.button(
        "Positive",
        on_click=set_review,
        args=("The headphone have excelent sound and battery life",),
    )

with col2:
    st.button(
        "Negative",
        on_click=set_review,
        args=("The product stopped working in 2days.Very disappointing",),
    )
with col3:
    st.button(
        "delivery Delay",
        on_click=set_review,
        args=("The item arrived late and Item was damaged",),
    )

review = st.text_area(
    "product Review",
    placeholder="Write or select comments fro review",
    height=150,
    key="review",
)

left, right = st.columns(2)

with left:
    analyze = st.button("sentiment Analyze", type="primary")


with right:
    st.button("clear", on_click=set_review, args=("",))

if analyze:
    if not review.strip():
        st.warning("Please enter product review")
    else:
        with st.spinner("Analyzing review"):
            classifier = load_classifier()
            result = classifier(review, truncation=True)[0]

        label = result["label"]
        confidence = result["score"] * 100

        if label == "POSITIVE":
            st.success("Sentiment: Positive")
        else:
            st.error("Sentiment: Negative")

        with st.expander("Technical details"):
            st.write(f"Model:`{MODEL_NAME}`")
            st.write(f"Raw label: `{label}`")
