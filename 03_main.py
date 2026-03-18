import json
import joblib
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from setfit import SetFitModel

# -----------------------------
# 1. Page config
# -----------------------------
st.set_page_config(
    page_title="Goods or Services Classifier",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Goods or Services Classifier")
st.write(
    """
This demo uses a multilingual SetFit model for **Goods vs Services**
and a separate SBERT + Logistic Regression classifier for **macro categories**.
Write the **name** and a **short description** in English, Spanish, French or German.
"""
)

# -----------------------------
# 2. Load models and metadata
# -----------------------------

@st.cache_resource
def load_models():
    # --- Load SetFit model for goods vs services ---
    gs_setfit_model = SetFitModel.from_pretrained("setfit-goods-services-multilingual")

    # --- Category metadata ---
    with open("sbert_category_meta.json", "r") as f:
        meta_cat = json.load(f)

    sbert_model_name_cat = meta_cat["sbert_model_name"]
    cat_classes = meta_cat["classes"]

    # --- Load SBERT encoder for categories ---
    sbert_cat = SentenceTransformer(sbert_model_name_cat)

    # --- Load category classifier ---
    cat_clf = joblib.load("sbert_category_classifier.pkl")

    return gs_setfit_model, sbert_cat, cat_clf, cat_classes


gs_setfit_model, sbert_cat, cat_clf, cat_classes = load_models()

# Map internal labels to display labels
DISPLAY_LABELS = {
    "goods": "Goods",
    "service": "Service"
}

# -----------------------------
# 3. Helper functions
# -----------------------------

def predict_goods_services(text: str):
    """
    Predict goods vs services using REAL SetFit model.
    Returns: (pred_label, pred_prob, prob_goods, prob_service)
    """
    # Predict probabilities
    probs = gs_setfit_model.predict_proba([text])[0]  # shape (2,)
    labels_order = list(getattr(gs_setfit_model, "labels", ["goods", "service"]))

    prob_dict = {lab: float(p) for lab, p in zip(labels_order, probs)}

    # Predict label
    pred_label = gs_setfit_model.predict([text])[0]
    pred_label = str(pred_label).lower().strip()

    # Extract probs safely
    prob_goods = prob_dict.get("goods", 0.0)
    prob_service = prob_dict.get("service", 0.0)
    pred_prob = prob_dict.get(pred_label, max(prob_goods, prob_service))

    return pred_label, pred_prob, prob_goods, prob_service


def predict_top3_categories(text: str):
    """Return top 3 categories with probabilities using the category classifier."""
    emb = sbert_cat.encode([text])
    probs = cat_clf.predict_proba(emb)[0].astype(float)
    top3_idx = np.argsort(probs)[::-1][:3]
    return [(cat_classes[i], probs[i]) for i in top3_idx]

# -----------------------------
# 4. Input form
# -----------------------------

st.subheader("🔎 Input")

name = st.text_input("Enter the name of the good or service")
description = st.text_area(
    "Enter a description of your good or service",
    height=150
)

if st.button("Classify"):
    if not name.strip() and not description.strip():
        st.warning("Please enter at least a name or a description.")
    else:
        if description.strip():
            text = f"{name}. {description}".strip()
        else:
            text = name.strip()

        with st.spinner("Running the models..."):
            # 1) Goods vs services (SetFit real)
            pred_label, pred_prob, prob_goods, prob_service = predict_goods_services(text)
            display_label = DISPLAY_LABELS.get(pred_label, pred_label)

            # 2) Top 3 categories (SBERT + LogReg)
            top3_cats = predict_top3_categories(text)

        # -----------------------------
        # 5. Output
        # -----------------------------
        st.subheader("📊 Result")

        st.markdown(
            f"**Prediction (type):** {display_label} "
            f"(confidence: {pred_prob * 100:.2f} %)"
        )

        st.write("Estimated probabilities for type:")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Goods", f"{prob_goods * 100:.2f} %")
        with col2:
            st.metric("Service", f"{prob_service * 100:.2f} %")

        st.subheader("📂 Top 3 suggested categories")
        for rank, (cat, prob) in enumerate(top3_cats, start=1):
            st.write(f"{rank}. **{cat}** ({prob * 100:.2f} %)")

        st.info(
            "Goods vs Services is predicted using a SetFit model (contrastive fine-tuning). "
            "Macro categories are predicted using SBERT embeddings and a logistic regression classifier."
        )

st.markdown("---")
st.caption(
    "Prototype for academic purposes. The classifier may be less accurate for very unusual or extremely short descriptions."
)
