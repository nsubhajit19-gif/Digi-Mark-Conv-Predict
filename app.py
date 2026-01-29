import streamlit as st
import joblib
import json
import pandas as pd
import requests
from src.config import BEST_MODEL_PATH, FEATURES_PATH
from streamlit_lottie import st_lottie

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Digital Marketing Conversion Prediction",
    page_icon="🤖",
    layout="centered"
)

# -------------------------------
# Custom CSS (Purple + Neon Gradient)
# -------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        color: #eaeaea;
        font-family: 'Segoe UI', sans-serif;
    }

    h1 {
        background: linear-gradient(90deg, #ff00ff, #00ffff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 30px;
        font-weight: bold;
    }

    label {
        color: #eaeaea !important;
        font-weight: 500;
    }

    .stSelectbox, .stNumberInput, .stSlider {
        background-color: #222831 !important;
        border-radius: 8px;
        color: #eaeaea;
    }

    div.stButton > button {
        background: linear-gradient(90deg, #ff00ff, #00ffff);
        color: #000000;
        font-weight: bold;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        border: none;
        margin-top: 20px;
        box-shadow: 0px 0px 15px rgba(255, 0, 255, 0.5);
    }

    div.stButton > button:hover {
        background: linear-gradient(90deg, #00ffff, #ff00ff);
        color: #000000;
    }

    .stSuccess {
        background-color: #4caf50;
        color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0px 0px 10px rgba(0, 255, 0, 0.5);
    }

    .stError {
        background-color: #e63946;
        color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0px 0px 10px rgba(255, 0, 0, 0.5);
    }

    .card {
        background-color: #222831;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0px 0px 20px rgba(255, 0, 255, 0.25);
        margin-bottom: 25px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Load model and features (SAFE)
# -------------------------------
@st.cache_resource
def load_artifacts():
    if not BEST_MODEL_PATH.exists():
        st.error("❌ Trained model not found. Please train the model first.")
        st.stop()

    if not FEATURES_PATH.exists():
        st.error("❌ feature_list.json not found. Please run training.")
        st.stop()

    model = joblib.load(BEST_MODEL_PATH)
    with open(FEATURES_PATH, "r") as f:
        feature_cols = json.load(f)

    return model, feature_cols


model, feature_cols = load_artifacts()

# -------------------------------
# Helper: Load Lottie Animation
# -------------------------------
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Fireworks animation from LottieFiles
lottie_fireworks = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json")

# -------------------------------
# App Title
# -------------------------------
st.title("🤖 Digital Marketing Conversion Prediction")
st.markdown("<div class='card'>", unsafe_allow_html=True)

# -------------------------------
# User Inputs
# -------------------------------
customer_id = st.number_input("Customer ID", min_value=8000, value=8000)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
income = st.number_input("Income", min_value=0, value=50000)

campaign_channel = st.selectbox(
    "Campaign Channel", ["Social Media", "Email", "PPC", "SEO", "Referral"]
)
campaign_type = st.selectbox(
    "Campaign Type", ["Awareness", "Retention", "Conversion", "Consideration"]
)

ad_spend = st.number_input("Ad Spend", min_value=0.0, value=1000.0)
ctr = st.slider("Click Through Rate", 0.0, 1.0, 0.05)
conversion_rate = st.slider("Conversion Rate", 0.0, 1.0, 0.10)

website_visits = st.number_input("Website Visits", min_value=0, value=10)
pages_per_visit = st.number_input("Pages Per Visit", min_value=0.0, value=3.0)
time_on_site = st.number_input("Time on Site (minutes)", min_value=0.0, value=5.0)

social_shares = st.number_input("Social Shares", min_value=0, value=5)
email_opens = st.number_input("Email Opens", min_value=0, value=2)
email_clicks = st.number_input("Email Clicks", min_value=0, value=1)

previous_purchases = st.number_input("Previous Purchases", min_value=0, value=2)
loyalty_points = st.number_input("Loyalty Points", min_value=0, value=500)

# advertising_platform = st.selectbox("Advertising Platform", ["IsConfid"])
# advertising_tool = st.selectbox("Advertising Tool", ["ToolConfid"])

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# Prepare input DataFrame
# -------------------------------
new_data = pd.DataFrame(
    [{
        "CustomerID": customer_id,
        "Age": age,
        "Gender": gender,
        "Income": income,
        "CampaignChannel": campaign_channel,
        "CampaignType": campaign_type,
        "AdSpend": ad_spend,
        "ClickThroughRate": ctr,
        "ConversionRate": conversion_rate,
        "WebsiteVisits": website_visits,
        "PagesPerVisit": pages_per_visit,
        "TimeOnSite": time_on_site,
        "SocialShares": social_shares,
        "EmailOpens": email_opens,
        "EmailClicks": email_clicks,
        "PreviousPurchases": previous_purchases,
        "LoyaltyPoints": loyalty_points,
        # "AdvertisingPlatform": advertising_platform,
        # "AdvertisingTool": advertising_tool
    }],
    columns=feature_cols
)

# -------------------------------
# Prediction (with Animation)
# -------------------------------
if st.button("🚀 Predict Conversion"):

    if not hasattr(model, "predict_proba"):
        st.error("❌ This model does not support probability prediction.")
        st.stop()

    # Probability of conversion
    conversion_proba = model.predict_proba(new_data)[0][1]

    # -----------------------------
    # Confidence-based Interpretation
    # -----------------------------
    if conversion_proba >= 0.40:
        st.success("✅ Customer is **LIKELY to convert**")
        st_lottie(lottie_fireworks, height=250, key="fireworks")  # 🎆 Animation here
    elif conversion_proba >= 0.30:
        st.warning("⚠️ Customer **MAY convert** (Needs nurturing)")
    else:
        st.error("❌ Customer is **UNLIKELY to convert**")

    # -----------------------------
    # Confidence Visualization
    # -----------------------------
    st.markdown("### 📊 Conversion Confidence")
    st.progress(int(conversion_proba * 100))
    st.write(f"Confidence Score: **{conversion_proba * 100:.2f}%**")

# -------------------------------
# Footer
# -------------------------------
st.markdown(
    "<hr><center style='color:gray;'>Smart Predictions. Digital Impact. 🌌</center>",
    unsafe_allow_html=True
)
