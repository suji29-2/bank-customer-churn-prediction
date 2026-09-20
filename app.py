import streamlit as st
import pandas as pd
import pickle

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Bank Churn AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 25px;
    background: linear-gradient(135deg, #1f4e79, #2878b5);
}

.hero h1 {
    color: white;
    font-size: 42px;
    margin-bottom: 8px;
}

.hero p {
    color: #e8f4ff;
    font-size: 18px;
}

.card {
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #d9e2ec;
    background-color: #ffffff;
    margin-bottom: 15px;
}

.result {
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    margin-top: 20px;
}

.result h2 {
    margin-bottom: 10px;
}

.small-text {
    font-size: 14px;
    color: #6b7280;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# LOAD MODEL AND PREPROCESSING FILES
# ==========================================================

with open("random_forest_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("country_encoder.pkl", "rb") as file:
    country_encoder = pickle.load(file)

with open("gender_encoder.pkl", "rb") as file:
    gender_encoder = pickle.load(file)


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.title("🏦 Bank Churn AI")

    st.markdown("---")

    st.subheader("🤖 Model Information")

    st.write("**Algorithm:** Random Forest")
    st.write("**Problem Type:** Binary Classification")
    st.write("**Target:** Customer Churn")

    st.markdown("---")

    st.subheader("🎯 Prediction Classes")

    st.success("0 → Customer Stayed")
    st.error("1 → Customer Churned")

    st.markdown("---")

    st.info(
        "This application predicts the likelihood "
        "of a bank customer leaving the bank based "
        "on customer information."
    )


# ==========================================================
# HERO SECTION
# ==========================================================

st.markdown("""
<div class="hero">

<h1>🏦 Bank Customer Churn Prediction</h1>

<p>
AI-powered system for predicting customer churn
using Machine Learning
</p>

</div>
""", unsafe_allow_html=True)


# ==========================================================
# TOP INFORMATION CARDS
# ==========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🤖 Model",
        "Random Forest"
    )

with col2:
    st.metric(
        "🎯 Problem",
        "Classification"
    )

with col3:
    st.metric(
        "📌 Target",
        "Churn"
    )

with col4:
    st.metric(
        "📊 Classes",
        "2"
    )


st.markdown("---")


# ==========================================================
# CUSTOMER DETAILS
# ==========================================================

st.header("👤 Customer Profile")

st.write(
    "Enter the customer's information to generate a churn prediction."
)

left, right = st.columns(2)


# ----------------------------------------------------------
# LEFT COLUMN
# ----------------------------------------------------------

with left:

    st.subheader("📋 Personal & Banking Details")

    credit_score = st.slider(
        "💳 Credit Score",
        min_value=300,
        max_value=850,
        value=650
    )

    country = st.selectbox(
        "🌍 Country",
        country_encoder.classes_
    )

    gender = st.selectbox(
        "👤 Gender",
        gender_encoder.classes_
    )

    age = st.slider(
        "🎂 Age",
        min_value=18,
        max_value=100,
        value=35
    )

    tenure = st.slider(
        "📅 Tenure (Years)",
        min_value=0,
        max_value=10,
        value=5
    )


# ----------------------------------------------------------
# RIGHT COLUMN
# ----------------------------------------------------------

with right:

    st.subheader("💰 Financial & Account Details")

    balance = st.number_input(
        "💰 Account Balance",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

    products_number = st.slider(
        "🏦 Number of Products",
        min_value=1,
        max_value=4,
        value=1
    )

    credit_card = st.selectbox(
        "💳 Has Credit Card?",
        ["Yes", "No"]
    )

    active_member = st.selectbox(
        "🟢 Active Member?",
        ["Yes", "No"]
    )

    estimated_salary = st.number_input(
        "💵 Estimated Salary",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )


st.markdown("---")


# ==========================================================
# PREDICTION BUTTON
# ==========================================================

predict = st.button(
    "🔮  PREDICT CUSTOMER CHURN",
    use_container_width=True
)


# ==========================================================
# PREDICTION PROCESS
# ==========================================================

if predict:

    # Encode categorical variables

    country_encoded = country_encoder.transform(
        [country]
    )[0]

    gender_encoded = gender_encoder.transform(
        [gender]
    )[0]

    credit_card_encoded = (
        1 if credit_card == "Yes" else 0
    )

    active_member_encoded = (
        1 if active_member == "Yes" else 0
    )


    # Create input dataframe

    input_data = pd.DataFrame({

        "credit_score": [credit_score],

        "country": [country_encoded],

        "gender": [gender_encoded],

        "age": [age],

        "tenure": [tenure],

        "balance": [balance],

        "products_number": [products_number],

        "credit_card": [credit_card_encoded],

        "active_member": [active_member_encoded],

        "estimated_salary": [estimated_salary]

    })


    # Scale the input

    input_scaled = scaler.transform(
        input_data
    )


    # Make prediction

    prediction = model.predict(
        input_scaled
    )[0]


    # Get probabilities

    probabilities = model.predict_proba(
        input_scaled
    )[0]

    stay_probability = probabilities[0] * 100
    churn_probability = probabilities[1] * 100


    # ======================================================
    # RESULT SECTION
    # ======================================================

    st.markdown("---")

    st.header("📊 Prediction Dashboard")


    # ------------------------------------------------------
    # Result
    # ------------------------------------------------------

    if prediction == 1:

        st.error(
            "⚠️ HIGH CHURN RISK — Customer is likely to CHURN"
        )

        st.metric(
            "Churn Probability",
            f"{churn_probability:.2f}%"
        )

    else:

        st.success(
            "✅ LOW CHURN RISK — Customer is likely to STAY"
        )

        st.metric(
            "Churn Probability",
            f"{churn_probability:.2f}%"
        )


    # ------------------------------------------------------
    # Probability Comparison
    # ------------------------------------------------------

    st.subheader("📈 Prediction Probability")

    probability_col1, probability_col2 = st.columns(2)

    with probability_col1:

        st.write("🟢 Stay Probability")

        st.progress(
            int(stay_probability)
        )

        st.write(
            f"**{stay_probability:.2f}%**"
        )

    with probability_col2:

        st.write("🔴 Churn Probability")

        st.progress(
            int(churn_probability)
        )

        st.write(
            f"**{churn_probability:.2f}%**"
        )


    # ------------------------------------------------------
    # Customer Summary
    # ------------------------------------------------------

    st.subheader("👤 Customer Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:

        st.metric(
            "Credit Score",
            credit_score
        )

        st.metric(
            "Age",
            age
        )

    with summary_col2:

        st.metric(
            "Balance",
            f"₹{balance:,.0f}"
        )

        st.metric(
            "Tenure",
            f"{tenure} years"
        )

    with summary_col3:

        st.metric(
            "Products",
            products_number
        )

        st.metric(
            "Estimated Salary",
            f"₹{estimated_salary:,.0f}"
        )


    # ------------------------------------------------------
    # Prediction Explanation
    # ------------------------------------------------------

    st.subheader("💡 Prediction Interpretation")

    if prediction == 1:

        st.warning(
            "The Machine Learning model identifies this "
            "customer as having a higher probability of churn."
        )

    else:

        st.info(
            "The Machine Learning model identifies this "
            "customer as having a higher probability of staying."
        )


# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center">

    <p>
    🏦 <b>Bank Customer Churn Prediction</b>
    </p>

    <p class="small-text">
    Machine Learning • Random Forest • Streamlit
    </p>

    </div>
    """,
    unsafe_allow_html=True
)