import streamlit as st
import joblib
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon', quiet=True)

# Load model once
@st.cache_resource
def load_models():
    return joblib.load('src/models/final_model.pkl')

bundle = load_models()
model = bundle['salary_model']
tfidf = bundle['tfidf_vectorizer']
tfidf_matrix = bundle['tfidf_matrix']
job_db = bundle['job_database']
le_exp = bundle['label_encoder_exp']
sia = SentimentIntensityAnalyzer()

# App UI
st.set_page_config(page_title="LinkedIn Job Buddy", layout="wide")
st.title("LinkedIn Job Buddy")
st.markdown("### Salary Prediction + Smart Job Recommendations from Real LinkedIn Data")

tab1, tab2 = st.tabs(["Salary Estimator", "Find My Dream Jobs"])

with tab1:
    st.header("How much will this job pay?")
    desc = st.text_area("Paste the full job description here", height=180)

    col1, col2 = st.columns(2)
    with col1:
        exp = st.selectbox("Experience Level", options=le_exp.classes_)
        remote = st.checkbox("Remote Job")
    with col2:
        employees = st.slider("Company Size (employees)", 1, 500000, 5000)
        followers = st.slider("Company Followers", 0, 10000000, 20000)

    if st.button("Predict Salary", type="primary"):
        if desc.strip() == "":
            st.error("Please paste a job description")
        else:
            sentiment = sia.polarity_scores(desc)['compound']
            length = len(desc)
            exp_code = le_exp.transform([exp])[0]

            features = np.array([[employees, followers, 100, int(remote), exp_code, length, sentiment]])
            salary = model.predict(features)[0]

            st.success(f"**Estimated Median Salary: ${salary:,.0f} per year**")
            st.info("Based on 33,000+ real LinkedIn jobs • Average error ±$12k–15k")

with tab2:
    st.header("Find Jobs That Match Your Skills")
    user_input = st.text_area(
        "Describe your skills or dream job (e.g. Python, remote, AWS, senior)",
        height=120
    )

    if st.button("Show Me Jobs!", type="primary"):
        if user_input.strip() == "":
            st.error("Please enter your skills or desired role")
        else:
            clean = re.sub(r'[^\w\s]', ' ', user_input.lower())
            vec = tfidf.transform([clean])
            sims = cosine_similarity(vec, tfidf_matrix).flatten()
            top5 = sims.argsort()[-5:][::-1]

            st.write("### Top 5 Matching Jobs from LinkedIn")
            for i, idx in enumerate(top5):
                job = job_db.iloc[idx]
                score = sims[idx]
                st.markdown(
                    f"""
                    **{i+1}. {job['Job_Title']}**  
                    {job['Company_Name']} • **${job['Median_Salary']:,.0f}** • {'Remote' if job['Is_Remote'] else 'On-site'}  
                    **Match Score: {score:.1%}**
                    """,
                    unsafe_allow_html=True
                )
                with st.expander("Show job description"):
                    st.write(job['Job_Description_Clean'][:600] + "...")

st.markdown("---")
st.caption("• Real LinkedIn data • XGBoost + TF-IDF • 2025 Portfolio Project")