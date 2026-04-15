import streamlit as st
import pandas as pd
import plotly.express as px
import requests

from predictive_model import load_and_prepare, create_features, train, predict_company

st.set_page_config(page_title="Indian Companies Hiring, Layoff and Fund Predictions", layout="wide")

# =========================
# STYLE
# =========================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #0f172a, #020617);
}
h1, h2, h3 {color: #f8fafc;}

[data-testid="stMetric"] {
    background: rgba(17, 24, 39, 0.85);
    padding: 18px;
    border-radius: 14px;
}

.card {
    background: rgba(17,24,39,0.85);
    padding: 20px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.08);
}

.avatar {
    width: 70px;
    height: 70px;
    border-radius: 50%;
    background: linear-gradient(135deg, #9333ea, #1e3a8a);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    color: white;
    font-size: 24px;
}
</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown("""
<div style="background: linear-gradient(90deg, #1e3a8a, #9333ea);
padding: 25px;border-radius: 15px;text-align: center;color: white;">
<h1>🚀 Indian Companies Hiring, Layoff and Fund Predictions</h1>
<p>Predict • Compare • Analyze Company Growth</p>
</div>
""", unsafe_allow_html=True)

# LOGO
def get_logo(company):
    domain = company.lower().replace(" ", "") + ".com"
    url = f"https://logo.clearbit.com/{domain}"
    try:
        r = requests.get(url, timeout=3)
        if r.status_code == 200:
            return url
    except:
        pass
    return None

# LOAD
@st.cache_data
def load_all():
    df = load_and_prepare()
    df_feat, le = create_features(df)
    models, features, accuracy = train(df_feat)
    return df, df_feat, models, features, le, accuracy

df, df_feat, models, features, le, accuracy = load_all()

# SIDEBAR
st.sidebar.title("🔍 Controls")
mode = st.sidebar.radio("Mode", ["Single Company", "Compare Companies"])
companies = df["Company"].unique()

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Model Confidence")

avg_acc = sum(accuracy.values()) / len(accuracy)
st.sidebar.metric("Accuracy", round(avg_acc, 3))

# =========================
# SINGLE COMPANY
# =========================
if mode == "Single Company":

    company = st.sidebar.selectbox("Select Company", companies)
    result = predict_company(company, df_feat, models, features, le)

    if result:
        hist = result["history"]
        preds = result["predictions"]
        pred_df = pd.DataFrame(preds)
        last = hist.iloc[-1]

        st.markdown("## 🏢 Company Overview")

        col_logo, col_info = st.columns([1,3])

        with col_logo:
            logo_url = get_logo(company)
            if logo_url:
                st.image(logo_url, width=80)
            else:
                st.markdown(f'<div class="avatar">{company[:2].upper()}</div>', unsafe_allow_html=True)

        with col_info:
            st.markdown(f"""
            <div class="card">
            <h3>{company}</h3>
            <p>{last['Industry']}</p>
            📍 {last['Location_HQ']}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("## 📌 Key Metrics")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Hiring", int(last["Hiring"]))
        c2.metric("Layoff", int(last["Layoff"]))
        c3.metric("Fund", int(last["Fund"]))

        growth = (preds[-1]["Hiring"] - last["Hiring"]) / (last["Hiring"] + 1e-9)
        c4.metric("Growth", f"{growth*100:.1f}%")

        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(["📈 Overview", "📊 Analytics", "🔮 Predictions"])

        with tab1:
            combined = pd.concat([
                hist[["Year","Hiring","Layoff","Fund"]],
                pred_df[["Year","Hiring","Layoff","Fund"]]
            ])
            st.plotly_chart(px.line(combined, x="Year",
                                    y=["Hiring","Layoff","Fund"],
                                    template="plotly_dark"),
                            use_container_width=True)

        with tab2:
            bar_df = pd.DataFrame({
                "Type": ["Hiring", "Layoff"],
                "Value": [last["Hiring"], last["Layoff"]]
            })
            st.plotly_chart(px.bar(bar_df, x="Type", y="Value",
                                   template="plotly_dark"),
                            use_container_width=True)

        with tab3:
            st.subheader("📊 Future Predictions")
            st.dataframe(pred_df)
            
            # 📈 Prediction Graph (Hiring, Layoff, Fund)
            st.subheader("📈 Future Trend Graph")

            st.plotly_chart(
                px.line(
                    pred_df,
                    x="Year",
                    y=["Hiring", "Layoff", "Fund"],
                    markers=True,
                    template="plotly_dark"
                ),
                use_container_width=True
            )

            ratio = preds[-1]["Layoff"] / (preds[-1]["Hiring"] + 1e-9)

            if ratio > 0.4:
                st.error("High Risk 🔴")
            elif ratio > 0.2:
                st.warning("Medium Risk 🟡")
            else:
                st.success("Low Risk 🟢")

            st.subheader("🎯 Company Health Score")

            net = last["Hiring"] - last["Layoff"]
            score = net / (last["Hiring"] + last["Layoff"] + 1e-9)
            score = (score + 1) / 2
            st.progress(score)

        st.markdown("## 📊 Model Accuracy")

        a1, a2, a3 = st.columns(3)
        a1.metric("Hiring", accuracy["Hiring"])
        a2.metric("Layoff", accuracy["Layoff"])
        a3.metric("Fund", accuracy["Fund"])

# =========================
# COMPARE MODE (FIXED)
# =========================
else:

    selected = st.sidebar.multiselect("Select Companies", companies)

    if len(selected) < 2:
        st.warning("Select at least 2 companies")
        st.stop()

    all_data = []
    pred_list = []

    for comp in selected:
        res = predict_company(comp, df_feat, models, features, le)

        if res:
            temp = res["history"].copy()
            temp["Company"] = comp
            all_data.append(temp)

            p = pd.DataFrame(res["predictions"])
            p["Company"] = comp
            pred_list.append(p)

    df_all = pd.concat(all_data)
    pred_all = pd.concat(pred_list)

    st.subheader("📈 Hiring Comparison")
    st.plotly_chart(px.line(df_all, x="Year", y="Hiring", color="Company", template="plotly_dark"))

    st.subheader("📉 Layoff Comparison")
    st.plotly_chart(px.line(df_all, x="Year", y="Layoff", color="Company", template="plotly_dark"))

    st.subheader("💰 Fund Comparison")
    st.plotly_chart(px.line(df_all, x="Year", y="Fund", color="Company", template="plotly_dark"))

    # 🔮 PREDICTIONS (FIXED)
    st.subheader("🔮 Future Hiring Comparison")
    st.plotly_chart(px.line(pred_all, x="Year", y="Hiring", color="Company", template="plotly_dark"))

    st.subheader("📉 Future Layoff Comparison")
    st.plotly_chart(px.line(pred_all, x="Year", y="Layoff", color="Company", template="plotly_dark"))

    st.subheader("💰 Future Fund Comparison")
    st.plotly_chart(px.line(pred_all, x="Year", y="Fund", color="Company", template="plotly_dark"))

    # RANKING
    st.subheader("🏆 Company Ranking (Future)")

    ranking = []
    for comp in selected:
        res = predict_company(comp, df_feat, models, features, le)
        last_pred = res["predictions"][-1]

        score = last_pred["Hiring"] - last_pred["Layoff"]

        ranking.append({
            "Company": comp,
            "Score": score
        })

    rank_df = pd.DataFrame(ranking).sort_values("Score", ascending=False)
    st.dataframe(rank_df)