"""
Gaming PC / Console Game Price Analysis
Big Data Analytics Lab - Complete Project Suite
Author: sahillchahare@gmail.com
Deployment: Streamlit Community Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
import io

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Gaming Price Analytics | Big Data Lab",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background: #F8FAFC;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #3B82F6;
    }
    .expander-header {
        font-weight: 600;
        color: #2563EB;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("gaming_game_prices.csv")
    except Exception:
        # Fallback dataset if file is not found
        data = {
            "id": list(range(1, 11)),
            "title": [
                "Cyberpunk 2077: Phantom Liberty", "Cyberpunk 2077 (PS5)", "Elden Ring (Steam)", 
                "Elden Ring (PS5)", "Baldur's Gate 3 (Steam)", "Spider-Man 2 (PS5)",
                "Zelda: Tears of Kingdom", "Forza Horizon 5 (Xbox)", "Hades (Steam)", "Red Dead Redemption 2"
            ],
            "platform": ["PC (Steam)", "PlayStation 5", "PC (Steam)", "PlayStation 5", "PC (Steam)", "PlayStation 5", "Nintendo Switch", "Xbox Series X", "PC (Steam)", "PC (Steam)"],
            "genre": ["Action RPG", "Action RPG", "Action RPG", "Action RPG", "Action RPG", "Open World", "Adventure", "Sports / Racing", "Indie / Roguelike", "Open World"],
            "originalPrice": [59.99, 49.99, 59.99, 59.99, 59.99, 69.99, 69.99, 59.99, 24.99, 59.99],
            "discountedPrice": [35.99, 29.99, 41.99, 49.99, 53.99, 49.99, 59.99, 34.99, 12.49, 19.79],
            "discountPercent": [40, 40, 30, 16, 10, 28, 14, 42, 50, 67],
            "metacriticScore": [89, 88, 96, 96, 96, 90, 96, 92, 93, 97],
            "userRating": [8.7, 8.4, 9.3, 9.2, 9.6, 8.9, 9.2, 8.9, 9.8, 9.4],
            "releaseYear": [2023, 2022, 2022, 2022, 2023, 2023, 2023, 2021, 2020, 2019],
            "publisherTier": ["AAA Studio", "AAA Studio", "AAA Studio", "AAA Studio", "AAA Studio", "AAA Studio", "AAA Studio", "AAA Studio", "Indie Developer", "AAA Studio"],
            "storageGB": [70, 56, 60, 45, 150, 86, 16, 103, 15, 150],
            "hardwareTier": ["High-End", "Console-Optimized", "High-End", "Console-Optimized", "High-End", "Console-Optimized", "Console-Optimized", "Console-Optimized", "Entry-Level", "High-End"],
            "hasMultiplayer": [False, False, True, True, True, False, False, True, False, True],
            "userReviewsCount": [650000, 220000, 820000, 410000, 540000, 310000, 480000, 220000, 280000, 610000]
        }
        df = pd.DataFrame(data)
    return df

df_raw = load_data()

# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/808/808439.png", width=70)
st.sidebar.title("Big Data Analytics Lab")
st.sidebar.caption("Gaming PC / Console Game Price Analysis")

topics = [
    "Project Overview & Raw Data",
    "Web Scraping and CSV Creation",
    "Data Cleaning and Exploratory Data Analysis",
    "Probability Distributions and Sampling",
    "Central Limit Theorem",
    "Confidence Intervals and Hypothesis Testing",
    "Chi-Square Test",
    "Covariance and Correlation",
    "Correlation Hypothesis Testing",
    "Two-Way ANOVA",
    "Regression Analysis & Price Predictor",
]

selected_topic = st.sidebar.radio("Select Practical Topic:", topics)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Viva Tip:** Expand the 'Theoretical Explanation & Formulas' tab inside any topic for viva voice preparation.")

# =========================================================
# 0. Overview
# =========================================================
if selected_topic == "Project Overview & Raw Data":
    st.markdown('<div class="main-header">Gaming PC & Console Game Price Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Big Data Analytics Lab Project | Streamlit Community Cloud</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Games", len(df_raw))
    col2.metric("Platforms", df_raw['platform'].nunique())
    col3.metric("Avg Discount", f"{df_raw['discountPercent'].mean():.1f}%")
    col4.metric("Avg Metacritic", f"{df_raw['metacriticScore'].mean():.1f}/100")
    
    st.markdown("### Dataset Preview")
    st.dataframe(df_raw, use_container_width=True)
    
    csv_bytes = df_raw.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Gaming Game Prices CSV",
        data=csv_bytes,
        file_name="gaming_game_prices.csv",
        mime="text/csv"
    )

# =========================================================
# 1. Web Scraping & CSV Creation
# =========================================================
elif selected_topic == "Web Scraping and CSV Creation":
    st.markdown('<div class="main-header">Web Scraping and CSV Creation</div>', unsafe_allow_html=True)
    
    with st.expander("📘 Theoretical Explanation & Important Concepts", expanded=True):
        st.markdown("""
        **What it is used for:** Web scraping automates the extraction of unstructured web data from digital storefronts (e.g. Steam Community Market, PlayStation Store, Deku Deals) and converts it into structured CSV tabular format for big data analytics.
        
        **Important Techniques & Best Practices:**
        - **HTTP Status Codes:** Handles `200 OK`, `403 Forbidden` (anti-bot / Cloudflare blocks), `404 Not Found`, and `429 Too Many Requests` gracefully.
        - **User-Agent Headers:** Pass realistic browser headers (`User-Agent: Mozilla/5.0...`) to prevent immediate 403 blocks.
        - **API Fallbacks:** When HTML scraping is blocked, fallback to permitted public REST APIs like Steam API (`https://store.steampowered.com/api/appdetails`).
        """)

    st.subheader("Simulated Web Scraper with Error Handling")
    scrape_target = st.selectbox("Select Target Storefront to Scrape:", ["Steam Storefront (Direct)", "PlayStation Network Store", "Xbox Marketplace"])
    
    if st.button("🚀 Run Live Scraping Simulator"):
        with st.spinner(f"Connecting to {scrape_target}..."):
            st.code("""
# Production Web Scraping Script
import requests
from bs4 import BeautifulSoup
import pandas as pd

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
url = 'https://store.steampowered.com/search/?filter=topsellers'

try:
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status() # Raises HTTPError for 403, 404, 500
    soup = BeautifulSoup(response.text, 'html.parser')
    print("Scraping Successful (Status 200 OK)!")
except requests.exceptions.HTTPError as err:
    print(f"HTTP Error encountered: {err}")
    print("Falling back to Steam Web API endpoint...")
            """, language="python")
            st.success("✅ Successfully extracted 100 game entries with zero missing fields! Ready for CSV serialization.")
            st.dataframe(df_raw[['title', 'platform', 'originalPrice', 'discountedPrice', 'metacriticScore']].head(10))

# =========================================================
# 2. Data Cleaning & Exploratory Data Analysis (EDA)
# =========================================================
elif selected_topic == "Data Cleaning and Exploratory Data Analysis":
    st.markdown('<div class="main-header">Data Cleaning and Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    with st.expander("📘 Theoretical Explanation, Formulas & Interpretations", expanded=True):
        st.markdown(r"""
        **What it is used for:** EDA uncovers data distributions, detects anomalies/outliers, checks for missing entries, and computes central tendency & dispersion.
        
        **Important Formulas:**
        - **Mean:** $\\bar{X} = \\frac{1}{n} \\sum X_i$
        - **Median:** Middle value of sorted observations.
        - **Sample Variance:** $S^2 = \\frac{1}{n-1} \\sum (X_i - \\bar{X})^2$
        - **Standard Deviation:** $S = \\sqrt{S^2}$
        - **Interquartile Range (IQR):** $IQR = Q_3 - Q_1$
        - **Outlier Boundary:** $[Q_1 - 1.5 \\times IQR, \\, Q_3 + 1.5 \\times IQR]$
        """)
        
    num_col = st.selectbox("Select Numerical Feature for EDA:", ["originalPrice", "discountedPrice", "discountPercent", "metacriticScore", "userRating", "storageGB"])
    
    colA, colB, colC, colD = st.columns(4)
    s = df_raw[num_col]
    colA.metric("Mean", f"{s.mean():.2f}")
    colB.metric("Median", f"{s.median():.2f}")
    colC.metric("Std Dev", f"{s.std():.2f}")
    colD.metric("IQR", f"{(s.quantile(0.75) - s.quantile(0.25)):.2f}")
    
    c1, c2 = st.columns(2)
    with c1:
        fig_hist = px.histogram(df_raw, x=num_col, color="platform", marginal="box", title=f"Distribution of {num_col} by Platform", nbins=20)
        st.plotly_chart(fig_hist, use_container_width=True)
    with c2:
        fig_box = px.box(df_raw, x="platform", y=num_col, color="platform", points="all", title=f"Box Plot & Outlier Detection for {num_col}")
        st.plotly_chart(fig_box, use_container_width=True)

# =========================================================
# 3. Probability Distributions & Sampling
# =========================================================
elif selected_topic == "Probability Distributions and Sampling":
    st.markdown('<div class="main-header">Probability Distributions and Sampling</div>', unsafe_allow_html=True)
    
    with st.expander("📘 Theoretical Explanation & Formulas", expanded=True):
        st.markdown(r"""
        **Techniques Covered:**
        - **Simple Random Sampling (SRS):** Every game record has an equal probability of selection $P = \\frac{n}{N}$.
        - **Systematic Sampling:** Select every $k$-th element where $k = \\lfloor N / n \\rfloor$.
        - **Stratified Sampling:** Divide population into strata (e.g. by Platform or Publisher Tier) and sample proportionally: $n_i = n \\times \\frac{N_i}{N}$.
        - **Normal Distribution PDF:** $f(x) = \\frac{1}{\\sigma \\sqrt{2\\pi}} e^{-\\frac{(x-\\mu)^2}{2\\sigma^2}}$
        """)

    tab1, tab2 = st.tabs(["🎲 Sampling Demonstrator", "📈 Distribution Fitting"])
    
    with tab1:
        method = st.radio("Select Sampling Technique:", ["Simple Random Sampling", "Systematic Sampling", "Stratified Sampling (by Platform)"])
        sample_size = st.slider("Select Sample Size (n):", min_value=5, max_value=min(60, len(df_raw)), value=20)
        
        if method == "Simple Random Sampling":
            sample_df = df_raw.sample(n=sample_size, random_state=42)
        elif method == "Systematic Sampling":
            k = max(1, len(df_raw) // sample_size)
            sample_df = df_raw.iloc[::k].head(sample_size)
        else:
            sample_df = df_raw.groupby('platform', group_keys=False).apply(
                lambda x: x.sample(max(1, int(len(x)/len(df_raw)*sample_size)), random_state=42)
            )
            
        st.write(f"Sampled {len(sample_df)} records using {method}:")
        st.dataframe(sample_df[['title', 'platform', 'originalPrice', 'discountedPrice', 'metacriticScore']], use_container_width=True)
        st.download_button("📥 Download Sampled CSV", sample_df.to_csv(index=False).encode('utf-8'), "sampled_games.csv", "text/csv")
        
    with tab2:
        feature = st.selectbox("Feature to fit Normal Distribution:", ["discountedPrice", "metacriticScore", "userRating"])
        vals = df_raw[feature].dropna()
        mu, std = stats.norm.fit(vals)
        
        x_axis = np.linspace(vals.min(), vals.max(), 100)
        pdf_fitted = stats.norm.pdf(x_axis, mu, std)
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=vals, histnorm='probability density', name='Observed Data', opacity=0.6))
        fig.add_trace(go.Scatter(x=x_axis, y=pdf_fitted, mode='lines', name=f'Fitted Normal (μ={mu:.2f}, σ={std:.2f})', line=dict(color='red', width=3)))
        fig.update_layout(title=f"Normal Curve Overlay for {feature}", xaxis_title=feature, yaxis_title="Probability Density")
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 4. Central Limit Theorem
# =========================================================
elif selected_topic == "Central Limit Theorem":
    st.markdown('<div class="main-header">Central Limit Theorem (CLT)</div>', unsafe_allow_html=True)
    
    with st.expander("📘 Theoretical Explanation & Formulas", expanded=True):
        st.markdown(r"""
        **Central Limit Theorem:** Given an arbitrary population distribution with mean $\\mu$ and standard deviation $\\sigma$, the sampling distribution of the sample mean $\\bar{X}$ approaches a Gaussian (normal) distribution as sample size $n \\ge 30$, regardless of the shape of the underlying population!
        
        **Theoretical Standard Error:** $SE_{theoretical} = \\frac{\\sigma}{\\sqrt{n}}$
        """)
        
    c1, c2 = st.columns(2)
    with c1:
        sample_n = st.slider("Sample Size (n per draw):", min_value=5, max_value=80, value=30, step=5)
    with c2:
        num_sims = st.slider("Number of Simulations (K trials):", min_value=100, max_value=2000, value=800, step=100)
        
    pop_values = df_raw['discountedPrice'].values
    pop_mean = np.mean(pop_values)
    pop_std = np.std(pop_values, ddof=1)
    theo_se = pop_std / np.sqrt(sample_n)
    
    sample_means = [np.mean(np.random.choice(pop_values, size=sample_n, replace=True)) for _ in range(num_sims)]
    obs_mean = np.mean(sample_means)
    obs_se = np.std(sample_means, ddof=1)
    
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Population Mean (μ)", f"${pop_mean:.2f}")
    colB.metric("Mean of Sample Means", f"${obs_mean:.2f}")
    colC.metric("Theoretical SE (σ/√n)", f"${theo_se:.3f}")
    colD.metric("Observed SE (s_means)", f"${obs_se:.3f}")
    
    fig = px.histogram(sample_means, nbins=30, title=f"Sampling Distribution of Mean Discounted Price (n={sample_n}, K={num_sims})", labels={'value': 'Sample Mean ($)'})
    fig.add_vline(x=pop_mean, line_dash="dash", line_color="green", annotation_text="True Pop Mean (μ)")
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 5. Confidence Intervals and Hypothesis Testing
# =========================================================
elif selected_topic == "Confidence Intervals and Hypothesis Testing":
    st.markdown('<div class="main-header">Confidence Intervals and Hypothesis Testing</div>', unsafe_allow_html=True)
    
    with st.expander("📘 Theoretical Explanation & Formulas", expanded=True):
        st.markdown(r"""
        **One-Sample Student's t-Test:**
        - **Null Hypothesis ($H_0$):** $\\mu = \\mu_0$
        - **Alternative Hypothesis ($H_1$):** $\\mu \\ne \\mu_0$
        - **Test Statistic:** $t = \\frac{\\bar{X} - \\mu_0}{S / \\sqrt{n}}$
        """)
        
    c1, c2 = st.columns(2)
    with c1:
        hyp_mean = st.number_input("Hypothesized Mean Price (μ₀ in $):", min_value=10.0, max_value=80.0, value=45.0, step=1.0)
    with c2:
        conf_level = st.selectbox("Confidence Level (1 - α):", [0.90, 0.95, 0.99], index=1)
        
    prices = df_raw['discountedPrice'].dropna()
    n = len(prices)
    xbar = np.mean(prices)
    s = np.std(prices, ddof=1)
    se = s / np.sqrt(n)
    alpha = 1 - conf_level
    df = n - 1
    
    t_stat, p_val = stats.ttest_1samp(prices, hyp_mean)
    t_crit = stats.t.ppf(1 - alpha/2, df)
    ci_lower = xbar - t_crit * se
    ci_upper = xbar + t_crit * se
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sample Mean (X̄)", f"${xbar:.2f}")
    col2.metric("t-Statistic", f"{t_stat:.3f}")
    col3.metric("p-Value", f"{p_val:.4f}")
    col4.metric(f"{int(conf_level*100)}% CI", f"[${ci_lower:.2f}, ${ci_upper:.2f}]")
    
    if p_val < alpha:
        st.error(f"❌ **Reject H₀** (p-value {p_val:.4f} < α {alpha:.2f}). Significant difference from ${hyp_mean:.2f}.")
    else:
        st.success(f"✅ **Fail to Reject H₀** (p-value {p_val:.4f} ≥ α {alpha:.2f}). Consistent with ${hyp_mean:.2f}.")

# =========================================================
# 6. Chi-Square Test
# =========================================================
elif selected_topic == "Chi-Square Test":
    st.markdown('<div class="main-header">Chi-Square Test of Independence</div>', unsafe_allow_html=True)
    
    df_raw['Discount_Status'] = df_raw['discountPercent'].apply(lambda x: "High (>35%)" if x > 35 else "Low/None (<=35%)")
    
    c1, c2 = st.columns(2)
    with c1:
        cat1 = st.selectbox("Categorical Factor A:", ["platform", "publisherTier", "hardwareTier"])
    with c2:
        cat2 = st.selectbox("Categorical Factor B:", ["Discount_Status", "hasMultiplayer", "genre"])
        
    contingency = pd.crosstab(df_raw[cat1], df_raw[cat2])
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
    
    st.write("### Observed Contingency Table (O):")
    st.dataframe(contingency, use_container_width=True)
    
    colA, colB, colC = st.columns(3)
    colA.metric("Chi-Square Statistic (χ²)", f"{chi2:.3f}")
    colB.metric("Degrees of Freedom (df)", dof)
    colC.metric("p-Value", f"{p_val:.4f}")
    
    if p_val < 0.05:
        st.error(f"❌ **Reject H₀** (p = {p_val:.4f} < 0.05). Significant association between {cat1} and {cat2}.")
    else:
        st.success(f"✅ **Fail to Reject H₀** (p = {p_val:.4f} >= 0.05). Variables are independent.")

# =========================================================
# 7. Covariance and Correlation
# =========================================================
elif selected_topic == "Covariance and Correlation":
    st.markdown('<div class="main-header">Covariance and Correlation Analysis</div>', unsafe_allow_html=True)
    
    num_cols = ["originalPrice", "discountedPrice", "discountPercent", "metacriticScore", "userRating", "storageGB"]
    corr_matrix = df_raw[num_cols].corr()
    
    fig_heat = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Pearson Correlation Matrix")
    st.plotly_chart(fig_heat, use_container_width=True)

# =========================================================
# 8. Correlation Hypothesis Testing
# =========================================================
elif selected_topic == "Correlation Hypothesis Testing":
    st.markdown('<div class="main-header">Correlation Hypothesis Testing</div>', unsafe_allow_html=True)
    
    num_cols = ["discountedPrice", "originalPrice", "metacriticScore", "userRating", "storageGB"]
    c1, c2 = st.columns(2)
    with c1:
        var1 = st.selectbox("Variable 1:", num_cols, index=0)
    with c2:
        var2 = st.selectbox("Variable 2:", num_cols, index=2)
        
    pearson_r, pearson_p = stats.pearsonr(df_raw[var1], df_raw[var2])
    spearman_rho, spearman_p = stats.spearmanr(df_raw[var1], df_raw[var2])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pearson r", f"{pearson_r:.3f}")
    col2.metric("Pearson p-value", f"{pearson_p:.4e}")
    col3.metric("Spearman ρ", f"{spearman_rho:.3f}")
    col4.metric("Spearman p-value", f"{spearman_p:.4e}")

# =========================================================
# 9. Two-Way ANOVA
# =========================================================
elif selected_topic == "Two-Way ANOVA":
    st.markdown('<div class="main-header">Two-Way Analysis of Variance (ANOVA)</div>', unsafe_allow_html=True)
    
    model = ols('discountedPrice ~ C(platform) + C(publisherTier) + C(platform):C(publisherTier)', data=df_raw).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    st.write("### Two-Way ANOVA Summary Table:")
    st.dataframe(anova_table.round(4), use_container_width=True)

# =========================================================
# 10. Regression Analysis
# =========================================================
elif selected_topic == "Regression Analysis & Price Predictor":
    st.markdown('<div class="main-header">Multiple Linear Regression & Price Predictor</div>', unsafe_allow_html=True)
    
    features = ["originalPrice", "metacriticScore", "userRating", "storageGB"]
    X = df_raw[features]
    y = df_raw["discountedPrice"]
    
    reg = LinearRegression()
    reg.fit(X, y)
    y_pred = reg.predict(X)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{r2_score(y, y_pred):.3f}")
    col2.metric("Mean Absolute Error (MAE)", f"${mean_absolute_error(y, y_pred):.2f}")
    col3.metric("RMSE", f"${np.sqrt(mean_squared_error(y, y_pred)):.2f}")
    
    st.subheader("🎮 Interactive Game Price Predictor")
    c1, c2, c3, c4 = st.columns(4)
    p_orig = c1.number_input("Original Retail Price ($)", 10.0, 100.0, 59.99)
    p_meta = c2.slider("Metacritic Score", 40, 100, 88)
    p_user = c3.slider("User Rating (0-10)", 3.0, 10.0, 8.5)
    p_stor = c4.number_input("Storage Size (GB)", 1, 250, 70)
    
    pred_val = reg.predict([[p_orig, p_meta, p_user, p_stor]])[0]
    st.success(f"💰 **Predicted Real-World Discounted Price:** ${max(4.99, pred_val):.2f}")
