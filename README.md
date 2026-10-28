# Gaming PC / Console Game Price Analysis

**Subject:** Big Data Analytics Lab  
**Author:** sahillchahare@gmail.com  
**Technology Stack:** Python 3.10+, Streamlit, Pandas, NumPy, SciPy, Statsmodels, Scikit-Learn, Plotly  
**Deployment Platform:** Streamlit Community Cloud  

---

## 📌 Project Overview
This project is an end-to-end Big Data Analytics suite investigating the pricing dynamics, discount structures, storage overhead, and review correlations of modern AAA and Indie games across **PC (Steam & Epic Games)** and modern consoles (**PlayStation 5, Xbox Series X, Nintendo Switch**).

The application integrates all **10 curriculum practicals** required by the Big Data Analytics syllabus into an interactive, cloud-ready Streamlit web app.

---

## 🧪 Comprehensive Practicals Implemented

1. **Web Scraping and CSV Creation:** Scraping digital storefronts, resilient handling of HTTP 403 Forbidden / Cloudflare obstacles, Steam API fallback, and clean CSV persistence.
2. **Data Cleaning & Exploratory Data Analysis (EDA):** Handling missing attributes, data type casting, Tukey IQR outlier fences, summary statistics (mean, median, mode, variance, std dev), and box/histogram distributions.
3. **Probability Distributions & Sampling:** Implementing Simple Random Sampling (SRS), Systematic Sampling, and Stratified Sampling by Platform; fitting empirical data against theoretical Gaussian / Poisson / Binomial curves.
4. **Central Limit Theorem (CLT):** Drawing $K$ samples of size $n$, computing sample mean distributions, proving convergence to Gaussian normality, and comparing observed standard error against theoretical $\sigma / \sqrt{n}$.
5. **Confidence Intervals & Hypothesis Testing:** One-sample Student's t-test and Z-test, formulating $H_0$ and $H_1$, calculating p-values, degrees of freedom, and two-tailed $95\%$ and $99\%$ confidence intervals.
6. **Chi-Square Test of Independence:** Testing association between categorical factors (e.g. Platform vs Discount Category) via contingency matrix, expected frequencies, $\chi^2$ statistic, and Cramer's V.
7. **Covariance & Correlation Analysis:** Multivariable covariance matrices, Pearson linear correlation coefficients $r \in [-1, 1]$, and interactive Plotly correlation heatmaps.
8. **Correlation Hypothesis Testing:** Statistical significance of correlation coefficients ($t = r\sqrt{(n-2)/(1-r^2)}$) and non-parametric Spearman rank correlation comparison.
9. **Two-Way ANOVA:** Factorial Analysis of Variance examining main effects of Platform and Genre/Publisher Tier, together with their interaction term, on game prices.
10. **Multiple Linear Regression:** Ordinary Least Squares (OLS) estimation predicting game sale prices based on retail price, Metacritic score, user rating, and storage size, with $R^2$, MAE, and RMSE evaluation metrics.

---

## 🚀 Running the Project Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/gaming-price-analytics.git
   cd gaming-price-analytics
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

---

## 🌐 Deploying to Streamlit Community Cloud (Beginner Guide)

1. **Create a GitHub Repository:**
   - Log into [GitHub](https://github.com) and click **New Repository**.
   - Name it `gaming-price-analytics`. Set visibility to **Public**.
2. **Upload Required Files:**
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `gaming_game_prices.csv`
   - Click **Commit changes**.
3. **Connect to Streamlit Cloud:**
   - Visit [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
   - Click **New app**.
   - Select your repository `gaming-price-analytics`, branch `main`, and file `app.py`.
   - Click **Deploy!**
4. **Obtain Public URL:**
   - Your web app will compile in 1-2 minutes and provide a shareable public URL like `https://gaming-price-analytics.streamlit.app`.

---

## 🎓 Viva Voce Preparation Guide (Sample Questions & Answers)

- **Q1: Why do we divide by $n-1$ instead of $n$ when computing sample variance?**
  *Answer:* Bessel's correction corrects the bias in sample variance because the sample mean $\bar{X}$ is closer to the sample observations than the true unknown population mean $\mu$.
- **Q2: State the Central Limit Theorem and why it is critical in Big Data.**
  *Answer:* As the sample size $n$ grows ($n \ge 30$), the sampling distribution of the sample mean approaches a Normal distribution regardless of the underlying population distribution. This allows parametric hypothesis tests to be applied to non-normal real-world data.
- **Q3: What does the p-value signify in hypothesis testing?**
  *Answer:* The probability of observing a test statistic as extreme as, or more extreme than, the observed value, assuming the null hypothesis $H_0$ is true. If $p < \alpha$, we reject $H_0$.
- **Q4: Differentiate Pearson vs Spearman correlation.**
  *Answer:* Pearson measures linear relationships between interval/ratio variables. Spearman measures monotonic relationships using ranks, making it non-parametric and resilient to outliers.
