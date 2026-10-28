import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
st.set_page_config(page_title="AI Business Intelligence", page_icon="📊", layout="wide")

DATA_URL = "Online Retail.xlsx"
REQUIRED = ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate",
            "UnitPrice", "CustomerID", "Country"]

@st.cache_data(show_spinner=False)
def load_sample():
    return pd.read_excel(DATA_URL)

def read_file(file):
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

def clean_data(raw):
    df = raw.copy()
    df.columns = df.columns.astype(str).str.strip()
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df = df.dropna(subset=["InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"]).copy()
    cancelled = df["InvoiceNo"].astype(str).str.startswith("C")
    df = df.loc[(~cancelled) & (df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Revenue"])
    return df

def create_features(series):
    data = pd.DataFrame(index=series.index)
    data["Sales"] = series
    for lag in [1, 2, 3, 7, 14, 28]:
        data[f"lag_{lag}"] = series.shift(lag)
    shifted = series.shift(1)
    for window in [7, 14, 28]:
        data[f"rolling_mean_{window}"] = shifted.rolling(window).mean()
        data[f"rolling_std_{window}"] = shifted.rolling(window).std()
    data["day_of_week"] = data.index.dayofweek
    data["day_of_month"] = data.index.day
    data["month"] = data.index.month
    data["quarter"] = data.index.quarter
    data["is_weekend"] = (data.index.dayofweek >= 5).astype(int)
    return data.dropna()

def future_features(history, date):
    row = {}
    for lag in [1, 2, 3, 7, 14, 28]:
        row[f"lag_{lag}"] = float(history.iloc[-lag])
    for window in [7, 14, 28]:
        tail = history.iloc[-window:]
        row[f"rolling_mean_{window}"] = float(tail.mean())
        row[f"rolling_std_{window}"] = float(tail.std(ddof=1))
    row.update(day_of_week=date.dayofweek, day_of_month=date.day, month=date.month,
               quarter=date.quarter, is_weekend=int(date.dayofweek >= 5))
    return pd.DataFrame([row], index=[date])

@st.cache_data(show_spinner=False)
def analyse(df, horizon, n_clusters):
    daily = df.set_index("InvoiceDate")["Revenue"].resample("D").sum().asfreq("D", fill_value=0)
    if len(daily) < 45:
        raise ValueError("Please use at least 45 days of data for sales forecasting.")
    model_df = create_features(daily)
    features = [c for c in model_df.columns if c != "Sales"]
    split = min(max(int(len(model_df) * .8), 1), len(model_df) - 1)
    X_train, X_test = model_df[features].iloc[:split], model_df[features].iloc[split:]
    y_train, y_test = model_df["Sales"].iloc[:split], model_df["Sales"].iloc[split:]
    model = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=.03,
                         subsample=.85, colsample_bytree=.85,
                         objective="reg:squarederror", random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = np.maximum(model.predict(X_test), 0)
    metrics = {
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "R²": r2_score(y_test, preds) if len(y_test) > 1 else np.nan,
        "WAPE (%)": np.abs(y_test.to_numpy() - preds).sum() /
                    max(np.abs(y_test.to_numpy()).sum(), 1e-9) * 100
    }
    comparison = pd.DataFrame({"Date": y_test.index, "Actual Sales": y_test.values,
                               "Predicted Sales": preds})
    history = daily.copy()
    dates = pd.date_range(daily.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    values = []
    for date in dates:
        pred = max(0.0, float(model.predict(future_features(history, date)[features])[0]))
        values.append(pred)
        history.loc[date] = pred
    forecast = pd.DataFrame({"Date": dates, "Forecast Sales": values})

    customer = df.dropna(subset=["CustomerID"]).copy()
    if customer["CustomerID"].nunique() >= 2:
        ref_date = customer["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
        rfm = customer.groupby("CustomerID").agg(
            Recency=("InvoiceDate", lambda x: (ref_date - x.max().normalize()).days),
            Frequency=("InvoiceNo", "nunique"), Monetary=("Revenue", "sum")
        ).reset_index()
        rfm_input = np.log1p(rfm[["Recency", "Frequency", "Monetary"]].clip(lower=0))
        scaler = StandardScaler()
        scaled = scaler.fit_transform(rfm_input)
        kmeans = KMeans(n_clusters=min(n_clusters, len(rfm)), random_state=42, n_init=10)
        rfm["Segment"] = kmeans.fit_predict(scaled)
        segments = rfm.groupby("Segment").agg(
            Customers=("CustomerID", "count"), Average_Recency=("Recency", "mean"),
            Average_Orders=("Frequency", "mean"), Average_Spend=("Monetary", "mean"),
            Total_Spend=("Monetary", "sum")).round(2).reset_index()
    else:
        rfm, segments = pd.DataFrame(), pd.DataFrame()

    anomalies = daily.rename("Sales").to_frame()
    anomalies["DayOfWeek"] = anomalies.index.dayofweek
    anomalies["Month"] = anomalies.index.month
    detector = IsolationForest(n_estimators=200, contamination=.03, random_state=42)
    anomalies["Prediction"] = detector.fit_predict(anomalies[["Sales", "DayOfWeek", "Month"]])
    anomalies["Status"] = np.where(anomalies["Prediction"] == -1, "Anomaly", "Normal")
    anomalies = anomalies.reset_index().rename(columns={"InvoiceDate": "Date"})
    importance = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
    monthly = df.set_index("InvoiceDate")["Revenue"].resample("MS").sum().reset_index()
    monthly.columns = ["Month", "Revenue"]
    products = df.groupby("Description", dropna=False)["Revenue"].sum().nlargest(10).reset_index()
    countries = df.groupby("Country").agg(Revenue=("Revenue", "sum"), Orders=("InvoiceNo", "nunique")).sort_values("Revenue", ascending=False).head(15).reset_index()
    return daily, monthly, products, countries, metrics, comparison, forecast, rfm, segments, anomalies, importance

st.markdown("""
<style>
.block-container{padding-top:1.5rem}
[data-testid="stMetric"]{background:rgba(120,120,120,.08);padding:14px;border-radius:12px}
.hero{padding:1.1rem 1.4rem;border-radius:16px;background:linear-gradient(120deg,#13243a,#285d73);color:white;margin-bottom:1rem}
.hero h1{margin:0;font-size:2rem}.hero p{margin:.4rem 0 0;opacity:.9}
</style>
<div class="hero"><h1>📊 AI-Powered Business Intelligence</h1>
<p>Sales analytics • Predictive forecasting • Customer segmentation • Anomaly detection</p></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Data & Settings")
    uploaded = st.file_uploader("Upload retail data", type=["csv", "xlsx", "xls"])
    use_sample = st.checkbox("Use sample Online Retail dataset", value=uploaded is None)
    horizon = st.slider("Forecast horizon (days)", 7, 60, 30)
    clusters = st.slider("Number of customer segments", 2, 8, 4)
    currency = st.selectbox("Currency display symbol", ["£", "$", "€", "₹"], index=0)
    st.caption("Expected retail columns are listed below in the Data & Downloads tab.")

try:
    with st.spinner("Loading and cleaning data..."):
        if uploaded is not None:
            raw = read_file(uploaded)
            source = uploaded.name
        elif use_sample:
            raw = load_sample()
            source = "UCI Online Retail sample"
        else:
            st.info("Upload a CSV/Excel dataset or select the sample dataset in the sidebar.")
            st.stop()
        clean = clean_data(raw)
    if clean.empty:
        st.error("No valid transactions remain after cleaning.")
        st.stop()

    lo, hi = clean["InvoiceDate"].min().date(), clean["InvoiceDate"].max().date()
    st.sidebar.markdown(f"**Source:** {source}")
    selected = st.sidebar.date_input("Analysis date range", value=(lo, hi), min_value=lo, max_value=hi)
    if isinstance(selected, (tuple, list)) and len(selected) == 2:
        start, end = selected
    else:
        start, end = lo, hi
    df = clean[clean["InvoiceDate"].dt.date.between(start, end)].copy()
    if df.empty:
        st.warning("No rows found in that date range.")
        st.stop()
    with st.spinner("Training forecasting and segmentation models..."):
        result = analyse(df, horizon, clusters)
except Exception as e:
    st.error(f"Could not load or analyse this dataset: {e}")
    st.info("Make sure the file contains the required Online Retail columns and valid dates/numeric values.")
    st.stop()

daily, monthly, products, countries, metrics, comparison, forecast, rfm, segments, anomalies, importance = result
revenue = df["Revenue"].sum()
orders = df["InvoiceNo"].nunique()
customers = df["CustomerID"].nunique()
product_count = df["StockCode"].nunique()
aov = revenue / max(orders, 1)
st.caption(f"Source: **{source}** · Date range: {start:%d %b %Y} – {end:%d %b %Y} · {len(df):,} cleaned transaction rows")
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Total Revenue", f"{currency}{revenue:,.2f}")
c2.metric("Orders", f"{orders:,}")
c3.metric("Customers", f"{customers:,}")
c4.metric("Products", f"{product_count:,}")
c5.metric("Average Order Value", f"{currency}{aov:,.2f}")

overview, forecast_tab, customer_tab, anomaly_tab, data_tab = st.tabs([
    "📈 Business Overview", "🔮 Sales Forecast", "👥 Customer Segments",
    "🚨 Anomaly Detection", "🗂️ Data & Downloads"
])
with overview:
    st.subheader("Revenue performance")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.line(monthly, x="Month", y="Revenue", markers=True, title="Monthly Revenue Trend"), use_container_width=True)
    with right:
        st.plotly_chart(px.bar(countries.head(10), x="Country", y="Revenue", title="Top Countries by Revenue"), use_container_width=True)
    left, right = st.columns(2)
    with left:
        products["Description"] = products["Description"].fillna("Unknown")
        st.plotly_chart(px.bar(products.sort_values("Revenue"), x="Revenue", y="Description", orientation="h", title="Top 10 Products by Revenue"), use_container_width=True)
    with right:
        st.markdown("#### Country performance")
        st.dataframe(countries, use_container_width=True, hide_index=True)

with forecast_tab:
    st.subheader(f"Next {horizon} days of sales")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("MAE", f"{currency}{metrics['MAE']:,.2f}")
    m2.metric("RMSE", f"{currency}{metrics['RMSE']:,.2f}")
    m3.metric("R²", f"{metrics['R²']:.3f}" if pd.notna(metrics["R²"]) else "N/A")
    m4.metric("WAPE", f"{metrics['WAPE (%)']:.2f}%")
    st.caption("Evaluation metrics use a chronological holdout set (last 20% of feature rows). Forecasts are estimates, not guarantees.")
    st.plotly_chart(px.line(comparison, x="Date", y=["Actual Sales", "Predicted Sales"], title="Actual vs Predicted Sales"), use_container_width=True)
    st.plotly_chart(px.line(forecast, x="Date", y="Forecast Sales", markers=True, title=f"{horizon}-Day Sales Forecast"), use_container_width=True)
    st.metric("Total forecast revenue", f"{currency}{forecast['Forecast Sales'].sum():,.2f}")
    st.dataframe(forecast, use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(importance.head(15).sort_values("Importance"), x="Importance", y="Feature", orientation="h", title="Forecast Feature Importance"), use_container_width=True)

with customer_tab:
    st.subheader("RFM customer segmentation")
    if rfm.empty:
        st.warning("Not enough distinct CustomerID values to build customer segments.")
    else:
        left, right = st.columns([1.2, 1])
        with left:
            st.plotly_chart(px.scatter(rfm, x="Frequency", y="Monetary", color=rfm["Segment"].astype(str),
                hover_data=["CustomerID", "Recency"], title="Customer Segments: Orders vs Spend"), use_container_width=True)
        with right:
            st.markdown("#### Segment summary")
            st.dataframe(segments, use_container_width=True, hide_index=True)
        st.markdown("#### Customer-level RFM table")
        st.dataframe(rfm.sort_values("Monetary", ascending=False), use_container_width=True, hide_index=True)

with anomaly_tab:
    st.subheader("Unusual daily sales")
    count = int((anomalies["Status"] == "Anomaly").sum())
    st.metric("Flagged days", f"{count:,}")
    st.plotly_chart(px.scatter(anomalies, x="Date", y="Sales", color="Status", title="Daily Sales Anomaly Detection",
        color_discrete_map={"Anomaly": "#e45756", "Normal": "#4c78a8"}), use_container_width=True)
    st.caption("Isolation Forest identifies statistical outliers; a flagged day is not automatically an error or fraud.")
    st.dataframe(anomalies[anomalies["Status"] == "Anomaly"].sort_values("Sales", ascending=False), use_container_width=True, hide_index=True)

with data_tab:
    st.subheader("Cleaned data and exports")
    st.write(f"Rows: **{len(df):,}** · Columns: **{len(df.columns)}**")
    st.dataframe(df.head(500), use_container_width=True)
    st.caption("Preview limited to 500 rows; downloads include all rows in the selected date range.")
    downloads = [
        ("Cleaned data", df, "cleaned_retail_data.csv"),
        ("Sales forecast", forecast, "sales_forecast.csv"),
        ("Anomaly report", anomalies, "daily_sales_anomalies.csv"),
        ("Forecast metrics", pd.DataFrame({"Metric": list(metrics.keys()), "Value": list(metrics.values())}), "forecast_metrics.csv"),
    ]
    if not rfm.empty:
        downloads.append(("Customer segments", rfm, "customer_segments.csv"))
    for label, table, filename in downloads:
        st.download_button(f"⬇️ Download {label} (CSV)", table.to_csv(index=False).encode("utf-8"),
                           file_name=filename, mime="text/csv", key=filename)
    st.markdown("**Required columns for uploaded files**")
    st.code(", ".join(REQUIRED), language="text")

st.divider()
st.caption("Built with Streamlit, Plotly, XGBoost and scikit-learn.")
