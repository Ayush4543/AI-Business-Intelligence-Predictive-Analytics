# AI-Powered Business Intelligence & Predictive Analytics Platform

An interactive data analytics application built with Python and Streamlit that transforms retail transaction data into actionable business insights through sales analytics, machine learning-based forecasting, customer segmentation, and anomaly detection.

## Overview

Businesses generate large volumes of sales and customer data, but extracting meaningful insights from this data can be challenging. This project provides an interactive dashboard that helps analyze historical retail performance, understand customer purchasing behavior, forecast future sales, and identify unusual sales patterns.

The application uses data analytics, visualization, and machine learning techniques to support data-driven business decisions.

## Key Features

### 1. Business Intelligence Dashboard
- Total revenue and order analysis
- Customer and product performance metrics
- Average Order Value (AOV)
- Monthly sales trends
- Top-performing products
- Country-wise sales analysis

### 2. Sales Forecasting
- Predicts future daily sales using XGBoost
- Configurable forecasting horizon from 7 to 60 days
- Evaluates model performance using MAE, RMSE, R², and WAPE
- Displays feature importance to help interpret model behavior

### 3. Customer Segmentation
- Uses RFM analysis: Recency, Frequency, and Monetary value
- Applies K-Means clustering to group customers
- Helps identify different customer purchasing patterns
- Supports targeted marketing and customer-retention strategies

### 4. Sales Anomaly Detection
- Uses Isolation Forest to identify unusual daily sales patterns
- Highlights potential spikes and drops in sales
- Helps businesses investigate unusual transactions or performance changes

### 5. Data Upload and Export
- Upload retail data in CSV or Excel format
- Analyze the provided sample retail dataset
- Download processed results and analytical outputs as CSV files

## Technologies Used

- **Programming Language:** Python
- **Web Framework:** Streamlit
- **Data Analysis:** Pandas, NumPy
- **Data Visualization:** Plotly
- **Machine Learning:** Scikit-learn, XGBoost
- **Data Processing:** CSV and Excel

## Dataset

The project is designed around the Online Retail dataset from the UCI Machine Learning Repository.

Dataset source: https://archive.ics.uci.edu/dataset/352/online+retail

The dataset contains retail transaction records with fields such as invoice number, product code, product description, quantity, invoice date, unit price, customer ID, and country.

The application expects the following columns:

- `InvoiceNo`
- `StockCode`
- `Description`
- `Quantity`
- `InvoiceDate`
- `UnitPrice`
- `CustomerID`
- `Country`

You can use the sample dataset option in the application or upload your own compatible CSV or Excel file.

## Project Structure

```text
AI-Business-Intelligence-Predictive-Analytics/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

### Prerequisites

- Python 3.10 or a compatible Python version
- pip package manager
- Git (optional, for cloning the repository)

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/AI-Business-Intelligence-Predictive-Analytics.git
cd AI-Business-Intelligence-Predictive-Analytics
```

Replace `YOUR-USERNAME` with your GitHub username.

### Step 2: Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell activation is blocked, you can install the dependencies without activating the environment by running `python -m pip install -r requirements.txt` with the correct Python interpreter.

### Step 3: Install dependencies

```bash
python -m pip install -r requirements.txt
```


## How to Use

1. Start the Streamlit application.
2. Choose the sample retail dataset or upload a compatible CSV/Excel file.
3. Explore business KPIs and sales visualizations.
4. Configure the sales forecasting period and review model predictions.
5. Analyze customer segments using RFM metrics and K-Means clustering.
6. Review unusual sales patterns detected by the anomaly detection model.
7. Download available analytical results for further analysis.

## Machine Learning Methods

| Method | Purpose |
|---|---|
| XGBoost | Daily sales forecasting |
| RFM Analysis | Customer behavior analysis |
| K-Means Clustering | Customer segmentation |
| Isolation Forest | Sales anomaly detection |

## Potential Business Applications

- Retail performance monitoring
- Sales planning and forecasting
- Customer segmentation and retention
- Identification of unusual sales behavior
- Data-driven marketing decisions
- Business reporting and analytics

## Future Enhancements

- Interactive date-range filters
- Product-level demand forecasting
- Advanced customer lifetime value analysis
- Automated PDF business reports
- Integration with live sales databases
- Cloud deployment and role-based access
- Forecast comparison across multiple machine learning models

## Author

**Dushyant Tomar**

Computer Science Engineering | Artificial Intelligence & Machine Learning


---

**Disclaimer:** Forecasts and anomaly flags are analytical estimates, not guarantees of future business performance. Model accuracy depends on data quality and the evaluation method.
