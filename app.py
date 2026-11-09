import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Retail Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load dataset
# LOAD AND CLEAN DATASET

raw_df = pd.read_csv("Retail_Sales_Data_Unlox (1).csv")

df = raw_df.copy()

# Initial dataset information
original_rows = len(df)
original_columns = len(df.columns)

# Remove duplicate records
duplicate_count = df.duplicated().sum()
df = df.drop_duplicates()

# Remove extra spaces from text columns
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()
    df[col] = df[col].replace("", pd.NA)

# Convert date column
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Convert numerical columns
numeric_columns = [
    "Unit_Price",
    "Units_Sold",
    "Total_Sales",
    "Discount_Percentage"
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove records with invalid dates
df = df.dropna(subset=["Date"])

# Fill missing numerical values
for col in numeric_columns:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Fill missing categorical values
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].fillna("Unknown")

# Data cleaning results
cleaned_rows = len(df)
removed_rows = original_rows - cleaned_rows

st.title("📊 Retail Analytics Dashboard")
# DATA CLEANING REPORT

st.header("🧹 Data Cleaning & Preprocessing")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Original Records", f"{original_rows:,}")

col2.metric("Cleaned Records", f"{cleaned_rows:,}")

col3.metric("Duplicates Found", f"{duplicate_count:,}")

col4.metric("Records Removed", f"{removed_rows:,}")

with st.expander("View Data Quality Report"):

    quality_report = pd.DataFrame({
        "Column": raw_df.columns,
        "Missing Values": raw_df.isnull().sum().values,
        "Data Type": raw_df.dtypes.astype(str).values
    })

    st.dataframe(quality_report, use_container_width=True)

st.success("Data Cleaning and Preprocessing Completed!")
st.markdown("### AI-Powered Sales Forecasting System")

# Sidebar
st.sidebar.header("Dashboard Filters")

locations = ["All"] + sorted(df["Store_Location"].dropna().unique())
selected_location = st.sidebar.selectbox("Select Location", locations)

categories = ["All"] + sorted(df["Product_Category"].dropna().unique())
selected_category = st.sidebar.selectbox("Select Category", categories)

filtered_df = df.copy()

if selected_location != "All":
    filtered_df = filtered_df[
        filtered_df["Store_Location"] == selected_location
    ]

if selected_category != "All":
    filtered_df = filtered_df[
        filtered_df["Product_Category"] == selected_category
    ]

# KPI Metrics
st.subheader("Business Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Sales", f"₹{filtered_df['Total_Sales'].sum():,.2f}")

with col2:
    st.metric("Units Sold", f"{filtered_df['Units_Sold'].sum():,}")

with col3:
    st.metric("Total Orders", f"{len(filtered_df):,}")

with col4:
    st.metric("Products", filtered_df["Product_ID"].nunique())

st.divider()

# Sales by Category
st.subheader("Sales by Product Category")

category_sales = (
    filtered_df.groupby("Product_Category")["Total_Sales"]
    .sum()
    .reset_index()
)

fig1 = px.bar(
    category_sales,
    x="Product_Category",
    y="Total_Sales",
    color="Product_Category",
    title="Category-wise Sales"
)

st.plotly_chart(fig1, use_container_width=True)
# STORE AND LOCATION ANALYSIS

st.divider()

st.header("🏬 Store & Location Analysis")

# Store-wise Sales
st.subheader("Store-wise Sales Performance")

store_sales = (
    filtered_df.groupby("Store_ID")["Total_Sales"]
    .sum()
    .reset_index()
    .sort_values("Total_Sales", ascending=False)
)

fig_store = px.bar(
    store_sales,
    x="Store_ID",
    y="Total_Sales",
    color="Store_ID",
    title="Sales Performance by Store"
)

st.plotly_chart(fig_store, use_container_width=True)

# City-wise Sales
st.subheader("City-wise Sales Performance")

city_sales = (
    filtered_df.groupby("Store_Location")["Total_Sales"]
    .sum()
    .reset_index()
    .sort_values("Total_Sales", ascending=False)
)

fig_city = px.pie(
    city_sales,
    names="Store_Location",
    values="Total_Sales",
    title="Sales Distribution by City",
    hole=0.4
)

st.plotly_chart(fig_city, use_container_width=True)

# Top Performing Stores
st.subheader("🏆 Top 5 Performing Stores")

st.dataframe(
    store_sales.head(5),
    use_container_width=True
)
# BRAND PERFORMANCE ANALYSIS

st.divider()

st.header("🏷️ Brand Performance Analysis")

# Brand-wise Sales
brand_sales = (
    filtered_df.groupby("Brand")
    .agg(
        Total_Sales=("Total_Sales", "sum"),
        Units_Sold=("Units_Sold", "sum")
    )
    .reset_index()
    .sort_values("Total_Sales", ascending=False)
)

# Top 10 Brands
st.subheader("Top 10 Performing Brands")

top_brands = brand_sales.head(10)

fig_brand = px.bar(
    top_brands,
    x="Brand",
    y="Total_Sales",
    color="Brand",
    title="Top 10 Brands by Sales"
)

st.plotly_chart(fig_brand, use_container_width=True)

# Brand Sales Table
st.subheader("Brand-wise Sales Details")

st.dataframe(
    brand_sales,
    use_container_width=True
)

# Best Brand
best_brand = brand_sales.iloc[0]

st.success(
    f"Best Performing Brand: {best_brand['Brand']}"
)
# CUSTOMER BEHAVIOUR ANALYSIS

st.divider()

st.header("👥 Customer Behaviour Analysis")

# Customer type analysis
customer_analysis = (
    filtered_df.groupby("Customer_Type")
    .agg(
        Total_Sales=("Total_Sales", "sum"),
        Total_Orders=("Customer_Type", "count"),
        Units_Purchased=("Units_Sold", "sum")
    )
    .reset_index()
)

col1, col2 = st.columns(2)

with col1:
    fig_customer = px.pie(
        customer_analysis,
        names="Customer_Type",
        values="Total_Orders",
        title="New vs Returning Customers",
        hole=0.4
    )
    st.plotly_chart(fig_customer, use_container_width=True)

with col2:
    fig_customer_sales = px.bar(
        customer_analysis,
        x="Customer_Type",
        y="Total_Sales",
        color="Customer_Type",
        title="Sales by Customer Type"
    )
    st.plotly_chart(fig_customer_sales, use_container_width=True)

st.subheader("Customer Analysis Details")

st.dataframe(
    customer_analysis,
    use_container_width=True
)
# PAYMENT METHOD ANALYSIS

st.divider()

st.header("💳 Payment Method Analysis")

# Find actual payment column automatically
payment_col = next(
    (col for col in filtered_df.columns
     if "payment" in col.lower()),
    None
)

if payment_col:

    payment_analysis = (
        filtered_df.groupby(payment_col)
        .agg(
            Total_Sales=("Total_Sales", "sum"),
            Total_Transactions=("Total_Sales", "count")
        )
        .reset_index()
        .sort_values("Total_Sales", ascending=False)
    )

    fig_payment = px.bar(
        payment_analysis,
        x=payment_col,
        y="Total_Sales",
        color=payment_col,
        title="Sales by Payment Method"
    )

    st.plotly_chart(
        fig_payment,
        use_container_width=True
    )

    st.dataframe(
        payment_analysis,
        use_container_width=True
    )

else:
    st.warning("Payment information is not available in this dataset.")
    st.write("Available columns:", filtered_df.columns.tolist())

# DISCOUNT ANALYSIS

st.divider()

st.header("🏷️ Discount Analysis")

# Discount-wise Sales
discount_analysis = (
    filtered_df.groupby("Discount_Percentage")
    .agg(
        Total_Sales=("Total_Sales", "sum"),
        Units_Sold=("Units_Sold", "sum")
    )
    .reset_index()
    .sort_values("Discount_Percentage")
)

st.subheader("Sales by Discount Percentage")

fig_discount = px.bar(
    discount_analysis,
    x="Discount_Percentage",
    y="Total_Sales",
    title="Discount Percentage vs Total Sales",
    color="Total_Sales"
)

st.plotly_chart(fig_discount, use_container_width=True)

# Discount summary
col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Average Discount",
        f"{filtered_df['Discount_Percentage'].mean():.2f}%"
    )

with col2:
    st.metric(
        "Maximum Discount",
        f"{filtered_df['Discount_Percentage'].max():.2f}%"
    )

st.dataframe(discount_analysis, use_container_width=True)

# PRODUCT PERFORMANCE ANALYSIS

st.divider()

st.header("📦 Product Performance Analysis")

# Product-wise analysis
product_analysis = (
    filtered_df.groupby("Product_ID")
    .agg(
        Total_Sales=("Total_Sales", "sum"),
        Units_Sold=("Units_Sold", "sum")
    )
    .reset_index()
    .sort_values("Total_Sales", ascending=False)
)

# Top 10 products
st.subheader("🏆 Top 10 Best-Selling Products")

top_products = product_analysis.head(10)

fig_products = px.bar(
    top_products,
    x="Product_ID",
    y="Total_Sales",
    color="Product_ID",
    title="Top 10 Products by Revenue"
)

st.plotly_chart(fig_products, use_container_width=True)

# Subcategory analysis
st.subheader("Product Subcategory Analysis")

subcategory_sales = (
    filtered_df.groupby("Product_Subcategory")
    ["Total_Sales"].sum()
    .reset_index()
    .sort_values("Total_Sales", ascending=False)
)

fig_subcategory = px.pie(
    subcategory_sales,
    names="Product_Subcategory",
    values="Total_Sales",
    title="Sales by Product Subcategory",
    hole=0.4
)

st.plotly_chart(fig_subcategory, use_container_width=True)

st.dataframe(
    product_analysis,
    use_container_width=True
)
# Monthly Sales Trend
st.subheader("Monthly Sales Trend")

monthly_sales = (
    filtered_df.set_index("Date")
    .resample("ME")["Total_Sales"]
    .sum()
    .reset_index()
)

fig2 = px.line(
    monthly_sales,
    x="Date",
    y="Total_Sales",
    markers=True,
    title="Monthly Revenue Trend"
)

st.plotly_chart(fig2, use_container_width=True)
# PROMOTION ANALYSIS

st.divider()
st.header("🎯 Promotion Analysis")

promotion_analysis = (
    filtered_df.groupby("Promotion_Applied")
    .agg(
        Total_Sales=("Total_Sales", "sum"),
        Units_Sold=("Units_Sold", "sum"),
        Total_Transactions=("Total_Sales", "count")
    )
    .reset_index()
)

st.subheader("Sales with and without Promotions")

fig_promotion = px.bar(
    promotion_analysis,
    x="Promotion_Applied",
    y="Total_Sales",
    color="Promotion_Applied",
    title="Promotion Impact on Sales",
    labels={
        "Promotion_Applied": "Promotion Applied",
        "Total_Sales": "Total Sales"
    }
)

st.plotly_chart(fig_promotion, use_container_width=True)

st.subheader("Promotion Performance Details")

st.dataframe(
    promotion_analysis,
    use_container_width=True
)
# INVENTORY ANALYSIS

st.divider()

st.header("📦 Inventory Analysis")

# Stock summary
col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Stock Available",
    f"{filtered_df['Stock_On_Hand'].sum():,}"
)

col2.metric(
    "Average Stock",
    f"{filtered_df['Stock_On_Hand'].mean():,.2f}"
)

col3.metric(
    "Low Stock Products",
    f"{(filtered_df['Stock_On_Hand'] < 100).sum():,}"
)

# Top 10 products by stock
st.subheader("Top 10 Products by Available Stock")

stock_analysis = (
    filtered_df.groupby("Product_ID")["Stock_On_Hand"]
    .sum()
    .reset_index()
    .sort_values("Stock_On_Hand", ascending=False)
)

fig_stock = px.bar(
    stock_analysis.head(10),
    x="Product_ID",
    y="Stock_On_Hand",
    color="Stock_On_Hand",
    title="Top 10 Products by Stock"
)

st.plotly_chart(fig_stock, use_container_width=True)

# Low stock products
st.subheader("⚠️ Low Stock Products")

low_stock = filtered_df[
    filtered_df["Stock_On_Hand"] < 100
]

st.dataframe(
    low_stock[
        ["Store_ID", "Product_ID", "Product_Category", "Stock_On_Hand"]
    ],
    use_container_width=True
)
# HOLIDAY SALES ANALYSIS

st.divider()

st.header("🎉 Holiday Sales Analysis")

holiday_analysis = (
    filtered_df.groupby("Holiday_Flag")
    .agg(
        Total_Sales=("Total_Sales", "sum"),
        Units_Sold=("Units_Sold", "sum"),
        Total_Transactions=("Total_Sales", "count")
    )
    .reset_index()
)

holiday_analysis["Day_Type"] = holiday_analysis["Holiday_Flag"].map({
    0: "Normal Day",
    1: "Holiday"
})

st.subheader("Holiday vs Normal Day Sales")

fig_holiday = px.bar(
    holiday_analysis,
    x="Day_Type",
    y="Total_Sales",
    color="Day_Type",
    title="Sales Comparison: Holiday vs Normal Days"
)

st.plotly_chart(fig_holiday, use_container_width=True)

st.subheader("Holiday Sales Details")

st.dataframe(
    holiday_analysis[
        ["Day_Type", "Total_Sales", "Units_Sold", "Total_Transactions"]
    ],
    use_container_width=True
)
# STORE RATING ANALYSIS

st.divider()

st.header("⭐ Store Rating Analysis")

# Store-wise rating and sales
rating_analysis = (
    filtered_df.groupby(["Store_ID", "Store_Location"])
    .agg(
        Average_Rating=("Store_Rating", "mean"),
        Total_Sales=("Total_Sales", "sum"),
        Total_Orders=("Store_ID", "count")
    )
    .reset_index()
    .sort_values("Average_Rating", ascending=False)
)

# Highest rated store
if not rating_analysis.empty:

    best_store = rating_analysis.iloc[0]

    col1, col2 = st.columns(2)

    col1.metric(
        "Highest Rated Store",
        best_store["Store_ID"]
    )

    col2.metric(
        "Highest Store Rating",
        f"{best_store['Average_Rating']:.2f} ⭐"
    )

# Store rating chart
st.subheader("Store-wise Average Ratings")

fig_rating = px.bar(
    rating_analysis,
    x="Store_ID",
    y="Average_Rating",
    color="Average_Rating",
    title="Average Rating by Store",
    hover_data=["Store_Location", "Total_Sales"]
)

st.plotly_chart(fig_rating, use_container_width=True)

# Store rating details
st.subheader("Store Performance Details")

st.dataframe(
    rating_analysis,
    use_container_width=True
)
# AI SALES FORECASTING

st.divider()

st.header("🤖 AI-Based Sales Forecasting")

forecast_data = (
    filtered_df.set_index("Date")
    .resample("ME")["Total_Sales"]
    .sum()
    .reset_index()
)

if len(forecast_data) >= 6:
    st.write("Total Monthly Records:", len(forecast_data))
    # Trend and Seasonality Features

    forecast_data["Month_Number"] = range(len(forecast_data))

    forecast_data["Month"] = forecast_data["Date"].dt.month

    forecast_data["Month_Sin"] = np.sin(
    2 * np.pi * forecast_data["Month"] / 12
 )

    forecast_data["Month_Cos"] = np.cos(
    2 * np.pi * forecast_data["Month"] / 12
 )

    X = forecast_data[
    ["Month_Number", "Month_Sin", "Month_Cos"]
 ]

    y = forecast_data["Total_Sales"] 

    # Split data chronologically
    split = int(len(forecast_data) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    # Train Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate Model
    test_predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, test_predictions)
    rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    r2 = r2_score(y_test, test_predictions)

    st.subheader("📊 Model Performance Evaluation")

    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", f"{mae:,.2f}")
    col2.metric("RMSE", f"{rmse:,.2f}")
    col3.metric("R² Score", f"{r2:.4f}")

    # Actual vs Predicted
    comparison_df = pd.DataFrame({
        "Actual Sales": y_test.values,
        "Predicted Sales": test_predictions
    })

    st.subheader("Actual vs Predicted Sales")

    st.dataframe(comparison_df, use_container_width=True)

    # Train final model using full data
    model.fit(X, y)

    # Predict Next 6 Months
    # Generate future dates

    future_dates = pd.date_range(
    start=forecast_data["Date"].max() + pd.offsets.MonthEnd(1),
    periods=6,
    freq="ME"
 )

    future_months = pd.DataFrame({
    "Month_Number": range(
        len(forecast_data),
        len(forecast_data) + 6
    ),
    "Month_Sin": np.sin(
        2 * np.pi * future_dates.month / 12
    ),
    "Month_Cos": np.cos(
        2 * np.pi * future_dates.month / 12
    )
  })

    predictions = model.predict(future_months)

    future_dates = pd.date_range(
        start=forecast_data["Date"].max() + pd.offsets.MonthEnd(1),
        periods=6,
        freq="ME"
    )

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted_Sales": predictions.clip(min=0)
    })

    st.subheader("Next 6 Months Sales Prediction")

    st.dataframe(forecast_df, use_container_width=True)

    fig3 = px.line(
        forecast_df,
        x="Date",
        y="Predicted_Sales",
        markers=True,
        title="AI Sales Forecast"
    )

    st.plotly_chart(fig3, use_container_width=True)

    st.success("Sales Forecast Generated Successfully!")

else:
    st.warning("Not enough monthly data for forecasting.")