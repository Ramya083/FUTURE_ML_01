import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Sales Forecasting App", layout="wide")

st.title("📊 Sales Forecasting System")
st.write("Upload Superstore sales dataset to predict future sales using Machine Learning")


uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    
    df = pd.read_csv(uploaded_file, encoding='latin1')

    st.subheader("📁 Raw Dataset Preview")
    st.dataframe(df.head())

    
    df['Order Date'] = pd.to_datetime(df['Order Date'])

    
    daily_sales = df.groupby('Order Date')['Sales'].sum().reset_index()

    st.subheader("📊 Daily Sales Data")
    st.dataframe(daily_sales.head())

    
    daily_sales['Day'] = daily_sales['Order Date'].dt.day
    daily_sales['Month'] = daily_sales['Order Date'].dt.month
    daily_sales['Year'] = daily_sales['Order Date'].dt.year
    daily_sales['Weekday'] = daily_sales['Order Date'].dt.dayofweek

    
    train = daily_sales[:-30]
    test = daily_sales[-30:]

    X_train = train[['Day','Month','Year','Weekday']]
    y_train = train['Sales']

    X_test = test[['Day','Month','Year','Weekday']]
    y_test = test['Sales']

    
    model = LinearRegression()
    model.fit(X_train, y_train)

    
    predictions = model.predict(X_test)

    
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    st.subheader("📏 Model Performance")
    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**RMSE:** {rmse:.2f}")

 
    st.subheader("📈 Actual vs Predicted Sales")

    fig1, ax1 = plt.subplots(figsize=(10,5))
    ax1.plot(train['Order Date'], train['Sales'], label="Training Data")
    ax1.plot(test['Order Date'], test['Sales'], label="Actual Sales")
    ax1.plot(test['Order Date'], predictions, linestyle="--", label="Predicted Sales")

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Sales")
    ax1.set_title("Actual vs Predicted Sales")
    ax1.legend()
    st.pyplot(fig1)

  
    st.subheader("🔮 Future Sales Forecast (Next 7 Days)")

    future_dates = pd.date_range(daily_sales['Order Date'].max() + pd.Timedelta(days=1), periods=7)

    future_df = pd.DataFrame({'Order Date': future_dates})

    future_df['Day'] = future_df['Order Date'].dt.day
    future_df['Month'] = future_df['Order Date'].dt.month
    future_df['Year'] = future_df['Order Date'].dt.year
    future_df['Weekday'] = future_df['Order Date'].dt.dayofweek

    future_predictions = model.predict(future_df[['Day','Month','Year','Weekday']])
    future_df['Forecasted Sales'] = future_predictions

    st.dataframe(future_df)

    
    st.subheader("📊 Future Predicted Sales Visualization")

    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.plot(future_df['Order Date'], future_df['Forecasted Sales'], marker='o', linestyle='--', label="Future Sales Forecast")

    ax2.set_xlabel("Date")
    ax2.set_ylabel("Forecasted Sales")
    ax2.set_title("Next 7 Days Sales Prediction")
    ax2.legend()
    st.pyplot(fig2)

  
    st.subheader("🏪 Business Interpretation")
    st.write("""
    - The forecast represents expected sales for the upcoming days based on historical trends.
    - Higher predicted values indicate higher customer demand.
    - This helps businesses plan inventory, staffing, and finances in advance.
    - Using these predictions reduces the risk of overstocking or running out of stock.
    """)

else:
    st.warning("Please upload a CSV file to continue.")
