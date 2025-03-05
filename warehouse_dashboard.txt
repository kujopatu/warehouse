import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Database connection
conn = sqlite3.connect("inventory.db")
c = conn.cursor()

# Create table if not exists
c.execute('''CREATE TABLE IF NOT EXISTS inventory (
                Product_Name TEXT,
                Stock_Quantity INTEGER,
                Previous_Month_Sales INTEGER,
                Current_Month_Sales INTEGER,
                Reorder_Level INTEGER)''')
conn.commit()

# Load inventory data from database
def load_data():
    return pd.read_sql("SELECT * FROM inventory", conn)

def save_data(df):
    df.to_sql("inventory", conn, if_exists="replace", index=False)

data = load_data()

# Train a simple predictive model
if not data.empty:
    X = data[['Previous_Month_Sales']]
    y = data['Current_Month_Sales']
    model = LinearRegression()
    model.fit(X, y)

def predict_sales(previous_sales):
    return model.predict([[previous_sales]])[0] if not data.empty else 0

# Streamlit UI
st.title("Warehouse Inventory Dashboard")

# File upload option (CSV or Excel)
st.subheader("Upload Inventory Data")
uploaded_file = st.file_uploader("Upload Inventory Data (CSV or Excel)", type=["csv", "xlsx"])
if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        new_data = pd.read_csv(uploaded_file)
    else:
        new_data = pd.read_excel(uploaded_file)
    save_data(new_data)
    data = load_data()

# Stock overview
st.subheader("Current Stock Levels")
st.bar_chart(data.set_index('Product_Name')['Stock_Quantity'])

# Low-stock alerts
st.subheader("Low Stock Alerts")
low_stock = data[data['Stock_Quantity'] < data['Reorder_Level']]
if not low_stock.empty:
    st.warning("The following items need restocking:")
    st.table(low_stock[['Product_Name', 'Stock_Quantity', 'Reorder_Level']])
else:
    st.success("All items are sufficiently stocked.")

# Select product for details
product = st.selectbox("Select a Product", data['Product_Name'] if not data.empty else ["No Data"])
if not data.empty:
    stock = data[data['Product_Name'] == product]['Stock_Quantity'].values[0]
    st.write(f"Current Stock: {stock}")

    # Predict demand
    previous_sales = data[data['Product_Name'] == product]['Previous_Month_Sales'].values[0]
    predicted_sales = predict_sales(previous_sales)
    st.write(f"Predicted Next Month Sales: {predicted_sales:.2f}")
