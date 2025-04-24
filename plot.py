import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Sample data
data = {
    'Region': ['North', 'South', 'East', 'West'] * 5,
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'] * 4,
    'Sales': [1500, 2000, 1800, 2200, 2100,
              1400, 1600, 1700, 1900, 2000,
              1300, 1250, 1350, 1450, 1550,
              1800, 1700, 1600, 1900, 2100]
}
df = pd.DataFrame(data)

# Sidebar filter
region = st.sidebar.selectbox("Select Region", df["Region"].unique())

# Filtered data
filtered_df = df[df["Region"] == region]

# Header
st.title("📊 Sales Dashboard")
st.write(f"### Sales data for **{region}**")

# Line chart
fig, ax = plt.subplots()
monthly_sales = filtered_df.groupby("Month")["Sales"].sum()
monthly_sales.plot(kind="bar", ax=ax, color="skyblue")
ax.set_ylabel("Sales")
st.pyplot(fig)

# Metrics
st.write("### Summary")
st.metric("Total Sales", f"${filtered_df['Sales'].sum():,.0f}")
st.metric("Average Monthly Sales", f"${filtered_df['Sales'].mean():,.0f}")