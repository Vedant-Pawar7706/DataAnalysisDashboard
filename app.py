import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from analysis import (
    load_data,
    get_basic_info,
    missing_values,
    statistical_summary
)

st.set_page_config(
    page_title="Data Analysis Dashboard",
    layout="wide"
)

st.title("📊 Data Analysis Dashboard")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    df = load_data(uploaded_file)

    # Create processed dataset
    processed_df = df.copy()

    # Fill numerical columns
    for col in processed_df.select_dtypes(include=['int64', 'float64']).columns:
        processed_df[col] = processed_df[col].fillna(processed_df[col].mean())

    # Fill categorical columns
    for col in processed_df.select_dtypes(include='object').columns:
        processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])

    # Dataset Preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Dataset Information
    st.subheader("Dataset Information")

    info = get_basic_info(df)

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", info["Rows"])
    col2.metric("Columns", info["Columns"])
    col3.metric("Features", len(info["Column Names"]))

    st.subheader("Column Names")
    st.write(info["Column Names"])

    # Missing Values
    st.subheader("Missing Values")
    st.dataframe(missing_values(df))

    # Null Comparison
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Original Null Values",
            df.isnull().sum().sum()
        )

    with col2:
        st.metric(
            "Processed Null Values",
            processed_df.isnull().sum().sum()
        )

    # Statistical Summary
    st.subheader("Statistical Summary")
    st.dataframe(statistical_summary(df))

    numeric_cols = df.select_dtypes(
        include=['int64', 'float64']
    ).columns

    if len(numeric_cols) > 0:

        selected_col = st.selectbox(
            "Select Column",
            numeric_cols
        )

        st.subheader("Histogram")

        fig, ax = plt.subplots()
        ax.hist(df[selected_col].dropna())
        ax.set_title(selected_col)

        st.pyplot(fig)

        st.subheader("Line Chart")
        st.line_chart(df[selected_col])

        st.subheader("Bar Chart")
        st.bar_chart(df[selected_col].value_counts())

    # Download Processed Dataset
    csv = processed_df.to_csv(index=False)

    st.download_button(
        label="📥 Download Processed Dataset",
        data=csv,
        file_name="processed_dataset.csv",
        mime="text/csv"
    )