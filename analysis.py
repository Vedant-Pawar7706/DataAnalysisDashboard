import pandas as pd

def load_data(file):
    df = pd.read_csv(file)
    return df

def get_basic_info(df):
    return {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Column Names": list(df.columns)
    }

def missing_values(df):
    return df.isnull().sum()

def statistical_summary(df):
    return df.describe()