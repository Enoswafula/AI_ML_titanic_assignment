import pandas as pd

def clean_data(file_path):
    df = pd.read_csv(file_path)

    # Handle missing values
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

    # Drop Cabin because of too many missing values
    df.drop("Cabin", axis=1, inplace=True)

    return df
