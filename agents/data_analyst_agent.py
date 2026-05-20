import pandas as pd


class DataAnalystAgent:

    def get_basic_summary(self, df):
        """
        Creates basic dataset summary.
        """

        summary = {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "numeric_columns": df.select_dtypes(include=["int64", "float64"]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=["object"]).columns.tolist()
        }

        return summary

    def get_numeric_summary(self, df):
        """
        Gives statistical summary for numeric columns.
        """

        numeric_df = df.select_dtypes(include=["int64", "float64"])

        if numeric_df.empty:
            return None

        return numeric_df.describe()

    def get_categorical_summary(self, df):
        """
        Gives top values for categorical columns.
        """

        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

        result = {}

        for col in categorical_cols:
            result[col] = df[col].value_counts().head(5)

        return result