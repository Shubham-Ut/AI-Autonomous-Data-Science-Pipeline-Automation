import pandas as pd


class DataCleaningAgent:

    def analyze_dataset(self, df):
        """
        Analyze dataset health
        """

        report = {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "missing_values": int(df.isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum())
        }

        return report

    def clean_dataset(self, df):
        """
        Automatically clean dataset
        """

        cleaned_df = df.copy()

        # Remove duplicates
        cleaned_df = cleaned_df.drop_duplicates()

        # Fill missing values
        for column in cleaned_df.columns:

            # Numeric columns
            if cleaned_df[column].dtype in ["int64", "float64"]:
                cleaned_df[column] = cleaned_df[column].fillna(
                    cleaned_df[column].mean()
                )

            # Text columns
            else:
                if not cleaned_df[column].mode().empty:
                    cleaned_df[column] = cleaned_df[column].fillna(
                        cleaned_df[column].mode()[0]
                    )

        return cleaned_df