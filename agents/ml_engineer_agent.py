import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score


class MLEngineerAgent:

    def train_models(self, df, target_column, test_size=0.2):
        """
        Automatically trains classification or regression models.
        """

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Convert input text columns to numbers
        for col in X.columns:
            if X[col].dtype == "object" or X[col].dtype.name == "category":
                encoder = LabelEncoder()
                X[col] = encoder.fit_transform(X[col].astype(str))

        # Convert boolean columns to 0/1
        for col in X.columns:
            if X[col].dtype == "bool":
                X[col] = X[col].astype(int)

        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.fillna(0)

        # Detect problem type
        if y.dtype == "object" or y.dtype.name == "category":
            problem_type = "Classification"

            y_encoder = LabelEncoder()
            y = y_encoder.fit_transform(y.astype(str))

            models = {
                "Random Forest Classifier": RandomForestClassifier(random_state=42),
                "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=1000)
            }

        else:
            problem_type = "Regression"

            y = pd.to_numeric(y, errors="coerce")

            valid_rows = y.notnull()
            X = X[valid_rows]
            y = y[valid_rows]

            if len(y) < 5:
                raise ValueError("Not enough valid rows for ML training.")

            models = {
                "Random Forest Regressor": RandomForestRegressor(random_state=42),
                "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
                "Linear Regression": LinearRegression()
            }

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42
        )

        results = []
        best_model = None
        best_score = -999
        best_predictions = None
        best_model_name = ""

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            if problem_type == "Classification":
                score = accuracy_score(y_test, predictions)
            else:
                score = r2_score(y_test, predictions)

            results.append({
                "Model": model_name,
                "Score": round(score, 3)
            })

            if score > best_score:
                best_score = score
                best_model = model
                best_predictions = predictions
                best_model_name = model_name

        return {
            "problem_type": problem_type,
            "results": pd.DataFrame(results),
            "best_model": best_model,
            "best_score": best_score,
            "best_predictions": best_predictions,
            "best_model_name": best_model_name,
            "X": X,
            "y_test": y_test
        }