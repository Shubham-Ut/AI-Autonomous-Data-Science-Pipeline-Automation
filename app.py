import os
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import requests

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score
from agents.data_cleaning_agent import DataCleaningAgent
from agents.data_analyst_agent import DataAnalystAgent
from agents.ml_engineer_agent import MLEngineerAgent
from agents.report_writer_agent import ReportWriterAgent
from agents.automation_agent import AutomationAgent
from agents.supervisor_agent import SupervisorAgent

cleaning_agent = DataCleaningAgent()
analyst_agent = DataAnalystAgent()
ml_agent = MLEngineerAgent()
report_agent = ReportWriterAgent()
automation_agent = AutomationAgent()
supervisor_agent = SupervisorAgent()

# -------------------------------
# App Config
# -------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")
st.set_page_config(
    page_title="AI Autonomous Data Science Company",
    page_icon="🤖",
    layout="wide"
)

# -------------------------------
# Custom CSS
# -------------------------------
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #00D4FF;
}
.section-card {
    background-color: #1E1E1E;
    padding: 18px;
    border-radius: 14px;
    margin-bottom: 15px;
    border: 1px solid #333;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🤖 AI Autonomous Data Science Company</div>', unsafe_allow_html=True)
st.write("An AI-powered platform for dataset analysis, cleaning, dashboards, ML training, and automated reports.")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("Navigation")
st.info(supervisor_agent.get_project_goal())
selected_section = st.sidebar.radio(
    "Go to",
    [
        "Dataset Overview",
        "Data Cleaning",
        "AI Suggestion",
        "Dashboard",
        "ML Training",
        "Run All Agents",
        "Supervisor"

    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Project Status")
st.sidebar.success("✅ Phase 1: CSV Upload")
st.sidebar.success("✅ Phase 2: Data Cleaning")
st.sidebar.success("✅ Phase 3: Dashboard")
st.sidebar.success("✅ Phase 4: ML Training")
st.sidebar.info("🔄 Phase 5: Automation Reports")

st.sidebar.markdown("---")
st.sidebar.subheader("Active AI Agents")

st.sidebar.success("🧹 Data Cleaning Agent")
st.sidebar.success("📊 Data Analyst Agent")
st.sidebar.success("🤖 ML Engineer Agent")
st.sidebar.success("📝 Report Writer Agent")
st.sidebar.success("⚙️ Automation Agent")

st.sidebar.markdown("---")
st.sidebar.subheader("Supervisor Agent")

system_agents = supervisor_agent.get_system_status()

for agent in system_agents:
    st.sidebar.info(f"✅ {agent}")
# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "cleaned_df" not in st.session_state:
        st.session_state["cleaned_df"] = df.copy()

    active_df = st.session_state["cleaned_df"]

    st.success("Dataset uploaded successfully!")

    # -------------------------------
    # Dataset Overview
    # -------------------------------
    if selected_section == "Dataset Overview":
        st.subheader("📌 Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Rows", df.shape[0])

        with col2:
            st.metric("Columns", df.shape[1])

        with col3:
            st.metric("Missing Values", int(df.isnull().sum().sum()))

        with col4:
            st.metric("Duplicate Rows", int(df.duplicated().sum()))

        st.write("### Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        st.write("### Columns and Data Types")

        column_info = pd.DataFrame({
            "Column Name": df.columns,
            "Data Type": df.dtypes.astype(str).values
        })

        st.dataframe(column_info, use_container_width=True)

    # -------------------------------
    # Data Cleaning
    # -------------------------------
    elif selected_section == "Data Cleaning":
        st.subheader("🧹 Data Cleaning")

        analysis_report = cleaning_agent.analyze_dataset(df)

        missing_values = df.isnull().sum()
        duplicate_count = analysis_report["duplicate_rows"]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Missing Values", int(missing_values.sum()))

        with col2:
            st.metric("Duplicate Rows", int(duplicate_count))

        with col3:
            st.metric("Columns", df.shape[1])

        st.write("### Missing Values by Column")

        missing_table = missing_values.reset_index()
        missing_table.columns = ["Column Name", "Missing Values"]
        st.dataframe(missing_table, use_container_width=True)

        numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

        st.write("### Numeric Columns")
        st.write(numeric_columns)

        st.write("### Categorical Columns")
        st.write(categorical_columns)

        st.write("### Cleaning Suggestions")

        if missing_values.sum() == 0 and duplicate_count == 0:
            st.success("Dataset looks clean. No missing values or duplicate rows found.")
        else:
            if missing_values.sum() > 0:
                st.warning("Missing values found. Numeric values will be filled with mean and text values with mode.")

            if duplicate_count > 0:
                st.warning("Duplicate rows found. They will be removed.")

        if st.button("Clean Dataset"):
            before_rows = df.shape[0]
            before_missing = df.isnull().sum().sum()
            before_duplicates = df.duplicated().sum()

            cleaned_df = cleaning_agent.clean_dataset(df)
            after_rows = cleaned_df.shape[0]
            after_missing = cleaned_df.isnull().sum().sum()
            after_duplicates = cleaned_df.duplicated().sum()

            st.session_state["cleaned_df"] = cleaned_df

            st.success("Dataset cleaned successfully!")

            comparison = pd.DataFrame({
                "Metric": ["Rows", "Missing Values", "Duplicate Rows"],
                "Before Cleaning": [before_rows, before_missing, before_duplicates],
                "After Cleaning": [after_rows, after_missing, after_duplicates]
            })

            st.write("### Before vs After Cleaning")
            st.dataframe(comparison, use_container_width=True)

            st.write("### Cleaned Dataset Preview")
            st.dataframe(cleaned_df.head(), use_container_width=True)

    # -------------------------------
    # AI Suggestion
    # -------------------------------
    elif selected_section == "AI Suggestion":
        st.subheader("🧠 AI ML Model Suggestion")

        column_info = pd.DataFrame({
            "Column Name": active_df.columns,
            "Data Type": active_df.dtypes.astype(str).values
        })

        st.write("### Dataset Column Information")
        st.dataframe(column_info, use_container_width=True)

        dataset_summary = f"""
Dataset has {active_df.shape[0]} rows and {active_df.shape[1]} columns.

Columns:
{', '.join(active_df.columns)}

Data Types:
{active_df.dtypes.to_string()}
"""

        if st.button("Ask AI for Model Suggestion"):
            if not GEMINI_API_KEY:
                st.error("Gemini API key not found. Please add it in your .env file.")
            else:
                try:
                    with st.spinner("Gemini is analyzing your dataset..."):
                        client = genai.Client(api_key=GEMINI_API_KEY)

                        prompt = f"""
You are a senior data scientist.

Analyze this dataset summary and suggest:
1. What real-world problem this dataset can solve
2. Which column could be the target variable
3. Whether it is classification, regression, or clustering
4. Which ML models can be used
5. Which one best model should be used
6. What should be done next

Keep the answer simple for a beginner.

Dataset summary:
{dataset_summary}
"""

                        response = client.models.generate_content(
                            model="gemini-2.5-flash-lite",
                            contents=prompt
                        )

                    st.success("AI suggestion generated!")
                    st.markdown(response.text)

                except Exception as e:
                    st.warning("Gemini quota/server issue. Showing offline suggestion.")

                    st.markdown("""
### Offline ML Suggestion

- If target column has categories like Yes/No, Churn/Not Churn, Pass/Fail → Classification
- If target column has numbers like Price, Salary, Sales, Marks → Regression
- If there is no target column → Clustering

Suggested models:
- Classification: Random Forest Classifier
- Regression: Random Forest Regressor
- Clustering: K-Means
""")

                    st.code(str(e))

    # -------------------------------
    # Dashboard
    # -------------------------------
    elif selected_section == "Dashboard":
        st.subheader("📊 Interactive Auto Dashboard")

        dashboard_df = active_df

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Rows", dashboard_df.shape[0])

        with col2:
            st.metric("Columns", dashboard_df.shape[1])

        with col3:
            st.metric("Missing Values", int(dashboard_df.isnull().sum().sum()))

        with col4:
            st.metric("Duplicate Rows", int(dashboard_df.duplicated().sum()))

        numeric_cols = dashboard_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = dashboard_df.select_dtypes(include=["object"]).columns.tolist()

        st.write("### Correlation Heatmap")

        numeric_df = dashboard_df.select_dtypes(include=["int64", "float64"])

        if numeric_df.shape[1] >= 2:
            corr = numeric_df.corr()

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)

            st.pyplot(fig)
        else:
            st.info("Need at least 2 numeric columns to show correlation heatmap.")

        numeric_summary = analyst_agent.get_numeric_summary(dashboard_df)

        if numeric_summary is not None:
            st.write("### Numeric Summary")
            st.dataframe(numeric_summary, use_container_width=True)
        else:
            st.info("No numeric columns found.")

        # Histogram
        if len(numeric_cols) >= 1:

            selected_numeric = st.selectbox(
                "Choose numeric column for distribution chart",
                numeric_cols
            )

            fig_hist = px.histogram(
                dashboard_df,
                x=selected_numeric,
                title=f"Distribution of {selected_numeric}"
            )

            st.plotly_chart(fig_hist, use_container_width=True)

        if len(categorical_cols) >= 1:
            selected_category = st.selectbox(
                "Choose categorical column for top categories",
                categorical_cols
            )

            category_count = dashboard_df[selected_category].value_counts().head(10).reset_index()
            category_count.columns = [selected_category, "Count"]

            fig_bar = px.bar(
                category_count,
                x=selected_category,
                y="Count",
                title=f"Top 10 Categories in {selected_category}"
            )

            st.plotly_chart(fig_bar, use_container_width=True)

        if len(numeric_cols) >= 2:
            st.write("### Relationship Between Two Numeric Columns")

            x_col = st.selectbox("Choose X-axis column", numeric_cols)
            y_col = st.selectbox("Choose Y-axis column", numeric_cols, index=1)

            fig_scatter = px.scatter(
                dashboard_df,
                x=x_col,
                y=y_col,
                title=f"{x_col} vs {y_col}"
            )

            st.plotly_chart(fig_scatter, use_container_width=True)

    # -------------------------------
    # ML Training
    # -------------------------------
    elif selected_section == "ML Training":
        st.subheader("🤖 Basic ML Model Training")

        ml_df = active_df.copy()

        st.write("### Recommended Target Columns")

        numeric_targets = ml_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_targets = ml_df.select_dtypes(include=["object"]).columns.tolist()

        if numeric_targets:
            st.success("Good regression targets:")
            st.write(numeric_targets)

        if categorical_targets:
            st.info("Good classification targets:")
            st.write(categorical_targets)

        target_column = st.selectbox(
            "Select target column for prediction",
            ml_df.columns
        )

        test_size = st.slider(
            "Select test data size",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.1
        )

        if st.button("Train Basic ML Model"):
            try:
                ml_result = ml_agent.train_models(
                    df=ml_df,
                    target_column=target_column,
                    test_size=test_size
                )

                problem_type = ml_result["problem_type"]
                results_df = ml_result["results"]
                best_model = ml_result["best_model"]
                best_score = ml_result["best_score"]
                best_predictions = ml_result["best_predictions"]
                best_model_name = ml_result["best_model_name"]
                X = ml_result["X"]
                y_test = ml_result["y_test"]

                st.success(f"Model comparison completed! Problem Type: {problem_type}")

                st.write("### Model Comparison")
                st.dataframe(results_df, use_container_width=True)

                st.write(f"### Best Model: {best_model_name}")
                st.metric("Best Score", round(best_score, 3))

                st.write("### Feature Importance")

                if hasattr(best_model, "feature_importances_"):
                    importance_df = pd.DataFrame({
                        "Feature": X.columns,
                        "Importance": best_model.feature_importances_
                    })

                    importance_df = importance_df.sort_values(
                        by="Importance",
                        ascending=False
                    )

                    st.dataframe(importance_df, use_container_width=True)
                    st.bar_chart(importance_df.set_index("Feature")["Importance"])
                else:
                    st.info("Feature importance is not available for this model.")

                st.write("### Prediction Results")

                prediction_results_df = pd.DataFrame({
                    "Actual Value": y_test,
                    "Predicted Value": best_predictions
                })

                st.dataframe(prediction_results_df.head(20), use_container_width=True)

                if problem_type == "Regression":
                    st.write("### Actual vs Predicted Chart")

                    chart_df = pd.DataFrame({
                        "Actual": y_test,
                        "Predicted": best_predictions
                    })

                    st.line_chart(chart_df)

                st.write("### Download ML Report")

                csv_report = prediction_results_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download Prediction Report as CSV",
                    data=csv_report,
                    file_name="ml_prediction_report.csv",
                    mime="text/csv"
                )

                st.write("### Auto Generated ML Summary Report")

                summary_report = report_agent.generate_ml_report(
                target_column=target_column,
                problem_type=problem_type,
                best_model_name=best_model_name,
                best_score=best_score,
                test_size=test_size,
                rows_used=len(X),
                columns_used=len(X.columns)
            )

                st.text_area("Generated Report", summary_report, height=300)

                st.download_button(
                    label="Download Summary Report as TXT",
                    data=summary_report,
                    file_name="ml_summary_report.txt",
                    mime="text/plain"
                )
                st.write("### Save Report for Automation")

                if st.button("Save Report for n8n"):
                    try:
                        file_name = automation_agent.save_report(summary_report)

                        st.success("Report saved successfully!")
                        st.code(file_name)

                    except Exception as e:
                        st.error("Automation Agent failed.")
                        st.write(e)

            except Exception as e:
                st.error("ML Engineer Agent failed.")
                st.write(e)
        # -------------------------------
    # Run All Agents
    # -------------------------------
    elif selected_section == "Run All Agents":
        st.subheader("🚀 Run All Agents")

        st.write("This will run the complete autonomous workflow.")

        target_column = st.selectbox(
            "Select target column for full automation",
            active_df.columns
        )

        test_size = st.slider(
            "Select test data size",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.1
        )

        if st.button("Run Complete Agent Workflow"):
            workflow_logs = []
            progress_bar = st.progress(0)
            try:
                st.info("Step 1: Data Cleaning Agent started...")
                cleaned_df = cleaning_agent.clean_dataset(active_df)
                st.success("Data Cleaning Agent completed.")
                workflow_logs.append("✅ Data Cleaning Agent completed")


                st.info("Step 2: Data Analyst Agent started...")
                numeric_summary = analyst_agent.get_numeric_summary(cleaned_df)
                st.success("Data Analyst Agent completed.")
                workflow_logs.append("✅ Data Analyst Agent completed")


                if numeric_summary is not None:
                    st.write("### Numeric Summary")
                    st.dataframe(numeric_summary, use_container_width=True)

                st.info("Step 3: ML Engineer Agent started...")
                ml_result = ml_agent.train_models(
                    df=cleaned_df,
                    target_column=target_column,
                    test_size=test_size
                )
                st.success("ML Engineer Agent completed.")

                problem_type = ml_result["problem_type"]
                results_df = ml_result["results"]
                best_score = ml_result["best_score"]
                best_model_name = ml_result["best_model_name"]
                X = ml_result["X"]

                st.write("### Model Results")
                st.dataframe(results_df, use_container_width=True)

                st.metric("Best Score", round(best_score, 3))
                st.write(f"Best Model: {best_model_name}")
                workflow_logs.append("✅ ML Engineer Agent completed")


                st.info("Step 4: Report Writer Agent started...")
                summary_report = report_agent.generate_ml_report(
                    target_column=target_column,
                    problem_type=problem_type,
                    best_model_name=best_model_name,
                    best_score=best_score,
                    test_size=test_size,
                    rows_used=len(X),
                    columns_used=len(X.columns)
                )
                st.success("Report Writer Agent completed.")

                st.text_area("Generated Report", summary_report, height=300)
                st.download_button(
                        label="Download Full Agent Report",
                        data=summary_report,
                        file_name="full_agent_report.txt",
                        mime="text/plain"
                    )
                workflow_logs.append("✅ Report Writer Agent completed")


                st.info("Step 5: Automation Agent started...")
                file_name = automation_agent.save_report(summary_report)
                st.success("Automation Agent completed.")
                st.code(file_name)
                workflow_logs.append("✅ Automation Agent completed")
                st.write("### Agent Workflow Logs")

                for log in workflow_logs:
                    st.success(log)

                st.success("🎉 Complete autonomous workflow finished successfully!")
                st.write("### Send Report to n8n Email Automation")

                if st.button("Send Report to n8n"):

                    if not N8N_WEBHOOK_URL:
                        st.error("N8N_WEBHOOK_URL not found in .env file.")

                    else:
                        try:
                            payload = {
                                "report": summary_report,
                                "project": "AI Autonomous Data Science Company"
                            }

                            response = requests.post(
                                N8N_WEBHOOK_URL,
                                json=payload
                            )

                            if response.status_code in [200, 201]:
                                st.success("Report sent to n8n successfully!")

                            else:
                                st.error("n8n returned an error.")
                                st.write(response.text)

                        except Exception as e:
                            st.error("Failed to send report to n8n.")
                            st.write(e)

            except Exception as e:
                st.error("Run All Agents failed.")
                st.write(e)

    # -------------------------------
    # Supervisor
    # -------------------------------
    elif selected_section == "Supervisor":
        st.subheader("🧠 Supervisor Agent Dashboard")

        st.write("### Project Mission")
        st.info(supervisor_agent.get_project_goal())

        st.write("### Active Agents")

        system_agents = supervisor_agent.get_system_status()

        for agent in system_agents:
            st.success(f"✅ {agent}")

        st.write("### Multi-Agent Workflow")

        st.code("""
    User Uploads Dataset
            ↓
    Supervisor Agent
            ↓
    Data Cleaning Agent
            ↓
    Data Analyst Agent
            ↓
    ML Engineer Agent
            ↓
    Report Writer Agent
            ↓
    Automation Agent
            ↓
    Final Report + Automation File
    """)
else:
    st.info("Please upload a CSV file to start.")