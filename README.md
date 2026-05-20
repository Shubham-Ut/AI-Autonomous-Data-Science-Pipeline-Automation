# AI-Autonomous-Data-Science-Pipeline-Automation

AI Autonomous Data Science Company

An advanced Multi-Agent AI Data Science Automation Platform built using Streamlit, Machine Learning, Gemini AI, and n8n Automation.

This project automatically:

cleans datasets
analyzes data
creates dashboards
trains machine learning models
compares models
generates reports
automates workflows using AI agents
🚀 Project Overview

The AI Autonomous Data Science Company simulates a real AI-powered autonomous data science workflow where multiple AI agents collaborate together to perform end-to-end data analysis and machine learning tasks automatically.

Users simply upload a CSV dataset, and the platform:

Cleans the data
Generates insights
Suggests ML approaches
Trains multiple models
Selects the best model
Creates downloadable reports
Sends reports through n8n automation
🧠 Multi-Agent AI Architecture
Supervisor Agent

Coordinates the complete AI workflow.

Data Cleaning Agent
Detects missing values
Removes duplicates
Cleans datasets automatically
Data Analyst Agent
Generates statistical summaries
Creates dataset insights
Helps dashboard generation
ML Engineer Agent
Detects ML problem type
Trains multiple ML models
Compares model performance
Selects the best model automatically
Report Writer Agent
Generates AI-powered ML reports
Creates downloadable summaries
Automation Agent
Saves reports automatically
Connects with n8n workflows
Enables email automation
🔄 Autonomous Workflow
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
Final Report + n8n Automation
✨ Features
📂 Dataset Upload
Upload CSV datasets
Automatic dataset preview
Column detection
🧹 Automatic Data Cleaning
Missing value handling
Duplicate removal
Numeric and categorical processing
🧠 AI Dataset Analysis
Gemini AI integration
ML model recommendations
Target column suggestions
Problem type detection
📊 Interactive Dashboard
KPI cards
Histograms
Scatter plots
Correlation heatmaps
Category distribution charts
🤖 Machine Learning Automation
Classification Models
Random Forest Classifier
Decision Tree Classifier
Logistic Regression
Regression Models
Random Forest Regressor
Decision Tree Regressor
Linear Regression
📈 Model Evaluation
Accuracy Score
R² Score
Feature Importance
Model comparison table
📄 Report Generation
Prediction reports
AI-generated summary reports
TXT and CSV downloads
⚙️ Automation Integration
n8n integration
Webhook automation
Email automation pipeline
🛠️ Tech Stack
Frontend
Streamlit
Data Processing
Pandas
NumPy
Machine Learning
Scikit-learn
Data Visualization
Plotly
Matplotlib
Seaborn
AI Integration
Gemini API
Automation
n8n
Webhooks
📁 Project Structure
AI_Autonomous_DataScience_Company/
│
├── agents/
│   ├── automation_agent.py
│   ├── data_analyst_agent.py
│   ├── data_cleaning_agent.py
│   ├── ml_engineer_agent.py
│   ├── report_writer_agent.py
│   └── supervisor_agent.py
│
├── reports/
│
├── screenshots/
│
├── .streamlit/
│
├── app.py
├── requirements.txt
├── README.md
├── .env
└── .gitignore
⚡ Installation
1. Clone Repository
git clone https://github.com/your-username/AI_Autonomous_DataScience_Company.git
2. Create Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate
3. Install Requirements
pip install -r requirements.txt
🔑 Environment Variables

Create a .env file:

GEMINI_API_KEY=your_gemini_api_key
N8N_WEBHOOK_URL=http://localhost:5678/webhook/ai-report
▶️ Run Project
streamlit run app.py
🔗 n8n Automation Setup
Workflow
Streamlit App
      ↓
Webhook Trigger
      ↓
Gmail SMTP Node
      ↓
Email Sent
Webhook URL
http://localhost:5678/webhook/ai-report
📸 Screenshots

Add screenshots here:

screenshots/dashboard.png
screenshots/ml_training.png
screenshots/automation.png
🎯 Future Scope
LangChain Integration
CrewAI Multi-Agent System
AutoML
SHAP Explainability
User Authentication
Database Integration
Cloud Deployment
Real-Time AI Monitoring
Advanced AI Memory Systems
🌟 Project Highlights

✅ Multi-Agent AI Architecture
✅ Autonomous Data Science Workflow
✅ AI + ML + Automation Integration
✅ End-to-End Pipeline
✅ Professional Dashboard
✅ Real-Time ML Training
✅ n8n Email Automation
✅ Production-Ready Architecture

👨‍💻 Author

Shubham Utekar

BTech Computer Science Engineering
DY Patil International University

📜 License

This project is developed for educational, research, and portfolio purposes.
