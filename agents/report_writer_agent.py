class ReportWriterAgent:

    def generate_ml_report(
        self,
        target_column,
        problem_type,
        best_model_name,
        best_score,
        test_size,
        rows_used,
        columns_used
    ):
        """
        Generates a simple ML summary report.
        """

        report = f"""
AI Autonomous Data Science Company - ML Report

Selected Target Column:
{target_column}

Problem Type:
{problem_type}

Best Model:
{best_model_name}

Best Score:
{round(best_score, 3)}

Test Size:
{test_size}

Dataset Rows Used:
{rows_used}

Dataset Columns Used:
{columns_used}

Conclusion:
The Report Writer Agent generated this report after the ML Engineer Agent trained multiple models,
compared their performance, selected the best model, and generated predictions.
"""

        return report