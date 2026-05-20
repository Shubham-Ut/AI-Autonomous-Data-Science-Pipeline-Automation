import os
from datetime import datetime


class AutomationAgent:

    def save_report(self, report_text, reports_folder="reports"):
        """
        Saves report as TXT file for automation tools like n8n.
        """

        os.makedirs(reports_folder, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        file_name = os.path.join(
            reports_folder,
            f"automation_report_{timestamp}.txt"
        )

        with open(file_name, "w", encoding="utf-8") as file:
            file.write(report_text)

        return file_name