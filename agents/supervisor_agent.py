class SupervisorAgent:

    def get_system_status(self):
        """
        Returns active AI agents in system.
        """

        agents = [
            "Data Cleaning Agent",
            "Data Analyst Agent",
            "ML Engineer Agent",
            "Report Writer Agent",
            "Automation Agent"
        ]

        return agents

    def get_project_goal(self):
        """
        Returns system mission.
        """

        return """
The AI Autonomous Data Science Company is a multi-agent AI platform
that automatically cleans datasets, analyzes data, trains machine learning models,
creates reports, and prepares automation workflows.
"""