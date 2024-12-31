import os
from datetime import date

from decouple import config

def setup_environment_variables(project_name: str = "") -> None:
    if not project_name:
        project_name = f"Test_{date.today()}"
    
    os.environ['OPENAI_API_KEY'] = str(config("OPENAI_API_KEY"))

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = str(config("LANGCHAIN_API_KEY"))
    os.environ["LANGCHAIN_PROJECT"] = project_name

    os.environ["TAVI_API_KEY"] = str(config("TAVI_API_KEY"))

    print("API Keys loaded and tracing set with project name:", project_name)

