"""
Data Loading Module for CSV Analysis
Handles CSV loading, schema extraction, and metadata generation.
"""

import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables from project root
project_root = Path(__file__).parent.parent.resolve()
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

def get_csv_metadata(file_path: str) -> Dict[str, Any]:
    """
    Extract basic metadata from a CSV file.
    
    Returns:
        Dict containing column names, types, and sample data.
    """
    df = pd.read_csv(file_path)
    
    # Basic schema info
    columns = df.columns.tolist()
    dtypes = df.dtypes.astype(str).to_dict()
    sample_data = df.head(3).to_dict(orient='records')
    shape = df.shape
    
    return {
        "columns": columns,
        "dtypes": dtypes,
        "sample_data": sample_data,
        "num_rows": shape[0],
        "num_cols": shape[1],
        "file_name": Path(file_path).name
    }

async def generate_dataset_description(metadata: Dict[str, Any], llm) -> str:
    """
    Use an LLM to generate a descriptive summary of the dataset based on metadata.
    """
    columns_list = ", ".join(metadata['columns'])
    
    prompt = f"""
    Given the following metadata for a dataset named '{metadata['file_name']}':
    
    Available Columns: {columns_list}
    
    Column Types:
    {metadata['dtypes']}
    
    Sample Data (first 3 rows):
    {metadata['sample_data']}
    
    Total Rows: {metadata['num_rows']}
    
    Please provide a concise description of what this dataset appears to represent, 
    explain what the most important columns likely signify, and suggest 3-5 key analytical 
    questions that would be most interesting to explore based on this data's domain 
    (e.g., if it's social data, suggest trend analysis; if financial, suggest volatility or correlations).
    Be specific about column names and their likely meanings. This description will be used 
    to help an AI agent analyze the data and design an insightful EDA.
    """
    
    response = await llm.acomplete(prompt)
    return str(response)
