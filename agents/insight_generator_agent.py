from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import logging
import json
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def compute_statistical_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute basic statistical patterns and correlations.
    Returns a dictionary ready for the LLM to interpret.
    """
    logger.info("Computing statistical insights...")

    insights = {}

    # 1️⃣ Numeric columns
    num_df = df.select_dtypes(include=np.number)
    if not num_df.empty:
        corr = num_df.corr()
        # Take top correlations (excluding self)
        corr_pairs = (
            corr.unstack()
            .sort_values(ascending=False)
            .drop_duplicates()
            .head(10)
            .to_dict()
        )
        insights["top_correlations"] = {
            str(k): float(v) for k, v in corr_pairs.items() if k[0] != k[1]
        }
    else:
        insights["top_correlations"] = {}

    # 2️⃣ Missing values %
    missing_pct = (df.isnull().sum() / len(df) * 100).to_dict()
    insights["missing_percentage"] = missing_pct

    # 3️⃣ Basic numeric summary
    insights["numeric_summary"] = df.describe().to_dict()

    logger.info("Statistical insights computed successfully.")
    return insights



def llm_generate_insights(
    stat_summary: Dict[str, Any],
    model_name: str = "llama-3.1-8b-instant",
    groq_api_key: Optional[str] = None,
) -> str:
    """
    Calls Groq's LLM to interpret the statistical summary.
    Returns natural-language insights.
    """
    logger.info("Generating insights using Groq LLM...")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Groq API key not found. Please set GROQ_API_KEY environment variable.")

    client = Groq(api_key=api_key)

    prompt = f"""
    You are an expert data analyst.
    Analyze the following dataset summary and produce
    a clear, concise set of insights:
    {json.dumps(stat_summary, indent=2)}

    Highlight trends, strong correlations, anomalies,
    or interesting patterns in natural language.
    """

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful data analyst."},
            {"role": "user", "content": prompt},
        ],
    )

    message = response.choices[0].message.content
    logger.info("LLM insights generated.")
    return message



def insight_generator_agent(
    df: pd.DataFrame,
    use_llm: bool = True,
    model_name: str = "llama-3.1-8b-instant",
    groq_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Combined agent function.
    - Computes statistical insights
    - Optionally generates LLM insights
    """
    stats = compute_statistical_insights(df)
    text_insights = ""

    if use_llm:
        try:
            text_insights = llm_generate_insights(
                stats, model_name=model_name, groq_api_key=groq_api_key
            )
        except Exception as e:
            logger.exception("LLM insight generation failed: %s", e)
            text_insights = "LLM insight generation failed."

    return {
        "statistical_insights": stats,
        "text_insights": text_insights,
    }

