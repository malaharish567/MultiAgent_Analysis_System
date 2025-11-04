from workflow.langgraph_pipeline import run_langgraph_pipeline
import seaborn as sns

df = sns.load_dataset("tips")

result = run_langgraph_pipeline(df, use_llm=True)
print("✅ PDF report saved to:", result["report_path"])
