import os
import logging
from typing import Dict, Any, Optional, TypedDict

from agents.Data_parser_agent import data_parser_agent
from agents.insight_generator_agent import insight_generator_agent
from agents.viz_generator import viz_generator_agent
from agents.report_generator_agent import report_generator_agent
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class AnalysisState(TypedDict, total=False):
    df: Any
    summary: Dict[str, Any]
    insights: Any
    visuals: Dict[str, Any]
    report_path: str
    output_dir: str
    use_llm: bool
    groq_api_key: Optional[str]
    model_name: str


def build_analysis_graph():
    """
    Build and return a LangGraph StateGraph that connects all four agents.
    """

    graph = StateGraph(AnalysisState)


    #1️ Data Parser Node
    def node_data_parser(state: Dict[str, Any]) -> Dict[str, Any]:
        df = state["df"]
        summary = data_parser_agent(df)
        logger.info("✅ Data parsed successfully: %s rows, %s columns",
                    summary.get("num_rows"), summary.get("num_columns"))
        state["summary"] = summary
        return state

    #2 Insight Generator Node
    def node_insight_generator(state: Dict[str, Any]) -> Dict[str, Any]:
        df = state["df"]
        use_llm = state.get("use_llm", True)
        groq_api_key = state.get("groq_api_key", None)
        model_name = state.get("model_name", "llama-3.1-8b-instant")

        insights = insight_generator_agent(
            df, use_llm=use_llm, groq_api_key=groq_api_key, model_name=model_name
        )
        logger.info("✅ Insights generated successfully.")
        state["insights"] = insights
        return state

    #3 Visualization Node
    def node_visualization(state: Dict[str, Any]) -> Dict[str, Any]:
        df = state["df"]
        output_dir = state.get("output_dir", "output/visuals")
        visuals = viz_generator_agent(df, output_dir=output_dir, save_images=True)
        logger.info("✅ %s visualizations created.", len(visuals.get("visualizations", {})))
        state["visuals"] = visuals
        return state

    #4️ Report Generator Node
    def node_report_generator(state: Dict[str, Any]) -> Dict[str, Any]:
        summary = state["summary"]
        insights = state["insights"]
        visuals = state["visuals"]
        output_dir = state.get("output_dir", "output")
        pdf_path = os.path.join(output_dir, "data_analysis_report.pdf")

        report_path = report_generator_agent(
            summary_data=summary,
            insights_data=insights,
            viz_data=visuals,
            output_path=pdf_path,
        )
        logger.info("✅ Report generated successfully: %s", report_path)
        state["report_path"] = report_path
        return state

    graph.add_node("data_parser", node_data_parser)
    graph.add_node("insight_generator", node_insight_generator)
    graph.add_node("visualization", node_visualization)
    graph.add_node("report_generator", node_report_generator)

    graph.add_edge("data_parser", "insight_generator")
    graph.add_edge("insight_generator", "visualization")
    graph.add_edge("visualization", "report_generator")
    graph.add_edge("report_generator", END)

    graph.set_entry_point("data_parser")

    return graph


def run_langgraph_pipeline(
    df,
    output_dir: str = "output",
    use_llm: bool = True,
    model_name: str = "llama-3.1-8b-instant",
    groq_api_key: Optional[str] = None,
):
    """
    Executes the full analysis pipeline as a LangGraph flow.
    """
    

    # Build the graph
    graph = build_analysis_graph()
    runner = graph.compile()

    # Initial state
    state = {
        "df": df,
        "output_dir": output_dir,
        "use_llm": use_llm,
        "groq_api_key": groq_api_key,
        "model_name": model_name,
    }

    # Run the graph
    logger.info("🚀 Executing LangGraph pipeline...")
    final_state = runner.invoke(state)
    logger.info("🎯 Pipeline execution completed.")

    return final_state



