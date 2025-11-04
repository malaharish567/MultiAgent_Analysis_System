from typing import Dict, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import logging
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



def _fig_to_base64() -> str:
    """Converts the current Matplotlib figure to a base64 string."""
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close()
    return img_base64



def viz_generator_agent(
    df: pd.DataFrame,
    output_dir: Optional[str] = None,
    save_images: bool = True,
) -> Dict[str, Any]:
    """
    Generates automated visualizations for numeric columns.

    Args:
        df: Pandas DataFrame
        output_dir: optional folder to save PNG files
        save_images: if True, saves charts to output_dir

    Returns:
        Dict containing:
            - message: status
            - visualizations: {title: base64_img_string}
    """

    logger.info("Starting visualization generation...")

    visualizations = {}
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if len(numeric_cols) == 0:
        logger.warning("No numeric columns found for visualization.")
        return {"message": "No numeric columns found.", "visualizations": {}}

    # Create output folder if required
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # --- Correlation Heatmap ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    heatmap_b64 = _fig_to_base64()
    visualizations["correlation_heatmap"] = heatmap_b64

    if save_images and output_dir:
        with open(os.path.join(output_dir, "correlation_heatmap.png"), "wb") as f:
            f.write(base64.b64decode(heatmap_b64))

    # ---Distribution plots for each numeric column ---
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True, bins=20)
        plt.title(f"Distribution of {col}")
        dist_b64 = _fig_to_base64()
        visualizations[f"{col}_distribution"] = dist_b64

        if save_images and output_dir:
            with open(os.path.join(output_dir, f"{col}_distribution.png"), "wb") as f:
                f.write(base64.b64decode(dist_b64))

    # ---Optional Pairplot (for <=5 numeric columns) ---
    if len(numeric_cols) <= 5:
        sns.pairplot(df[numeric_cols])
        plt.suptitle("Pairwise Relationships", y=1.02)
        pairplot_b64 = _fig_to_base64()
        visualizations["pairplot"] = pairplot_b64

        if save_images and output_dir:
            with open(os.path.join(output_dir, "pairplot.png"), "wb") as f:
                f.write(base64.b64decode(pairplot_b64))

    logger.info("Visualization generation completed successfully.")
    return {
        "message": "Visualizations created successfully!",
        "visualizations": visualizations,
    }


