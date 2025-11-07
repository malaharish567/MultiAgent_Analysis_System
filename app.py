from flask import Flask, render_template, request
import os
from workflow.langgraph_pipeline import run_langgraph_pipeline
import pandas as pd



app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    report = None
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", report="❌ No file part in request")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", report="❌ No file selected")

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        try:
            
            df = pd.read_csv(filepath)

            
            report = run_langgraph_pipeline(df)

        
            if not isinstance(report, str):
                report = str(report)

            report = f"✅ File '{file.filename}' uploaded successfully.\n\nReport:\n\n{report}"

        except Exception as e:
            report = f"⚠️ Error while generating report: {str(e)}"

        return render_template("index.html", report=report)

    return render_template("index.html", report=report)

if __name__ == "__main__":
    app.run(debug=True)
