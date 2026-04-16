from fastapi import FastAPI
from run_pipeline import main as run_pipeline

app = FastAPI(title="Opportunity Discovery API")


@app.get("/")
def root():
    return {"status": "API is running"}


@app.post("/run-pipeline")
def run_nlp_pipeline():
    """
    Runs the full NLP pipeline and stores results in MongoDB
    """
    run_pipeline()
    return {"message": "Pipeline executed successfully"}
