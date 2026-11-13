"""FastAPI serving layer: loan approval model (/predict), LangGraph pipeline (/invoke), /health.

Run from the project root:  uvicorn app:app --reload
Interactive docs:           http://127.0.0.1:8000/docs
"""
import os
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from serving import ROOT, load_model, predict_application

app = FastAPI(title="LangGraph Stateful Agent Pipeline API", version="1.0")
model = load_model()
_graph = None


class LoanApplication(BaseModel):
    """One loan application. Only the applicant's income is required; missing fields are filled."""
    ApplicantIncome: float = Field(ge=0)
    CoapplicantIncome: float = Field(default=0, ge=0)
    LoanAmount: Optional[float] = Field(default=None, ge=0)
    Loan_Amount_Term: Optional[float] = Field(default=None, gt=0)
    Credit_History: Optional[Literal[0, 1]] = None
    Gender: Optional[Literal["Male", "Female"]] = None
    Married: Optional[Literal["Yes", "No"]] = None
    Dependents: Optional[Literal["0", "1", "2", "3+"]] = None
    Education: Literal["Graduate", "Not Graduate"] = "Graduate"
    Self_Employed: Optional[Literal["Yes", "No"]] = None
    Property_Area: Literal["Urban", "Semiurban", "Rural"] = "Urban"


class InvokeRequest(BaseModel):
    input: str
    thread_id: str


def get_graph():
    """Import the LangGraph pipeline on first use (it reads data/ relative to the project root)."""
    global _graph
    if _graph is None:
        previous = os.getcwd()
        os.chdir(ROOT)
        try:
            from week1_run import app as pipeline_graph
        finally:
            os.chdir(previous)
        _graph = pipeline_graph
    return _graph


@app.get("/health")
def health():
    return {"status": "ok", "model": type(model).__name__}


@app.post("/predict")
def predict(application: LoanApplication):
    return predict_application(model, application.model_dump())


@app.post("/invoke")
def invoke(request: InvokeRequest):
    try:
        graph = get_graph()
        state = graph.invoke(
            {"question": request.input, "notes": "", "summary": "", "score": 0, "loops": 0},
            config={"configurable": {"thread_id": request.thread_id}},
        )
    except Exception as exc:  # report pipeline errors as an HTTP error instead of crashing
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")
    return {"thread_id": request.thread_id, "state": state}
