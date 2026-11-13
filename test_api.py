"""Call every endpoint of the running API and print the results, for a single proof screenshot.

Start the server first:  uvicorn app:app --reload
Then in a second terminal: python test_api.py
"""
import json
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"

APPLICATION = {
    "ApplicantIncome": 5000, "CoapplicantIncome": 0, "LoanAmount": 130, "Loan_Amount_Term": 360,
    "Credit_History": 1, "Gender": "Male", "Married": "Yes", "Dependents": "0",
    "Education": "Graduate", "Self_Employed": "No", "Property_Area": "Urban",
}


def call(method: str, path: str, payload: dict | None = None) -> None:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        BASE + path, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    print(f"\n{method} {path}")
    if payload is not None:
        print("request: ", json.dumps(payload))
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            print(f"status:   {response.status}")
            print("response:", json.dumps(json.loads(response.read()), indent=2))
    except urllib.error.HTTPError as err:
        print(f"status:   {err.code}")
        print("response:", err.read().decode())
    except urllib.error.URLError as err:
        print(f"Could not reach the server ({err.reason}). Is uvicorn running?")


if __name__ == "__main__":
    print("API check:", BASE)
    call("GET", "/health")
    call("POST", "/predict", APPLICATION)
    call("POST", "/invoke", {"input": "What is LangGraph?", "thread_id": "demo-1"})
