# API dependencies
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import HTMLResponse

# Modeldependencies
from transformers import pipeline
import shap

# Initialize explainer
classifier = pipeline("sentiment-analysis")
explainer = shap.Explainer(classifier)

# run App
app = FastAPI()

class TextParam(BaseModel):
    text: str

@app.post("/explain")
def explain_text(data:TextParam):
    text = data.text
    shap_values = explainer([text])

    html_content = shap.plots.text(shap_values[0], display=False)
    return HTMLResponse(content=html_content)

@app.post("/predict")
def predict_text(data:TextParam):
    text = data.text
    prediction = classifier(text)
    return {'predictions': prediction[0]}
