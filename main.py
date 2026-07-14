from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, this is my first AI-ready API"}

@app.get("/predict")
def predict():
    return {"prediction": "This is where a model's output would go"}