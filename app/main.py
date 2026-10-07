import os
from fastapi import FastAPI

app = FastAPI(title="invisibleai2026")

@app.get("/health")
def health():
    return {"status": "ok", "env": os.getenv("APP_ENV", "dev")}

@app.get("/")
def root():
    return {"message": "Hello World"}