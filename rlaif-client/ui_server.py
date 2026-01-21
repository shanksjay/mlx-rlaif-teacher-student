import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import rlaif_core
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow browser extension
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/system_specs")
def get_system_specs():
    raw_specs = rlaif_core.detect_system_capabilities()
    # The rust function currently returns a formatted string.
    # In a real implementation, we'd have it return a dictionary/struct.
    return {"specs": raw_specs}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
