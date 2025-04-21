#pip install fastapi uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Lupus Diagnosis API",
    description="API for lupus diagnosis with 24 ordered integer inputs and 1 float output",
    version="1.0.0"
)

# Load the trained model
try:
    model = joblib.load("trained_model.pkl")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise Exception(f"Failed to load model: {str(e)}")

# Define the input schema with 24 integer fields
class ModelInput(BaseModel):
    Ana_test: int
    Fever: int
    Leukopenia: int
    Thrombocytopenia: int
    Autoimmune_hemolysis: int
    Delirium: int
    Psychosis: int
    Seizure: int
    Non_scarring_alopecia: int
    Oral_ulcers: int
    Cutaneous_lupus: int
    Pleural_effusion: int
    Pericardial_effusion: int
    Acute_pericarditis: int
    Joint_involvement: int
    Proteinuria: int
    Renal_biopsy: int
    anti_cardiolipin_anitbody: int
    anti_b2gp1_antibody: int
    lupus_anticoagulant: int
    low_c3: int
    low_c4: int
    anti_dsDNA_antibody: int
    anti_smith_antibody: int

# Define the output schema
class ModelOutput(BaseModel):
    prediction: int

# Prediction function using the trained model
def predict(inputs: list[int]) -> int:
    """
    Makes a prediction using the trained SVM model.
    Inputs must be a list of 24 integers in the correct order.
    """
    input_array = np.array([inputs])  # Shape: (1, 24)
    return int(model.predict(input_array)[0])

# Prediction endpoint
@app.post("/predict", response_model=ModelOutput)
async def predict_endpoint(data: ModelInput):
    try:
        # Extract inputs in order to maintain column correspondence
        input_list = [
            data.Ana_test, data.Fever, data.Leukopenia, data.Thrombocytopenia,
            data.Autoimmune_hemolysis, data.Delirium, data.Psychosis, data.Seizure,
            data.Non_scarring_alopecia, data.Oral_ulcers, data.Cutaneous_lupus, data.Pleural_effusion,
            data.Pericardial_effusion, data.Acute_pericarditis, data.Joint_involvement, data.Proteinuria,
            data.Renal_biopsy, data.anti_cardiolipin_anitbody, data.anti_b2gp1_antibody, data.lupus_anticoagulant,
            data.low_c3, data.low_c4, data.anti_dsDNA_antibody, data.anti_smith_antibody
        ]
        
        # Get prediction from the model
        result = predict(input_list)
        logger.info("Prediction successful")
        
        # Return the result
        return {"prediction": result}
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "Lupus Diagnosis API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)