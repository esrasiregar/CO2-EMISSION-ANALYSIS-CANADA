from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="CO2 Emission Prediction (XGBoost)")

with open("best_co2_emission_model.pkl", "rb") as f:
    model = pickle.load(f)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class VehicleInput(BaseModel):
    Engine_Size_L: float = Field(..., alias="Engine Size(L)")
    Cylinders: int
    Fuel_Consumption_Comb_L100km: float = Field(..., alias="Fuel Consumption Comb (L/100 km)")
    Fuel_Type_D: float = Field(..., alias="Fuel Type_D")
    Fuel_Type_E: float = Field(..., alias="Fuel Type_E")
    Fuel_Type_N: float = Field(..., alias="Fuel Type_N")
    Fuel_Type_X: float = Field(..., alias="Fuel Type_X")
    Fuel_Type_Z: float = Field(..., alias="Fuel Type_Z")

@app.get("/")
def read_root():
    return {"message": "CO2 Emission Predict API is running"}

@app.post("/predict")
async def predict(co2_input: VehicleInput):
    input_data = co2_input.dict(by_alias=True)
    
    input_df = pd.DataFrame([[
        input_data["Engine Size(L)"],
        input_data["Cylinders"],
        input_data["Fuel Consumption Comb (L/100 km)"],
        input_data["Fuel Type_E"],
        input_data["Fuel Type_N"],
        input_data["Fuel Type_X"],
        input_data["Fuel Type_Z"]
    ]], columns=[
        "Engine Size(L)",
        "Cylinders", 
        "Fuel Consumption Comb (L/100 km)",
        "Fuel Type_E",
        "Fuel Type_N",
        "Fuel Type_X",
        "Fuel Type_Z"
    ])
    
    prediction = model.predict(input_df)
    
    return {
    "predicted_co2_emission": round(float(prediction[0]), 2),
    "unit": "grams per kilometer (g/km)"
}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)