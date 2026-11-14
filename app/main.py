from pathlib import Path
from fastapi import UploadFile, File
from io import BytesIO
from fastapi.responses import StreamingResponse
import joblib
import pandas as pd
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel, Field

# Initialize FastAPI
app = FastAPI(
    title="House Price Prediction API",
    description="California Housing Price Prediction using Machine Learning",
    
)


#load the train model
BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "house_model.joblib")
feature = joblib.load(BASE_DIR / "house_feature.joblib")



# input schemas

class HouseFeatureData(BaseModel):
    MedInc: float = Field(gt=0)
    HouseAge: float = Field(gt=0, le=100)
    AveRooms: float = Field(gt=0)
    AveBedrms: float = Field(gt=0)
    Population: float = Field(gt=0)
    AveOccup: float = Field(gt=0)
    Latitude: float = Field(ge=32, le=42)
    Longitude: float = Field(ge=-125, le=-114)

 
    

# Create the route

@app.get("/")
def home():
      return{
            "message":" house price prediction API",
            "status": "Running"
      }

#Health Check
@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": "RandomForestRegressor",
        "feature":feature,
      
    }

# prediction
@app.post("/predict")
def predict_housePrice(house: HouseFeatureData):
    try:
        input_data = pd.DataFrame([{
            "MedInc": house.MedInc,
            "HouseAge": house.HouseAge,
            "AveRooms": house.AveRooms,
            "AveBedrms": house.AveBedrms,
            "Population": house.Population,
            "AveOccup": house.AveOccup,
            "Latitude": house.Latitude,
            "Longitude": house.Longitude
        }])
        predicted=model.predict(input_data)[0]
         # Convert prediction into dollars
        predicted_price = predicted* 100000
        return {
            "success": True,
            "predicted_house_price": f"${ predicted_price:,.0f}",
            "predicted_price_short": f"${ predicted:,.2f}"
        }
    
    except Exception as e:
         raise HTTPException(
              status_code=500,
              detail=f"Prediction fail:{str(e)}"
         )
       

# @app.post("/predict-file")
# async def predict_file(file: UploadFile = File(...)):
#     # check file extension
#     if not file.filename or not file.filename.endswith(".csv"):
#         raise HTTPException(
#             status_code=400,
#             detail="Please upload a csv file only"
#         )

#     contents = await file.read()
#     df = pd.DataFrame(BytesIO(contents))

#     required_columns = [
#         "MedInc",
#         "HouseAge",
#         "AveRooms",
#         "AveBedrms",
#         "Population",
#         "AveOccup",
#         "Latitude",
#         "Longitude",
#     ]

#     missing_columns = [
#         col for col in required_columns if col not in df.columns
#     ]

#     if missing_columns:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Missing columns: {missing_columns}"
#         )

#     if len(df)==0:
#         raise HTTPException(
#             status_code=400,
#             detail="The uploaded file has no data row"
#         )

#     predictions = model.predict(df[required_columns])
#     predicted_prices = predictions * 100000

#     return {
#         "success": True,
#         "count": len(predictions),
#         "predictions": [
#             {
#                 "predicted_house_price": f"${price:,.0f}",
#                 "predicted_price_short": f"${pred:,.2f}",
#             }
#             for pred, price in zip(predictions, predicted_prices)
#         ],
#     }
     
     
 # Multiple house prediction using CSV upload
@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):

    # STEP 1: Check file extension
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file only"
        )

    try:

        # STEP 2: Read uploaded CSV
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded CSV file is empty"
            )

        df = pd.read_csv(BytesIO(contents))

        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="CSV contains no records"
            )

        # STEP 3: Required columns
        required_column = feature

        # STEP 4: Check missing columns
        missing_column = [
            col for col in required_column
            if col not in df.columns
        ]

        if missing_column:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns: {missing_column}"
            )

        # STEP 5: Select model features
        X = df[feature].copy()

        # STEP 6: Convert values to numeric
        X = X.apply(
            pd.to_numeric,
            errors="coerce"
        )

        # STEP 7: Check missing or invalid values
        if X.isnull().any().any():
            raise HTTPException(
                status_code=400,
                detail="CSV contains missing or invalid feature values"
            )

        # STEP 8: Check infinity values
        import numpy as np

        if not np.isfinite(X.to_numpy()).all():
            raise HTTPException(
                status_code=400,
                detail="CSV contains infinite values"
            )

        # STEP 9: Predict house prices
        predictions = model.predict(X)

        # STEP 10: Add prediction columns
        df["Predicted_Price"] = predictions

        df["Predicted_Price_USD"] = (
            predictions * 100000
        )

        # STEP 11: Generate summary
        total_records = len(df)

        average_price = df["Predicted_Price_USD"].mean()

        minimum_price = df["Predicted_Price_USD"].min()

        maximum_price = df["Predicted_Price_USD"].max()

        # STEP 12: Convert results to CSV
        output = BytesIO()

        df.to_csv(
            output,
            index=False
        )

        output.seek(0)

        # STEP 13: Return downloadable CSV
        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={
                "Content-Disposition":
                "attachment; filename=house_predictions.csv",

                "X-Total-Records": str(total_records),

                "X-Average-Price": str(round(average_price, 2)),

                "X-Minimum-Price": str(round(minimum_price, 2)),

                "X-Maximum-Price": str(round(maximum_price, 2))
            }
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"File prediction failed: {str(e)}"
        )

    finally:
        await file.close()
  
          
          
     
     