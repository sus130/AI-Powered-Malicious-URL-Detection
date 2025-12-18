import os
import sys
import certifi
import pymongo
import pandas as pd
from dotenv import load_dotenv

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from uvicorn import run as app_run

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.constant.training_pipeline import UPLOAD_DIR, DATA_INGESTION_COLLECTION_NAME, DATA_INGESTION_DATABASE_NAME

# ------------------- Load environment variables -------------------
load_dotenv()
mongo_db_url = os.getenv("MONGO_DB_URL")

# ------------------- MongoDB client -------------------
ca = certifi.where()
client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

# ------------------- FastAPI app -------------------
app = FastAPI(title="Network Security ML System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------- Configure logging to stdout -------------------
import logging as pylogging

pylogging.getLogger().handlers = []
pylogging.basicConfig(
    stream=sys.stdout,
    level=pylogging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True
)

from networksecurity.logging.logger import logging  # Existing logger

# ------------------- Load trained model -------------------
network_model = None
try:
    preprocessor = load_object("final_model/preprocessor.pkl")
    final_model = load_object("final_model/model.pkl")
    network_model = NetworkModel(preprocessor=preprocessor, model=final_model)
except Exception:
    pass  # Model not found; will train when /train is called

# ------------------- Routes -------------------
@app.get("/", tags=["home"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train", tags=["training"])
async def train_route():
    """
    Triggers the ML training pipeline; logs appear live in the terminal.
    """
    global network_model
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()  # Logs printed live

        # Reload the trained model
        preprocessor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")
        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)

        return JSONResponse(content={"message": "Training completed successfully. Model ready for predictions."})

    except Exception as e:
        raise NetworkSecurityException(e, sys)


@app.post("/predict", tags=["prediction"])
async def predict_route(file: UploadFile = File(...)):
    if network_model is None:
        return JSONResponse(content={"error": "Model not trained yet. Please run /train first."})

    try:
        # Read CSV
        df = pd.read_csv(file.file)

        # Drop label column if present
        if 'label' in df.columns:
            df = df.drop(columns=['label'])

        # Ensure exact match with preprocessor features
        preprocessor_columns = list(network_model.preprocessor.feature_names_in_)

        # Add missing columns with 0
        for col in preprocessor_columns:
            if col not in df.columns:
                df[col] = 0

        # Reorder columns exactly as preprocessor expects
        df = df[preprocessor_columns]

        # Save uploaded file
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Predict
        y_pred = network_model.predict(df)
        df['predicted_column'] = y_pred
        df['predicted_column'].replace(-1, 0, inplace=True)

        # Map numeric predictions to human-readable labels
        df['predicted_label'] = df['predicted_column'].map({0: "Safe", 1: "Suspicious"})

        # Optionally remove raw numeric column
        df = df.drop(columns=['predicted_column'])

        return JSONResponse(content=df.to_dict(orient="records"))

    except Exception as e:
        raise NetworkSecurityException(e, sys)



# ------------------- Run app -------------------
if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8000, reload=True)
