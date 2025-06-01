import os
import io
import pandas as pd
import h2o

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, HTMLResponse

import mlflow
import mlflow.h2o
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

from utils.data_processing import separate_id_col, preprocess_for_model
from train import train as train_model

app = FastAPI()

# Initialize H2O
h2o.init(nthreads=-1, max_mem_size="2G")

# Set MLflow tracking URI
mlflow_tracking_uri = os.getenv("MLFLOW_BACKEND_STORE_URI", "file:/app/backend/mlruns")
mlflow.set_tracking_uri(mlflow_tracking_uri)

# Load the best model if available
def load_best_model():
    try:
        client = MlflowClient()
        all_exps = [exp.experiment_id for exp in client.list_experiments()]
        runs = mlflow.search_runs(experiment_ids=all_exps, run_view_type=ViewType.ALL)

        if not runs.empty and 'metrics.log_loss' in runs.columns:
            best_idx = runs['metrics.log_loss'].idxmin()
            run_id = runs.loc[best_idx]['run_id']
            exp_id = runs.loc[best_idx]['experiment_id']
            print(f'[+] Loading best model: Run {run_id} of Experiment {exp_id}')

            # Use MLflow's standard run URI format
            model_uri = f"runs:/{run_id}/model"
            return mlflow.h2o.load_model(model_uri)
        else:
            print("[!] No model with 'log_loss' metric found.")
            return None
    except Exception as e:
        print(f"[!] Failed to load model: {str(e)}")
        return None

# Store model in memory for reuse
best_model = load_best_model()

@app.post("/predict")
async def predict(file: bytes = File(...)):
    try:
        if best_model is None:
            return JSONResponse(status_code=400, content={"error": "No trained model available. Please train one first."})

        print('[+] Prediction started')

        # Read uploaded CSV
        file_obj = io.BytesIO(file)
        test_df = pd.read_csv(file_obj)

        # Apply same preprocessing as training
        test_df = preprocess_for_model(test_df)

        # Convert to H2OFrame
        test_h2o = h2o.H2OFrame(test_df)

        # Separate ID column
        id_name, X_id, X_h2o = separate_id_col(test_h2o)

        # Predict
        preds = best_model.predict(X_h2o)

        # Prepare results
        if id_name is not None:
            preds_list = preds.as_data_frame()['predict'].tolist()
            id_list = X_id.as_data_frame()[id_name].tolist()
            preds_final = dict(zip(id_list, preds_list))
        else:
            preds_final = preds.as_data_frame()['predict'].tolist()

        return JSONResponse(content=jsonable_encoder(preds_final))

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/train")
async def train_api(
    experiment_name: str = Form(...),
    target: str = Form(...),
    models: int = Form(...),
    file: UploadFile = File(...)
):
    try:
        contents = await file.read()
        file_path = f"/tmp/{file.filename}"

        with open(file_path, "wb") as f:
            f.write(contents)

        train_model(experiment_name, target, models, file_path)

        return JSONResponse(content={"message": "Training completed successfully."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
async def home():
    return HTMLResponse(content="""
    <h2>API AutoML – Assurance Cross-Sell</h2>
    <p>Send a CSV file to <code>/predict</code></p>
    <p>Swagger UI available at <code>/docs</code></p>
    """)
