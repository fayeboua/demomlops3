# =========================================
# H2O AutoML Training with MLflow Tracking
# Author: Kenneth Leung
# =========================================
import os
import argparse
import h2o
from h2o.automl import H2OAutoML, get_leaderboard

import mlflow
import mlflow.h2o
from mlflow.tracking import MlflowClient

import json

def train(experiment_name, target, models, file_path):
    # ========== Setup Tracking & Paths ==========
    mlruns_path = "/app/backend/mlruns"
    os.makedirs(mlruns_path, exist_ok=True)
    mlflow.set_tracking_uri(f"file:{mlruns_path}")

    # ========== Init H2O ==========
    h2o.init()

    # ========== Load Training Data ==========
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Training file not found: {file_path}")

    main_frame = h2o.import_file(path=file_path)

    # Save column types for prediction use
    col_type_path = os.path.join(os.path.dirname(file_path), 'train_col_types.json')
    with open(col_type_path, 'w') as fp:
        json.dump(main_frame.types, fp)

    predictors = [n for n in main_frame.col_names if n != target]
    main_frame[target] = main_frame[target].asfactor()

    # ========== MLflow Experiment Setup ==========
    client = MlflowClient()
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        experiment = client.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)

    # ========== AutoML + Tracking ==========
    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # Log input artifacts
        mlflow.log_artifact(file_path, artifact_path="input_data")
        mlflow.log_artifact(col_type_path, artifact_path="input_data")

        # Train H2O model
        aml = H2OAutoML(
            max_models=models,
            seed=42,
            balance_classes=True,
            sort_metric='logloss',
            verbosity='info',
            exclude_algos=['GLM', 'DRF']
        )
        aml.train(x=predictors, y=target, training_frame=main_frame)

        # Log metrics
        mlflow.log_metric("log_loss", aml.leader.logloss())
        mlflow.log_metric("AUC", aml.leader.auc())

        # Log model
        mlflow.h2o.log_model(aml.leader, artifact_path="model")

        # Save leaderboard
        model_uri = mlflow.get_artifact_uri("model")
        if model_uri.startswith("file:/"):
            model_path = model_uri.replace("file:", "")
            os.makedirs(model_path, exist_ok=True)
            leaderboard_path = os.path.join(model_path, "leaderboard.csv")

            lb = get_leaderboard(aml, extra_columns='ALL')
            lb.as_data_frame().to_csv(leaderboard_path, index=False)

            print(f"[✓] Model saved at: {model_uri}")
            print(f"[✓] Leaderboard saved at: {leaderboard_path}")
        else:
            print(f"[!] Unexpected artifact URI: {model_uri}")


def parse_args():
    parser = argparse.ArgumentParser(description="H2O AutoML Training and MLflow Tracking")

    parser.add_argument('--name', '--experiment_name',
                         metavar='', 
                        #  required=True,
                         default='automl-insurance',
                         help='Name of Experiment. Default is automl-insurance', 
                         type=str
                        )

    parser.add_argument('--target', '--t',
                        metavar='', 
                        required=True,
                        help='Name of Target Column (y)', 
                        type=str
                    )

    parser.add_argument('--models', '--m',
                        metavar='', 
                        # required=True,
                        default=10,
                        help='Number of AutoML models to train. Default is 10', 
                        type=int
                    )

    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_args()

    # Initiate H2O cluster
    h2o.init()

    # Initiate MLflow client
    client = MlflowClient()

    # Get parsed experiment name
    experiment_name = args.name

    # Create MLflow experiment
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        experiment = client.get_experiment_by_name(experiment_name)
    except:
        experiment = client.get_experiment_by_name(experiment_name)
    
    mlflow.set_experiment(experiment_name)

    # Print experiment details
    print(f"Name: {experiment_name}")
    print(f"Experiment_id: {experiment.experiment_id}")
    print(f"Artifact Location: {experiment.artifact_location}")
    print(f"Lifecycle_stage: {experiment.lifecycle_stage}")
    print(f"Tracking uri: {mlflow.get_tracking_uri()}")

    # Import data directly as H2O frame (default location is data/processed)
    main_frame = h2o.import_file(path='data/processed/train.csv')

    # Save column data types of H2O frame (for matching with test set during prediction)
    with open('data/processed/train_col_types.json', 'w') as fp:
        json.dump(main_frame.types, fp)

    # Set predictor and target columns
    target = args.target
    predictors = [n for n in main_frame.col_names if n != target]

    # Factorize target variable so that autoML tackles classification problem
    main_frame[target] = main_frame[target].asfactor()

    # Setup and wrap AutoML training with MLflow
    with mlflow.start_run():
        aml = H2OAutoML(
                        max_models=args.models, # Run AutoML for n base models
                        seed=42, 
                        balance_classes=True, # Target classes imbalanced, so set this as True
                        sort_metric='logloss', # Sort models by logloss (metric for multi-classification)
                        verbosity='info', # Turn on verbose info
                        exclude_algos = ['GLM', 'DRF'], # Specify algorithms to exclude
                    )
        
        # Initiate AutoML training
        aml.train(x=predictors, y=target, training_frame=main_frame)
        
        # Set metrics to log
        mlflow.log_metric("log_loss", aml.leader.logloss())
        mlflow.log_metric("AUC", aml.leader.auc())
        
        # Log and save best model (mlflow.h2o provides API for logging & loading H2O models)
        mlflow.h2o.log_model(aml.leader, artifact_path="model")
        
        model_uri = mlflow.get_artifact_uri("model")
        print(f'AutoML best model saved in {model_uri}')
        
        # Get IDs of current experiment run
        exp_id = experiment.experiment_id
        run_id = mlflow.active_run().info.run_id
        
        # Save leaderboard as CSV
        lb = get_leaderboard(aml, extra_columns='ALL')
        lb_path = f'mlruns/{exp_id}/{run_id}/artifacts/model/leaderboard.csv'
        lb.as_data_frame().to_csv(lb_path, index=False) 
        print(f'AutoML Complete. Leaderboard saved in {lb_path}')


if __name__ == "__main__":
    main()