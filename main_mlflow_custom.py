import os
from json import load
from typing_extensions import Annotated  # or `from typing import Annotated on Python 3.9+
from typing import Tuple
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pendulum
from dotenv import load_dotenv
import warnings

from zenml import pipeline, step, ArtifactConfig
from zenml.client import Client
from zenml.integrations.mlflow.flavors.mlflow_experiment_tracker_flavor import MLFlowExperimentTrackerSettings

import mlflow
from mlflow.models import infer_signature

import logging

warnings.filterwarnings('ignore')
load_dotenv()

PROJECT_NAME = "SVC_example" # ОБЯЗАТЕЛЬНО ЗАПОЛНИТЬ
S3_SUBFOLDER_NAME = pendulum.today().to_date_string()

experiment_tracker = Client().active_stack.experiment_tracker

@step(
    enable_cache=False,
    name=f"{PROJECT_NAME}/{S3_SUBFOLDER_NAME}/training_data_loader",
    enable_artifact_metadata=True,
    enable_artifact_visualization=True,
    enable_step_logs=True
)
def training_data_loader() -> Tuple[
    # Notice we use a Tuple and Annotated to return 
    # multiple named outputs
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """Load the iris dataset as a tuple of Pandas DataFrame / Series."""

    logging.info("Loading iris...")
    iris = load_iris(as_frame=True)
    logging.info("Splitting train and test...")
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, shuffle=True, random_state=42
    )
    return X_train, X_test, y_train, y_test

@step(
    enable_cache=False,
    name=f"{PROJECT_NAME}/{S3_SUBFOLDER_NAME}/svc_trainer",
    enable_artifact_metadata=True,
    enable_artifact_visualization=True,
    enable_step_logs=True,
    experiment_tracker=experiment_tracker.name,
    settings={
       "experiment_tracker.mlflow": {
           "experiment_name": PROJECT_NAME,
           "nested": False,
           "tags":{
                "model_trainer": "e.chernov",
                "model_customer": "y.smyk"
            }
       }
   } 
)
def svc_trainer(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    # gamma: float = 0.001,
    svc_params: dict
) -> Tuple[
    Annotated[ClassifierMixin, "model"],
    Annotated[float, "training_acc"],
]:
    """Train a sklearn SVC classifier."""

    model = SVC(**svc_params)
    model.fit(X_train.to_numpy(), y_train.to_numpy())

    train_acc = accuracy_score(y_train.to_numpy(), model.predict(X_train.to_numpy()))

    # Логируем метрики модели
    mlflow.log_params(svc_params)
    mlflow.log_metric("accuracy", train_acc)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="mlflow/model",
        signature=infer_signature(X_train, model.predict(X_train.to_numpy())),
        input_example=X_train,
        registered_model_name="SVC_registered_model_name"
    )

    # Фиксируем что модель принимает на вход и выдает на выходе
    mlflow.log_input(
        mlflow.data.from_pandas(X_train, source="ORACLE_DB"),
        context='training'
    )

    mlflow.log_table(
        X_train,
        "log_table/table.json"
    )

    return model, train_acc

@pipeline(
    enable_cache=False,
    name=f"{PROJECT_NAME}",
    enable_artifact_metadata=True,
    enable_step_logs=True,
    tags=["tag_1"]
)
def training_pipeline(svc_params):
    X_train, X_test, y_train, y_test = training_data_loader()
    svc_trainer(svc_params=svc_params, X_train=X_train, y_train=y_train)


if __name__ == "__main__":
    svc_params = {
        "C": 1,
        "gamma": 0.002,
        "kernel": "rbf",
        "degree": 3
    }

    training_pipeline(svc_params=svc_params)

    # for i in [0.002, 0.004, 0.006]:
    #     svc_params["gamma"] = i
        # training_pipeline(svc_params=svc_params)