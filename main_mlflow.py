import os
from json import load
from typing_extensions import Annotated  # or `from typing import Annotated on Python 3.9+
from typing import Tuple
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin
from sklearn.svm import SVC
import pendulum
from dotenv import load_dotenv
import warnings

from zenml import pipeline, step, ArtifactConfig
from zenml.client import Client
from zenml.integrations.mlflow.flavors.mlflow_experiment_tracker_flavor import MLFlowExperimentTrackerSettings
import mlflow
import logging

warnings.filterwarnings('ignore')
load_dotenv()

PROJECT_NAME = "SVC_new" # ОБЯЗАТЕЛЬНО ЗАПОЛНИТЬ
S3_SUBFOLDER_NAME = pendulum.today().to_date_string()

experiment_tracker = Client().active_stack.experiment_tracker
mlflow_settings = MLFlowExperimentTrackerSettings(
    experiment_name=PROJECT_NAME,
    experiment_id=S3_SUBFOLDER_NAME,
    nested=True,
    tags={"key": "value"}  
)

@step(
    enable_cache=False,
    name=f"{PROJECT_NAME}/{S3_SUBFOLDER_NAME}/training_data_loader"
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
    experiment_tracker=experiment_tracker.name,
    settings={
       "experiment_tracker": mlflow_settings
   } 
)
def svc_trainer(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    gamma: float = 0.001,
) -> Tuple[
    Annotated[ClassifierMixin, "model"],
    Annotated[float, "training_acc"],
]:
    """Train a sklearn SVC classifier."""
    mlflow.sklearn.autolog() # turn on autologging for mlflow

    model = SVC(gamma=gamma)
    model.fit(X_train.to_numpy(), y_train.to_numpy())

    train_acc = model.score(X_train.to_numpy(), y_train.to_numpy())
    print(f"Train accuracy: {train_acc}")

    return model, train_acc

@pipeline(
    enable_cache=False,
    name=f"{PROJECT_NAME}"
)
def training_pipeline(gamma: float = 0.002):
    X_train, X_test, y_train, y_test = training_data_loader()
    svc_trainer(gamma=gamma, X_train=X_train, y_train=y_train)


if __name__ == "__main__":
    training_pipeline(gamma=0.004)