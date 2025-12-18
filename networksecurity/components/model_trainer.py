import os
import sys
import asyncio

import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object, load_object, load_numpy_array_data, evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

# ------------------ Windows async fix ------------------
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ------------------ MLflow local setup ------------------
mlruns_dir = os.path.join(os.getcwd(), "mlruns")
os.makedirs(mlruns_dir, exist_ok=True)
mlflow.set_tracking_uri(f"file:///{mlruns_dir}")
print("✅ MLflow Tracking URI set to:", mlflow.get_tracking_uri())


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def track_mlflow(self, model, model_name, metric, X_sample=None):
        """Log model and metrics to MLflow."""
        try:
            with mlflow.start_run(run_name=model_name):
                mlflow.log_metric("f1_score", metric.f1_score)
                mlflow.log_metric("precision", metric.precision_score)
                mlflow.log_metric("recall_score", metric.recall_score)
                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model",
                    input_example=X_sample[:5] if X_sample is not None else None
                )
                print(f"[MLflow] Logged {model_name} successfully!")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def train_all_models(self, X_train, y_train, X_test, y_test):
        models = {
            "Random Forest": RandomForestClassifier(verbose=1),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(verbose=1),
            "Logistic Regression": LogisticRegression(verbose=1, max_iter=500),
            "AdaBoost": AdaBoostClassifier(),
        }

        # Hyperparameters for each model
        params = {
            "Random Forest": {"n_estimators": [100, 200], "max_depth": [None, 10, 20]},
            "Decision Tree": {"criterion": ["gini", "entropy"], "max_depth": [None, 10, 20]},
            "Gradient Boosting": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1]},
            "Logistic Regression": {"C": [0.1, 1.0, 10], "solver": ["liblinear"]},
            "AdaBoost": {"n_estimators": [50, 100], "learning_rate": [0.5, 1.0]},
        }

        # Evaluate all models
        model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)

        # Sort models by score descending and get top 2
        sorted_models = sorted(model_report.items(), key=lambda x: x[1], reverse=True)
        top_2_models = [sorted_models[0][0], sorted_models[1][0]]
        best_model_name = top_2_models[0]
        best_model = models[best_model_name]

        print(f"🚀 Training all models...")
        for name, model in models.items():
            print(f"🔹 Training {name}...")
            model.fit(X_train, y_train)
            print(f"✅ {name} trained.")

            if name in top_2_models:
                y_train_pred = model.predict(X_train)
                train_metric = get_classification_score(y_train, y_train_pred)

                y_test_pred = model.predict(X_test)
                test_metric = get_classification_score(y_test, y_test_pred)

                self.track_mlflow(model, f"{name}_train", train_metric, X_train)
                self.track_mlflow(model, f"{name}_test", test_metric, X_test)

        # Save best model for prediction
        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
        os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)

        network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path, network_model)
        save_object("final_model/model.pkl", best_model)

        logging.info(f"Best Model: {best_model_name} saved at {self.model_trainer_config.trained_model_file_path}")

        return ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=get_classification_score(y_train, best_model.predict(X_train)),
            test_metric_artifact=get_classification_score(y_test, best_model.predict(X_test))
        )

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            return self.train_all_models(X_train, y_train, X_test, y_test)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
