"""partB/eta_pyfunc.py -- the custom MLflow pyfunc that wraps an ETA joblib bundle.

Kept in its own small module (shipped with the model via `code_paths`) so that
`mlflow models serve` can unpickle it without importing the rest of the project.
"""
import joblib
import mlflow.pyfunc
import pandas as pd


class EtaPyfunc(mlflow.pyfunc.PythonModel):
    """Input: DataFrame with the 7 features.  Output: DataFrame[eta_minutes (1 decimal), model_version]."""

    def load_context(self, context):
        bundle = joblib.load(context.artifacts["bundle"])
        self.model = bundle["model"]
        self.features = list(bundle["feature_columns"])
        self.version = str(bundle["version"])

    def predict(self, context, model_input, params=None):
        X = pd.DataFrame(model_input)[self.features]
        y = self.model.predict(X)
        return pd.DataFrame({"eta_minutes": [round(float(v), 1) for v in y],
                             "model_version": [self.version] * len(X)})
