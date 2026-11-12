"""MLflow "models from code" entry point for the student pyfunc (see ``train.py``)."""

import mlflow

from understudy.student.model import StudentPyfunc

mlflow.models.set_model(StudentPyfunc())
