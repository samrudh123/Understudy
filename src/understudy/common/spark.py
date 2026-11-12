"""SparkSession factory for the local-mode batch jobs."""

from __future__ import annotations

import os
import sys
from typing import Any

from pyspark.sql import SparkSession


def get_spark(app_name: str, spark_params: dict[str, Any]) -> SparkSession:
    """Build (or reuse) a local SparkSession configured from ``params.yaml: spark``."""
    # Make the Python workers use the same interpreter (and conda env) as the driver.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    spark = (
        SparkSession.builder.appName(app_name)
        .master(spark_params.get("master", "local[*]"))
        .config("spark.driver.memory", spark_params.get("driver_memory", "4g"))
        .config("spark.sql.shuffle.partitions", str(spark_params.get("shuffle_partitions", 8)))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark
