# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <!--
# MAGIC    Copyright 2023 Databricks Inc.
# MAGIC
# MAGIC    Licensed under the Apache License, Version 2.0 (the "License");
# MAGIC    you may not use this file except in compliance with the License.
# MAGIC    You may obtain a copy of the License at
# MAGIC
# MAGIC        http://www.apache.org/licenses/LICENSE-2.0
# MAGIC
# MAGIC    Unless required by applicable law or agreed to in writing, software
# MAGIC    distributed under the License is distributed on an "AS IS" BASIS,
# MAGIC    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# MAGIC    See the License for the specific language governing permissions and
# MAGIC    limitations under the License.
# MAGIC -->
# MAGIC
# MAGIC # Predict Proba Wrapper Example
# MAGIC
# MAGIC This is a simple demonstration of using a custom wrapper to have
# MAGIC MLflow's PyFunc flavor call predict_proba. In the first part, we 
# MAGIC show an example of a simple wrapper class and then compare and contrast
# MAGIC the various ways to log and load it using PyFunc vs scikit-learn
# MAGIC flavors. In the second part we look at loading and applying it
# MAGIC as a Spark UDF on a Spark DataFrame.

# COMMAND ----------

# DBTITLE 1,Imports and setup
from pprint import pprint

from pyspark.sql import functions as F
from pyspark.sql import types as T

import xgboost as xgb
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import mlflow
import mlflow.xgboost
import mlflow.sklearn

%config InlineBackend.figure_format = "retina"

# COMMAND ----------

# MAGIC %md
# MAGIC # Basic example

# COMMAND ----------

# DBTITLE 1,Simple wrapper class to override predict
class CustomWrapperModel(mlflow.pyfunc.PythonModel):

    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]

# COMMAND ----------

# DBTITLE 1,Train a simple example model; log both flavors
with mlflow.start_run() as run:

    # prepare example dataset
    X, y = load_wine(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = xgb.XGBClassifier(n_estimators=20, reg_lambda=1, gamma=0, max_depth=3)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    run_id = mlflow.last_active_run().info.run_id
    print("Logged data and model in run {}".format(run_id))

    wrapper = CustomWrapperModel(model)

    # logging the sklearn model will use the default predict
    mlflow.sklearn.log_model(sk_model=model, artifact_path="model")

    # we could instead log it to override the method pyfunc calls
    mlflow.sklearn.log_model(sk_model=model, artifact_path="predict_fn_model", pyfunc_predict_fn="predict_proba")

    # we can use a wrapper to have it call predict_proba (or do other processing)
    mlflow.pyfunc.log_model(python_model=wrapper, artifact_path="wrapper")


# COMMAND ----------

# DBTITLE 1,Load the model using various flavors
# The /model artifact path was logged with the sklearn flavor on the raw model
# We can load it with both the pyfunc flavor and sklearn flavor
model_uri = f"runs:/{run_id}/model"
loaded_model_pyfunc = mlflow.pyfunc.load_model(model_uri)
loaded_model_sklearn = mlflow.sklearn.load_model(model_uri)

# These two artifact paths were logged with the alternative predict_fn specified.
predict_fn_uri = f"runs:/{run_id}/predict_fn_model"
loaded_model_predict_fn = mlflow.pyfunc.load_model(predict_fn_uri)

# The /wrapper artifact path was logged with the pyfunc flavor.
# We can load it with the pyfunc flavor
wrapper_uri = f"runs:/{run_id}/wrapper"
loaded_model_wrapper = mlflow.pyfunc.load_model(wrapper_uri)

# COMMAND ----------

# DBTITLE 1,Pyfunc model has predict and returns predict
loaded_model_pyfunc.predict(X_test)

# COMMAND ----------

# DBTITLE 1,But it only has predict
# Note: expected to fail
# loaded_model_pyfunc.predict_proba(X_test)

# COMMAND ----------

# DBTITLE 1,Loaded as sklearn we can also call predict
loaded_model_sklearn.predict(X_test)

# COMMAND ----------

# DBTITLE 1,But we can also use predict_proba
# Note: expected to work
loaded_model_sklearn.predict_proba(X_test)[:, 1]

# COMMAND ----------

# DBTITLE 1,Pyfunc wrapper returns predict_proba from predict
loaded_model_wrapper.predict(X_test)

# COMMAND ----------

# DBTITLE 1,scikit-learn logged with pyfunc_predict_fn
loaded_model_predict_fn.predict(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC # Spark UDF example
# MAGIC
# MAGIC The advantage of having PyFunc call `predict_proba` is that we can let MLflow 
# MAGIC create our Spark UDF for us, and still get our custom results out, such as
# MAGIC probabilities via `predict_proba`.

# COMMAND ----------

# DBTITLE 1,Create Spark DataFrame and UDF's
X_test_df = spark.createDataFrame(X_test)

# pyfunc can automatically create a UDF for you
# Note: you can also create your own UDF's
#       this is just one approach (and the most common)
pyfunc_udf = mlflow.pyfunc.spark_udf(spark, model_uri)
wrapper_udf = mlflow.pyfunc.spark_udf(spark, wrapper_uri)
predict_fn_udf = mlflow.pyfunc.spark_udf(spark, predict_fn_uri, result_type="array<double>")

display(X_test_df)

# COMMAND ----------

# DBTITLE 1,See prediction results with default PyFunc
pyfunc_results = X_test_df.withColumn("prediction", pyfunc_udf(F.struct(*X_test_df.columns)))
display(pyfunc_results.select("prediction"))

# COMMAND ----------

# DBTITLE 1,See prediction results with default PyFunc wrapper
wrapper_results = X_test_df.withColumn("prediction", wrapper_udf(F.struct(*X_test_df.columns)))
display(wrapper_results.select(F.round("prediction", 3).alias("prediction")))

# COMMAND ----------

# DBTITLE 1,See prediction results with default predict_fn wrapper
predict_fn_results = X_test_df.withColumn("prediction", predict_fn_udf(F.struct(*X_test_df.columns)))
display(predict_fn_results.select(F.round(F.col("prediction")[1], 3)))
