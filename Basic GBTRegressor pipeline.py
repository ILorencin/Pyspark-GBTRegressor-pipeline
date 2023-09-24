from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator

# File location and type
file_location = "/FileStore/tables/data_set.csv"
file_type = "csv"

# Dataframe cration using .csv dataset 
df=spark.read.csv(file_location,header=True,inferSchema=True)
df=df.na.drop('any')

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = df.randomSplit([0.7, 0.3])

# Create feature assembler
feature_assembler=VectorAssembler(inputCols=['AT','V','AP','RH'],outputCol='Input vector')

# Create a GBT model.
gbt = GBTRegressor(featuresCol="Input vector",labelCol='PE', maxIter=10)

# Chain feature assembler and GBT in a Pipeline
pipeline = Pipeline(stages=[feature_assembler,gbt])

# Train model
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display. in this case, first 5 rows are displayed
predictions.select("PE", "prediction", "Input vector").show(5)

# Select (prediction, true label) and compute r2 score
evaluator = RegressionEvaluator(
    labelCol="prediction", predictionCol="PE", metricName="r2")
r2 = evaluator.evaluate(predictions)
print("R2 on test data = %g" % r2)

# Print model summary
gbtModel = model.stages[1]
print(gbtModel)  # summary only
