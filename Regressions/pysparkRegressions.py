from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
conf_spark = SparkConf().set("spark.driver.host", "127.0.0.1")
sc = SparkContext(conf=conf_spark)

spark = SparkSession.builder.appName('Cusomers').getOrCreate()


from pyspark.ml.regression import LinearRegression
dataset = spark.read.csv("customer.csv", inferSchema=True, header=True)
print(dataset)
dataset.show()
dataset.printSchema()
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
featureassembler = VectorAssembler(inputCols=["Avg Session Length","Time on App","Time on Website","Length of Membership"],outputCol="Independent Features")
output = featureassembler.transform(dataset)
output.show()
output.select("Independent Features").show()
print(output.columns)
finalized_data = output.select("Independent Features","Yearly Amount Spent")
finalized_data.show()
train_data,test_data=finalized_data.randomSplit([0.75,0.25])
regressor=LinearRegression(featuresCol='Independent Features', labelCol='Yearly Amount Spent')
regressor=regressor.fit(train_data)
print(regressor.coefficients)
(regressor.intercept)

pred_results = regressor.evaluate(test_data)
pred_results.predictions.show(40)
