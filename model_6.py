from pyspark.sql import SparkSession
from pyspark.sql import Row

from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.classification import NaiveBayesModel
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import DecisionTreeClassificationModel
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import LinearSVCModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression

def parse_line(p):
    cols = p.split(' ')
    label = cols[0]
    if label not in ('1', '0'):
        return None

    label = 1.0 if label == '1' else 0.0
    fname = ' '.join(cols[1:])

    return Row(label=label, sentence=fname)


def train(spark):
    sc = spark.sparkContext
    tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=10000)
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    srcdf = sc.textFile('file:///home/bigdata2019_group36/BigData/process_train.csv').map(parse_line)
    srcdf = srcdf.toDF()
    training, testing = srcdf.randomSplit([0.99, 0.01])

    wordsData = tokenizer.transform(training)
    featurizedData = hashingTF.transform(wordsData)
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)
    rescaledData.persist()

    trainDF = rescaledData.select("features", "label").rdd.map(
        lambda x: Row(label=float(x['label']), features=Vectors.dense(x['features']))
    ).toDF()
    logistic = LogisticRegression(maxIter=70, regParam=0.01)
    model = logistic.fit(trainDF)

    testWordsData = tokenizer.transform(testing)
    testFeaturizedData = hashingTF.transform(testWordsData)
    testIDFModel = idf.fit(testFeaturizedData)
    testRescaledData = testIDFModel.transform(testFeaturizedData)
    testRescaledData.persist()

    testDF = testRescaledData.select("features", "label").rdd.map(
        lambda x: Row(label=float(x['label']), features=Vectors.dense(x['features']))
    ).toDF()
    predictions = model.transform(testDF)
    predictions.show()

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("The accuracy on test-set is " + str(accuracy))
	
    sc = spark.sparkContext

    tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=10000)
    idf = IDF(inputCol="rawFeatures", outputCol="features")

    srcdf = sc.textFile('file:///home/bigdata2019_group36/BigData/process_test.csv').map(parse_line)
    testing = srcdf.toDF()

    testWordsData = tokenizer.transform(testing)
    testFeaturizedData = hashingTF.transform(testWordsData)
    testIDFModel = idf.fit(testFeaturizedData)
    testRescaledData = testIDFModel.transform(testFeaturizedData)
    testRescaledData.persist()

    testDF = testRescaledData.select("features", "label").rdd.map(
        lambda x: Row(label=float(x['label']), features=Vectors.dense(x['features']))
    ).toDF()
    predictions = model.transform(testDF)
    predictions.show()
    predictions.select('prediction').write.csv(path='submit', header=True, sep=',', mode='overwrite')


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("Bigdata").getOrCreate()

    train(spark)

    spark.stop()
