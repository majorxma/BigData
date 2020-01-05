from pyspark.sql import SparkSession
from pyspark.sql import Row

from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import NaiveBayesModel
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import DecisionTreeClassificationModel
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import LinearSVCModel
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import sys,importlib

importlib.reload(sys)

def parse_line(p):
    cols = p.split(' ')
    label = cols[0]
    if label not in ('1', '0'):
        print("ERROR")
        return None

    label = 1.0 if label == '1' else 0.0
    fname = ' '.join(cols[1:])

    return Row(label=label, sentence=fname)

def train(spark):
    sc = spark.sparkContext
    tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=1000)
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    srcdf = sc.textFile('file:///home/bigdata2019_group36/BigData/input/train.csv').map(parse_line)
    srcdf = srcdf.toDF()
    srcdf.show()
    training, testing = srcdf.randomSplit([0.9, 0.1])

    wordsData = tokenizer.transform(training)
    featurizedData = hashingTF.transform(wordsData)
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)
    rescaledData.persist()

    trainDF = rescaledData.select("features", "label")  .rdd.map(
        lambda x: Row(label=float(x['label']), features=Vectors.dense(x['features']))
    ).toDF()
    trainDF.show()

    testWordsData = tokenizer.transform(testing)
    testFeaturizedData = hashingTF.transform(testWordsData)
    testIDFModel = idf.fit(testFeaturizedData)
    testRescaledData = testIDFModel.transform(testFeaturizedData)
    testRescaledData.persist()

    testDF = testRescaledData.select("features", "label").rdd.map(
        lambda x: Row(label=float(x['label']), features=Vectors.dense(x['features']))
    ).toDF()

    print("Learning begin!")


    sc = spark.sparkContext

    tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=8000)
    idf = IDF(inputCol="rawFeatures", outputCol="features")

    srcdf = sc.textFile('file:///home/bigdata2019_group36/BigData/test_new.csv').map(parse_line)
    testing = srcdf.toDF()

    testWordsData1 = tokenizer.transform(testing)
    testFeaturizedData1 = hashingTF.transform(testWordsData1)
    testIDFModel1 = idf.fit(testFeaturizedData1)
    testRescaledData1 = testIDFModel1.transform(testFeaturizedData1)
    testRescaledData1.persist()

    testingDF = testRescaledData1.select("features").rdd.map(
        lambda x: Row(features=Vectors.dense(x['features']))
    ).toDF()

    naivebayes = NaiveBayes(modelType="multinomial")
    nbModel = naivebayes.fit(trainDF)
    print("naive Bayes finished")
    nbPredictions = nbModel.transform(testDF)
    nbPredictions.show()
    nbEvaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = nbEvaluator.evaluate(nbPredictions)
    print("The accuracy on test-set is " + str(accuracy))
    nbPredictions1 = nbModel.transform(testingDF)
    nbPredictions1.select("prediction").write.mode("overwrite").options(header="true").csv("submitnbPredictions2.csv")

if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("Bigdata").getOrCreate()

    train(spark)

    spark.stop()


