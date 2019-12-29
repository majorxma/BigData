
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
    print("Learning begin!")

    decisionTree = DecisionTreeClassifier()
    dtModel = decisionTree.fit(trainDF)
    print("decision Tree finished")

    #naivebayes = NaiveBayes(modelType="multinomial")
    #nbModel = naivebayes.fit(trainDF)
    #print("naive Bayes finished")

    #randomForest = RandomForestClassifier(numTrees=10)
    #rfModel = randomForest.fit(trainDF)
    #print("random Forest finished")

    #gbt = GBTClassifier(maxIter=10, maxDepth=5)
    #gbtModel = gbt.fit(trainDF)
    print("GBT finished")

    #lr = LogisticRegression(maxIter=10, regParam=0.05)
    #lrModel = lr.fit(trainDF)
    #print("logistic Regression finished")

    testWordsData = tokenizer.transform(testing)
    testFeaturizedData = hashingTF.transform(testWordsData)
    testIDFModel = idf.fit(testFeaturizedData)
    testRescaledData = testIDFModel.transform(testFeaturizedData)
    testRescaledData.persist()

    testDF = testRescaledData.select("features", "label").rdd.map(
        lambda x: Row(label=float(x['label']), features=Vectors.dense(x['features']))
    ).toDF()

    dtPredictions = dtModel.transform(testDF)
    #nbPredictions = nbModel.transform(testDF)
    #rfPredictions = rfModel.transform(testDF)
    #gbtPredictions = gbtModel.transform(testDF)
    #lrPredictions = lrModel.transform(testDF)

    dtPredictions.show()
    #nbPredictions.show()
    #rfPredictions.show()
    #gbtPredictions.show()
    #lrPredictions.show()

    dtEvaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = dtEvaluator.evaluate(dtPredictions)
    print("The accuracy on test-set is " + str(accuracy))
    
    #nbEvaluator = MulticlassClassificationEvaluator(
    #    labelCol="label", predictionCol="prediction", metricName="accuracy")
    #accuracy = nbEvaluator.evaluate(predictions)
    #print("The accuracy on test-set is " + str(accuracy))
    
    #rfEvaluator = MulticlassClassificationEvaluator(
    #    labelCol="label", predictionCol="prediction", metricName="accuracy")
    #print("The accuracy on test-set is " + str(accuracy))
    
    #gbtEvaluator = MulticlassClassificationEvaluator(
    #    labelCol="label", predictionCol="prediction", metricName="accuracy")
    #print("The accuracy on test-set is " + str(accuracy))
    
    #lrEvaluator = MulticlassClassificationEvaluator(
    #    labelCol="label", predictionCol="prediction", metricName="accuracy")
    #print("The accuracy on test-set is " + str(accuracy))
    
    dtModel.save('dt_1')
    #nbModel.save('nb_1')
    #rfModel.save('rf_1')
    #gbtModel.save('gbt_1')
    #lrModel.save('lr_1')


def test(spark):
    sc = spark.sparkContext

    tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=8000)
    idf = IDF(inputCol="rawFeatures", outputCol="features")

    srcdf = sc.textFile('/input/test.csv')
    testing = srcdf.toDF()

    dtModel = DecisionTreeClassificationModel.load('dt_1')
    nbModel = DecisionTreeClassificationModel.load('nb_1')
    rfModel = DecisionTreeClassificationModel.load('rf_1')
    gbtModel = DecisionTreeClassificationModel.load('gbt_1')
    lrModel = DecisionTreeClassificationModel.load('lr_1')

    testWordsData = tokenizer.transform(testing)
    testFeaturizedData = hashingTF.transform(testWordsData)
    testIDFModel = idf.fit(testFeaturizedData)
    testRescaledData = testIDFModel.transform(testFeaturizedData)
    testRescaledData.persist()

    testDF = testRescaledData.select("features").rdd.map(
        lambda x: Row(features=Vectors.dense(x['features']))
    ).toDF()
    
    dtPredictions = dtModel.transform(testDF)
    nbPredictions = nbModel.transform(testDF)
    rfPredictions = rfModel.transform(testDF)
    gbtPredictions = gbtModel.transform(testDF)
    lrPredictions = lrModel.transform(testDF)

    num = dtPredictions.count()

    for i in range(num):
        one = 0
        zero = 0
        one = dtPredictions[i] + nbPredictions[i] + rfPredictions[i] + gbtPredictions[i] + lrPredictions[i]
        zero = 5 - one
        if zero > one:
            dtPredictions[i] = 0
        else:
            dtPredictions[i] = 1

    dtPredictions.select('prediction').write.csv(path='submit.csv', header=True, sep=',', mode='overwrite')


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("Bigdata").getOrCreate()

    train(spark)

    spark.stop()

