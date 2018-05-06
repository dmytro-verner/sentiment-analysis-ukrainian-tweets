package analysis

import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.Try

object UaTweetsSentiment {

  val positiveLabelWord = "добре"
  val negativeLabelWord = "погано"

  //dictionary size - correlates with the quantity of training set
  val hashingTF = new HashingTF(2000)

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf()
    conf.setMaster("local")
    conf.setAppName("Ua Tweets Sentiment Analysis")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    val tweetDF = sqlContext.read.json("src/main/resources/dobre-vs-pogano/aggregate-new-structure.json")

    val messages = tweetDF.select("message", "isPositive")
    println("Total messages: " + messages.count())

    val positiveMessages = messages.filter(messages("isPositive").contains(true))
    val countPositive = positiveMessages.count()
    println("Number of positive messages: " +  countPositive)

    val negativeMessages = messages.filter(messages("isPositive").contains(false))
    val countNegative = negativeMessages.count()
    println("Number of negative messages: " + countNegative)

    val smallestCommonCount = Math.min(countPositive, countNegative).toInt

    val tweets = positiveMessages.limit(smallestCommonCount).unionAll(negativeMessages.limit(smallestCommonCount))

    val messagesRDD = tweets.rdd
    //filter out tweets that can't be parsed
    val positiveAndNegativeRecords = messagesRDD.map(
      row =>{
        Try{
          val msg = row(0).toString.toLowerCase()
          var isPositive:Int = 0
          if(msg.contains(negativeLabelWord)){
            isPositive = 0
          }else if(msg.contains(positiveLabelWord)){
            isPositive = 1
          }
          var messageSanitized = msg.replaceAll(positiveLabelWord, "")
          messageSanitized = messageSanitized.replaceAll(negativeLabelWord,"")

          (isPositive, messageSanitized.split(" ").toSeq) //tuple returned
        }
      }
    )

    //filter out exceptions
    val exceptions = positiveAndNegativeRecords.filter(_.isFailure)
    println("total records with exceptions: " + exceptions.count())
    exceptions.take(10).foreach(x => println(x.failed))

    val labeledTweets = positiveAndNegativeRecords.filter(_.isSuccess).map(_.get)
    println("total records with successes: " + labeledTweets.count())

    //Map the input strings to a tuple of labeled point + input text
    val inputLabeled = labeledTweets.map(
      t => (t._1, hashingTF.transform(t._2)))
      .map(x => new LabeledPoint(x._1.toDouble, x._2))

    val sampleSet = labeledTweets.take(1000).map(
      t => (t._1, hashingTF.transform(t._2), t._2))
      .map(x => (new LabeledPoint(x._1.toDouble, x._2), x._3))

    // split the data into training and validation sets (30% held out for validation testing)
    val splits = inputLabeled.randomSplit(Array(0.7, 0.3))
    val (trainingData, validationData) = (splits(0), splits(1))

    val boostingStrategy = BoostingStrategy.defaultParams("Classification")
    boostingStrategy.setNumIterations(20)
    boostingStrategy.treeStrategy.setNumClasses(2)
    boostingStrategy.treeStrategy.setMaxDepth(5)

    val model = GradientBoostedTrees.train(trainingData, boostingStrategy)

    // evaluate model on test instances and compute test error
    val labelAndClassTrainingSet = trainingData.map { point =>
      val prediction = model.predict(point.features)
      Tuple2(point.label, prediction)
    }

    val labelAndClassValidationSet = validationData.map { point =>
      val prediction = model.predict(point.features)
      Tuple2(point.label, prediction)
    }

    val results = labelAndClassTrainingSet.collect()

    var positiveTotal = 0
    var positiveCorrect = 0
    var negativeTotal = 0
    var negativeCorrect = 0
    results.foreach(
      r => {
        if (r._1 == 1) {
          positiveTotal += 1
        } else if (r._1 == 0) {
          negativeTotal += 1
        }
        if (r._1 == 1 && r._2 ==1) {
          positiveCorrect += 1
        } else if (r._1 == 0 && r._2 == 0) {
          negativeCorrect += 1
        }
      }
    )

    //calculate test error
    val testErrorTrainingSet = labelAndClassTrainingSet.filter(r => r._1 != r._2).count.toDouble / trainingData.count()

    //pull up the results for validation set
    val validSetResults = labelAndClassValidationSet.collect()

    var positiveTotalValidSet = 0
    var negativeTotalValidSet = 0
    var positiveCorrectValidSet = 0
    var negativeCorrectValidSet = 0
    validSetResults.foreach(
      r => {
        if (r._1 == 1) {
          positiveTotalValidSet += 1
        } else if (r._1 == 0) {
          negativeTotalValidSet += 1
        }
        if (r._1 == 1 && r._2 ==1) {
          positiveCorrectValidSet += 1
        } else if (r._1 == 0 && r._2 == 0) {
          negativeCorrectValidSet += 1
        }
      }
    )

    val testErrorValidationSet = labelAndClassValidationSet.filter(r => r._1 != r._2).count.toDouble / validationData.count()

    val predictions = sampleSet.map {
      point =>
        val classifiedValue = model.predict(point._1.features)
        (point._1.label, classifiedValue, point._2)
    }

    //the first value is the real class label. 1 is positive, 0 is negative.
    //class is the second value
    predictions.take(100).foreach(x => println("label: " + x._1 + " class: " + x._2 + " text: " + x._3.mkString(" ")))

    println("negative messages in Training Set: " + negativeTotal + " positive messages: " + positiveTotal)
    println("positive % correct: " + positiveCorrect.toDouble/positiveTotal)
    println("negative % correct: " + negativeCorrect.toDouble/negativeTotal)
    println("Test Error Training Set: " + testErrorTrainingSet)

    println("negative messages in Validation Set: " + negativeTotalValidSet + " positive messages: " + positiveTotalValidSet)
    println("positive % correct: " + positiveCorrectValidSet.toDouble/positiveTotalValidSet)
    println("negative % correct: " + negativeCorrectValidSet.toDouble/negativeTotalValidSet)
    println("Test Error Validation Set: " + testErrorValidationSet)
  }
}
