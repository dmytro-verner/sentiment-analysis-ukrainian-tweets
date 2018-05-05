package analysis

import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.feature.HashingTF

import scala.util.Try

object UaTweetsSentiment {
  def main(args: Array[String]): Unit = {

    val positiveLabelWord = "добрий"
    val negativeLabelWord = "поганий"

    val conf = new SparkConf()
    conf.setMaster("local")
    conf.setAppName("Ua Tweets Sentiment Analysis")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    val tweetDF = sqlContext.read.json("src/main/resources/dobryi-vs-poganyi/aggregate.json")
    tweetDF.show()

    val messages = tweetDF.select("msg")
    println("Total messages: " + messages.count())

    val happyMessages = messages.filter(messages("msg").contains(positiveLabelWord))
    val countHappy = happyMessages.count()
    println("Number of happy messages: " +  countHappy)

    val unhappyMessages = messages.filter(messages("msg").contains(negativeLabelWord))
    val countUnhappy = unhappyMessages.count()
    println("Unhappy Messages: " + countUnhappy)

    val smallestCommonCount = Math.min(countHappy, countUnhappy).toInt

    val tweets = happyMessages.limit(smallestCommonCount).unionAll(unhappyMessages.limit(smallestCommonCount))

    val messagesRDD = tweets.rdd
    //filter out tweets that can't be parsed
    val positiveAndNegativeRecords = messagesRDD.map(
      row =>{
        Try{
          val msg = row(0).toString.toLowerCase()
          var isHappy:Int = 0
          if(msg.contains(negativeLabelWord)){
            isHappy = 0
          }else if(msg.contains(positiveLabelWord)){
            isHappy = 1
          }
          var messageSanitized = msg.replaceAll(positiveLabelWord, "")
          messageSanitized = messageSanitized.replaceAll(negativeLabelWord,"")

          (isHappy, messageSanitized.split(" ").toSeq) //tuple returned
        }
      }
    )

    //filter out exceptions
    val exceptions = positiveAndNegativeRecords.filter(_.isFailure)
    println("total records with exceptions: " + exceptions.count())
    exceptions.take(10).foreach(x => println(x.failed))

    val labeledTweets = positiveAndNegativeRecords.filter(_.isSuccess).map(_.get)
    println("total records with successes: " + labeledTweets.count())

    //dictionary size - correlates with the quantity of training set
    val hashingTF = new HashingTF(2000)

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

    var happyTotal = 0
    var happyCorrect = 0
    var unhappyTotal = 0
    var unhappyCorrect = 0
    results.foreach(
      r => {
        if (r._1 == 1) {
          happyTotal += 1
        } else if (r._1 == 0) {
          unhappyTotal += 1
        }
        if (r._1 == 1 && r._2 ==1) {
          happyCorrect += 1
        } else if (r._1 == 0 && r._2 == 0) {
          unhappyCorrect += 1
        }
      }
    )

    //calculate test error
    val testErrorTrainingSet = labelAndClassTrainingSet.filter(r => r._1 != r._2).count.toDouble / trainingData.count()

    //pull up the results for validation set
    val validSetResults = labelAndClassValidationSet.collect()

    var happyTotalValidSet = 0
    var unhappyTotalValidSet = 0
    var happyCorrectValidSet = 0
    var unhappyCorrectValidSet = 0
    validSetResults.foreach(
      r => {
        if (r._1 == 1) {
          happyTotalValidSet += 1
        } else if (r._1 == 0) {
          unhappyTotalValidSet += 1
        }
        if (r._1 == 1 && r._2 ==1) {
          happyCorrectValidSet += 1
        } else if (r._1 == 0 && r._2 == 0) {
          unhappyCorrectValidSet += 1
        }
      }
    )

    val testErrorValidationSet = labelAndClassValidationSet.filter(r => r._1 != r._2).count.toDouble / validationData.count()

    val predictions = sampleSet.map {
      point =>
        val classifiedValue = model.predict(point._1.features)
        (point._1.label, classifiedValue, point._2)
    }

    //the first value is the truth label. 1 is happy, 0 is unhappy.
    //class is the second value
    predictions.take(100).foreach(x => println("label: " + x._1 + " class: " + x._2 + " text: " + x._3.mkString(" ")))

    println("unhappy messages in Training Set: " + unhappyTotal + " happy messages: " + happyTotal)
    println("happy % correct: " + happyCorrect.toDouble/happyTotal)
    println("unhappy % correct: " + unhappyCorrect.toDouble/unhappyTotal)
    println("Test Error Training Set: " + testErrorTrainingSet)

    println("unhappy messages in Validation Set: " + unhappyTotalValidSet + " happy messages: " + happyTotalValidSet)
    println("happy % correct: " + happyCorrectValidSet.toDouble/happyTotalValidSet)
    println("unhappy % correct: " + unhappyCorrectValidSet.toDouble/unhappyTotalValidSet)
    println("Test Error Validation Set: " + testErrorValidationSet)
  }
}
