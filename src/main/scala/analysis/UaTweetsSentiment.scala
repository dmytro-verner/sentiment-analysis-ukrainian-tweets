package analysis

import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.Try

object UaTweetsSentiment {
  def main(args: Array[String]): Unit = {

    val conf = new SparkConf()
    conf.setMaster("local")
    conf.setAppName("Ua Tweets Sentiment Analysis")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    val tweetDF = sqlContext.read.json("src/main/resources/tweets-up-to-20-04-unredacted.json")
    tweetDF.show()

    val messages = tweetDF.select("msg")
    println("Total messages: " + messages.count())

    val happyMessages = messages.filter(messages("msg").contains("добре"))
    val countHappy = happyMessages.count()
    println("Number of happy messages: " +  countHappy)

    val unhappyMessages = messages.filter(messages("msg").contains("погано"))
    val countUnhappy = unhappyMessages.count()
    println("Unhappy Messages: " + countUnhappy)

    val smallest = Math.min(countHappy, countUnhappy).toInt

    //Create a dataset with equal parts happy and unhappy messages
    val tweets = happyMessages.limit(smallest).unionAll(unhappyMessages.limit(smallest))

    val messagesRDD = tweets.rdd
    //We use scala's Try to filter out tweets that couldn't be parsed
    val goodBadRecords = messagesRDD.map(
      row =>{
        Try{
          val msg = row(0).toString.toLowerCase()
          var isHappy:Int = 0
          if(msg.contains("погано")){
            isHappy = 0
          }else if(msg.contains("добре")){
            isHappy = 1
          }
          var msgSanitized = msg.replaceAll("добре", "")
          msgSanitized = msgSanitized.replaceAll("погано","")
          //Return a tuple
          (isHappy, msgSanitized.split(" ").toSeq)
        }
      }
    )

    //We use this syntax to filter out exceptions
    val exceptions = goodBadRecords.filter(_.isFailure)
    println("total records with exceptions: " + exceptions.count())
    exceptions.take(10).foreach(x => println(x.failed))
    val labeledTweets = goodBadRecords.filter(_.isSuccess).map(_.get)
    println("total records with successes: " + labeledTweets.count())

    //transform data
    val hashingTF = new HashingTF(2000)

    //Map the input strings to a tuple of labeled point + input text
    val input_labeled = labeledTweets.map(
      t => (t._1, hashingTF.transform(t._2)))
      .map(x => new LabeledPoint(x._1.toDouble, x._2))

    //We're keeping the raw text for inspection later
    val sample = labeledTweets.take(1000).map(
      t => (t._1, hashingTF.transform(t._2), t._2))
      .map(x => (new LabeledPoint(x._1.toDouble, x._2), x._3))

    // Split the data into training and validation sets (30% held out for validation testing)
    val splits = input_labeled.randomSplit(Array(0.7, 0.3))
    val (trainingData, validationData) = (splits(0), splits(1))

    val boostingStrategy = BoostingStrategy.defaultParams("Classification")
    boostingStrategy.setNumIterations(20) //number of passes over our training data
    boostingStrategy.treeStrategy.setNumClasses(2) //We have two output classes: happy and sad
    boostingStrategy.treeStrategy.setMaxDepth(5)
    //Depth of each tree. Higher numbers mean more parameters, which can cause overfitting.
    //Lower numbers create a simpler model, which can be more accurate.
    //In practice you have to tweak this number to find the best value.

    val model = GradientBoostedTrees.train(trainingData, boostingStrategy)

    // Evaluate model on test instances and compute test error
    val labelAndPredsTrain = trainingData.map { point =>
      val prediction = model.predict(point.features)
      Tuple2(point.label, prediction)
    }

    val labelAndPredsValid = validationData.map { point =>
      val prediction = model.predict(point.features)
      Tuple2(point.label, prediction)
    }

    //Since Spark has done the heavy lifting already, lets pull the results back to the driver machine.
    //Calling collect() will bring the results to a single machine (the driver) and will convert it to a Scala array.
    //Start with the Training Set
    val results = labelAndPredsTrain.collect()

    var happyTotal = 0
    var unhappyTotal = 0
    var happyCorrect = 0
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

    val testErr = labelAndPredsTrain.filter(r => r._1 != r._2).count.toDouble / trainingData.count()

    //Compute error for validation Set
    val validSetResults = labelAndPredsValid.collect()

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

    val testErrValidSet = labelAndPredsValid.filter(r => r._1 != r._2).count.toDouble / validationData.count()


    val predictions = sample.map { point =>
      val prediction = model.predict(point._1.features)
      (point._1.label, prediction, point._2)
    }

    //The first entry is the true label. 1 is happy, 0 is unhappy.
    //The second entry is the prediction.
    predictions.take(100).foreach(x => println("label: " + x._1 + " prediction: " + x._2 + " text: " + x._3.mkString(" ")))

    println("unhappy messages in Training Set: " + unhappyTotal + " happy messages: " + happyTotal)
    println("happy % correct: " + happyCorrect.toDouble/happyTotal)
    println("unhappy % correct: " + unhappyCorrect.toDouble/unhappyTotal)
    println("Test Error Training Set: " + testErr)

    println("unhappy messages in Validation Set: " + unhappyTotalValidSet + " happy messages: " + happyTotalValidSet)
    println("happy % correct: " + happyCorrectValidSet.toDouble/happyTotalValidSet)
    println("unhappy % correct: " + unhappyCorrectValidSet.toDouble/unhappyTotalValidSet)
    println("Test Error Validation Set: " + testErrValidSet)
  }
}
