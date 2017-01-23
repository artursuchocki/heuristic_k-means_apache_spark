#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ---- K-means implementation using Python in Apache Spark ---- 

import os
import sys
os.chdir("/Users/suchy/Downloads/K_Means-2")
os.environ['SPARK_HOME'] = '/users/suchy/Documents/spark-2.1.0-bin-hadoop2.7'

os.curdir

# Create a variable for our root path
SPARK_HOME = os.environ['SPARK_HOME']

sys.path.insert(0,os.path.join(SPARK_HOME,"python"))
sys.path.insert(0,os.path.join(SPARK_HOME,"python","lib"))
sys.path.insert(0,os.path.join(SPARK_HOME,"python","lib","pyspark.zip"))
sys.path.insert(0,os.path.join(SPARK_HOME,"python","lib","py4j-0.10.4-src.zip"))

#Initialize SparkSession and SparkContext
from pyspark.sql import SparkSession
from pyspark import SparkContext

#Create a Spark Session
SpSession = SparkSession \
    .builder \
    .master("local[*]") \
    .appName("K-Means implementation") \
    .config("spark.driver.memory", "6g") \
    .config("spark.executor.memory", "1g") \
    .config("spark.cores.max","2") \
    .config("spark.sql.warehouse.dir", "file://///users/suchy/Documents/spark-2.1.0-bin-hadoop2.7/temp/spark-warehouse")\
    .getOrCreate()
    
#Get the Spark Context from Spark Session    
SpContext = SpSession.sparkContext

dataLines = SpContext.textFile("file:///Users/suchy/Downloads/K_Means-2/data.txt")

dataLines.count()

from pyspark.sql import Row

import math
from pyspark.ml.linalg import Vectors

    
#Convert to Local Vector.
def transformToNumeric( inputStr) :
    attList=inputStr.split()

    values=      Row(c0=float(attList[0]),  \
    		c1=float(attList[1]),  \
    		c2=float(attList[2]),  \
    		c3=float(attList[3]),  \
    		c4=float(attList[4]),  \
    		c5=float(attList[5]),  \
    		c6=float(attList[6]),  \
    		c7=float(attList[7]),  \
    		c8=float(attList[8]),  \
    		c9=float(attList[9]),  \
    		c10=float(attList[10]),  \
    		c11=float(attList[11]),  \
    		c12=float(attList[12]),  \
		c13=float(attList[13]),  \
		c14=float(attList[14]),  \
		c15=float(attList[15]),  \
		c16=float(attList[16]),  \
		c17=float(attList[17]),  \
		c18=float(attList[18]),  \
		c19=float(attList[19]),  \
		c20=float(attList[20]),  \
		c21=float(attList[21]),  \
		c22=float(attList[22]),  \
		c23=float(attList[23]),  \
		c24=float(attList[24]),  \
		c25=float(attList[25]),  \
		c26=float(attList[26]),  \
		c27=float(attList[27]),  \
		c28=float(attList[28]),  \
		c29=float(attList[29]),  \
		c30=float(attList[30]),  \
		c31=float(attList[31]),  \
		c32=float(attList[32]),  \
		c33=float(attList[33]),  \
		c34=float(attList[34]),  \
		c35=float(attList[35]),  \
		c36=float(attList[36]),  \
		c37=float(attList[37]),  \
		c38=float(attList[38]),  \
		c39=float(attList[39]),  \
		c40=float(attList[40]),  \
		c41=float(attList[41]),  \
		c42=float(attList[42]),  \
		c43=float(attList[43]),  \
		c44=float(attList[44]),  \
		c45=float(attList[45]),  \
		c46=float(attList[46]),  \
		c47=float(attList[47]),  \
		c48=float(attList[48]),  \
		c49=float(attList[49]),  \
		c50=float(attList[50]),  \
		c51=float(attList[51]),  \
		c52=float(attList[52]),  \
		c53=float(attList[53]),  \
		c54=float(attList[54])		
             )
    return values

numericData = dataLines.map(transformToNumeric)
numericData.collect()

dataDf = SpSession.createDataFrame(numericData)


#Centering and scaling. To perform this every value should be subtracted
#from that column's mean and divided by its Std. Deviation.

summStats=dataDf.describe().toPandas()
meanValues=summStats.iloc[1,1:55].values.tolist()
stdValues=summStats.iloc[2,1:55].values.tolist()

#place the means and std.dev values in a broadcast variable
bcMeans=SpContext.broadcast(meanValues)
bcStdDev=SpContext.broadcast(stdValues)

def centerAndScale(inRow) :
    global bcMeans
    global bcStdDev
    
    meanArray=bcMeans.value
    stdArray=bcStdDev.value

    retArray=[]
    for i in range(len(meanArray)):
        retArray.append( (float(inRow[i]) - float(meanArray[i])) /\
            float(stdArray[i]) )
    return Vectors.dense(retArray)
    
dataNormalized = dataDf.rdd.map(centerAndScale)
dataNormalized.collect()

#Create a Spark Data Frame
dataRows=dataNormalized.map( lambda f:Row(features=f))
finalDf = SpSession.createDataFrame(dataRows)

finalDf.select("features").show(10)

from pyspark.ml.clustering import KMeans
kmeans = KMeans(k=50, seed=10)
model = kmeans.fit(finalDf)
predictions = model.transform(finalDf)
predictions.select("prediction").take(360)
predictions.groupBy("prediction").count().sort("count", ascending=False).show(51)
#Plot the results in a scatter plot
import pandas as pd


#sample plot according to chosen 16th and 20th feature
def unstripData(instr) :
    return ( instr["prediction"], instr["features"][0], \
        instr["features"][1],instr["features"][15],instr["features"][19])
    
unstripped=predictions.rdd.map(unstripData)
predList=unstripped.collect()
predPd = pd.DataFrame(predList)

import matplotlib.pylab as plt
plt.cla()
plt.scatter(predPd[3],predPd[4], c=predPd[0])