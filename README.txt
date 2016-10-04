
This is an example using Spark Machine Learning decision trees , written in Scala, 
to demonstrate How to get started with Spark ML on a MapR sandbox 

There are  1 datafile  in this directory :
	wbcd.csv 
 
You will need to copy these files to your MapR sandbox, or wherever you have Spark installed.

You can run these examples in the spark shell by putting the code from the scala file in the spark shell after launching:
 
$spark-shell 

Or you can run the applications with these steps:

Step 1: First compile the project: Select project  -> Run As -> Maven Install

Step 2: Copy the sparkflightmllab-1.0.jar to the sandbox 

To run the  standalone :

spark-submit --class example.Cancer --master yarn sparkcancermllab-1.0.jar


