@echo off
REM Generate sample data for testing
REM Usage: gendata.bat [local|hdfs]
REM   local = writes to ./sample_data/ (default)
REM   hdfs  = writes to hdfs:///data/

set JAVA_HOME=C:\Program Files\Java\jre1.8.0_501
set SPARK_HOME=C:\Users\tanay\Big_Data\spark-3.5.1-bin-hadoop3
set HADOOP_HOME=C:\Users\tanay\Big_Data\hadoop
set PATH=%SPARK_HOME%\bin;%HADOOP_HOME%\bin;%JAVA_HOME%\bin;%PATH%

set TARGET=%1
if "%TARGET%"=="" set TARGET=local

echo Generating sample data to: %TARGET%
spark-submit --master "local[*]" --driver-memory 4g --class neurotraffic.SampleDataGenerator target\scala-2.12\neurotraffic-graphx.jar %TARGET%
