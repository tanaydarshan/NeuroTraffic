@echo off
REM NeuroTraffic GraphX — build and run helper
REM Usage:
REM   run.bat build           — compile + build fat JAR
REM   run.bat gendata         — generate sample data locally
REM   run.bat stats           — print graph statistics
REM   run.bat pagerank        — run PageRank, show top nodes
REM   run.bat timewindow      — compare rush hour vs midnight
REM   run.bat full            — full pipeline (PageRank + LPA + save features)

set JAVA_HOME=C:\Program Files\Java\jre1.8.0_501
set SPARK_HOME=C:\Users\tanay\Big_Data\spark-3.5.1-bin-hadoop3
set HADOOP_HOME=C:\Users\tanay\Big_Data\hadoop
set PATH=%SPARK_HOME%\bin;%HADOOP_HOME%\bin;%JAVA_HOME%\bin;%PATH%

set SBT="C:\Program Files (x86)\sbt\bin\sbt.bat"
set JAR=target\scala-2.12\neurotraffic-graphx.jar

if "%1"=="build" (
    echo Building fat JAR...
    %SBT% assembly
    goto :end
)

if not exist "%JAR%" (
    echo JAR not found. Building first...
    %SBT% assembly
)

echo Running: spark-submit --class neurotraffic.NeuroTrafficApp %JAR% %*
spark-submit --master "local[*]" --driver-memory 8g --class neurotraffic.NeuroTrafficApp %JAR% %*

:end
