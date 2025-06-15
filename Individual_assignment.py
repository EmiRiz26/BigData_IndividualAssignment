from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, col, udf
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from textblob import TextBlob
from pyspark.sql.functions import when
from pyspark.sql.types import StringType, DoubleType
from pyspark.sql.functions import udf
from pyspark.sql.functions import month, year
import matplotlib.pyplot as plt
import pandas as pd

import os
os.environ['JAVA_HOME'] = '/opt/homebrew/opt/openjdk@11'

# # Initialize Spark session
# spark = SparkSession.builder \
#     .appName("SentimentAnalysis") \
#     .getOrCreate()

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("SentimentAnalysis") \
    .config("spark.hadoop.security.authentication", "simple") \
    .getOrCreate()

# Load the dataset
df = spark.read.csv("abcnews-date-text.csv", header=True)

# Convert 'publish_date' to date
df = df.withColumn("publish_date", to_date(col("publish_date").cast(StringType()), "yyyyMMdd"))

df.show(5)

# Define a UDF to extract sentiment
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

sentiment_udf = udf(get_sentiment, DoubleType())

# Apply sentiment analysis
df = df.withColumn("sentiment", sentiment_udf(col("headline_text")))

# Label sentiment as positive, negative, or neutral
df = df.withColumn(
    "label",
    when(col("sentiment") > 0, 1).when(col("sentiment") < 0, -1).otherwise(0)
)

df.show(5, truncate=False)

# Extract year and month
df = df.withColumn("year", year(df.publish_date))
df = df.withColumn("month", month(df.publish_date))

# Calculate average sentiment per month
monthly_sentiment = df.groupBy("year", "month").avg("sentiment")

# Order by year and month
monthly_sentiment = monthly_sentiment.orderBy("year", "month")

# Show the trend
monthly_sentiment.show()

# Convert to Pandas DataFrame
pandas_df = monthly_sentiment.toPandas()

# Plot
# Plot
plt.figure(figsize=(12, 6))
plt.plot(pandas_df['year'].astype(str) + '-' + pandas_df['month'].astype(str), pandas_df['avg(sentiment)'], marker='o')

# Set x-ticks to show every 6th label
xticks = range(0, len(pandas_df), 6)
plt.xticks(xticks, (pandas_df['year'].astype(str) + '-' + pandas_df['month'].astype(str)).iloc[xticks], rotation=45)

plt.title("Sentiment Trend Over Time")
plt.xlabel("Month-Year")
plt.ylabel("Average Sentiment")
plt.grid(True)
plt.tight_layout()
plt.show()

# Calculate average sentiment per year
yearly_sentiment = df.groupBy("year").avg("sentiment")

# Order by year
yearly_sentiment = yearly_sentiment.orderBy("year")

# Show the trend
yearly_sentiment.show()


# Convert to Pandas DataFrame
pandas_yearly_df = yearly_sentiment.toPandas()

# Plot
# Plot
plt.figure(figsize=(10, 5))
plt.plot(pandas_yearly_df['year'], pandas_yearly_df['avg(sentiment)'], marker='o')

# Convert year labels to integers
plt.xticks(ticks=pandas_yearly_df['year'], labels=pandas_yearly_df['year'].astype(int))

plt.title("Yearly Sentiment Trend Over Time")
plt.xlabel("Year")
plt.ylabel("Average Sentiment")
plt.grid(True)
plt.tight_layout()
plt.show()
