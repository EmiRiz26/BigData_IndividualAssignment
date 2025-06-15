# Big Data individual assignment

This project performs sentiment analysis on news headlines from the ABC News dataset using Apache Spark. The goal is to visualize the sentiment trend over time, both monthly and yearly.

## Requirements

To run this project, ensure you have the following installed:

- Apache Spark
- Python 3.x
- TextBlob (Python package)
- PySpark (Python package)
- Matplotlib (Python package)
- Pandas (Python package)
- Java 11 

## Code Explanation

1. **Setup**: Initialize a Spark session.
2. **Loading Data**: Load the CSV dataset into a Spark DataFrame.
3. **Data Transformation**: 
   - Convert the `publish_date` to a date format.
   - Calculate sentiment scores using TextBlob.
   - Label each headline as positive, negative, or neutral.
   - Extract year and month from the publish date.
4. **Analysis**:
   - Aggregate sentiment scores to find average sentiment per month and year.
5. **Visualization**:
   - Plot the monthly and yearly average sentiment trends using Matplotlib.
