# Kyle Hart
# Project: Commodities vs Weather
# Date: Dec 05, 2017
#
# Description: With commodity production and weather data for a number of countries (i.e., most of them) going back to 1960,
#              this program attempts to leverage a Naive Bayes Classifier to predict the quintile of a country's per-capita GDP.
#              A separate classifier for each commodity is fed a number of transactions, each having feature vectors including
#              the countries standardized precipitation level (zscore) and the increase or decrease of X commodity's
#              production for a given year, over the previous year, AS WELL AS labels indicating the quintile per-capita GDP.
#              After the classifiers are ran on each commodity, they are evaluated for accuracy, then saved to a csv.
#              The final product is a csv with each commodity, its model's accuracy, number of transactions used by the model,
#              and the number of countries that product that good.


from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.mllib.classification import LabeledPoint, NaiveBayes
import os


# Get Spark context
os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3'
config = SparkConf().setAppName("commodities model")
config = (config.setMaster('local[4]')
        .set('spark.executor.memory', '7G')
        .set('spark.driver.memory', '10G')
        .set('spark.driver.maxResultSize', '4G'))
sc = SparkContext(conf=config)
sc.setLogLevel(logLevel="OFF")
spark = SparkSession(sparkContext=sc)

# Load Dataframes
crops_df = spark.read.csv('data/crops.csv', header=True, inferSchema=True)
crops_df.createTempView("crops")
livestock_df = spark.read.csv('data/livestock.csv', header=True, inferSchema=True)
livestock_df.createTempView("livestock")
weather_df = spark.read.csv('data/global_weather.csv', header=True, inferSchema=True)
weather_df.createTempView("weather")
gdp_df = spark.read.csv('data/gdp.csv', header=True, inferSchema=True, nullValue="..")
gdp_df.createTempView("gdp")

# Get list of countries (NOAA data has least expansive list)
countries = spark.sql('''SELECT DISTINCT country FROM weather''').collect()

# Create dataframe with change, mean and stddev columns for crop production
print("Getting Crop Statistics...")
df = spark.sql('''
                SELECT 
                    year,
                    country, 
                    item,
                    production,
                    (CASE 
                        WHEN production >= LAG(production, 1) OVER (PARTITION BY country, item ORDER BY year)
                        THEN 1
                        ELSE 0
                    END) change, 
                    (SELECT mean(production)
                        FROM crops
                        WHERE 1=1
                            AND item=c.item
                            AND country=c.country) mean,
                    (SELECT stddev_pop(production)
                         FROM crops
                        WHERE 1=1
                            AND item=c.item
                            AND country=c.country) stddev
                FROM crops c
                WHERE 1=1
                    AND country NOT IN ('China', 'Russia', 'Yemen', 'Serbia', 'Vietnam', 'United Kingdom', 'France'I )
                ''')

# Add a zscore to each crop for each year of production
crops_norm = df.withColumn("zscore", (df.production - df.mean) / df.stddev)
crops_norm.createOrReplaceTempView("crops")

# Create dataframe with change, mean and stddev columns for livestock production
print("Getting Livestock Statistics...")
df = spark.sql('''
                SELECT 
                    lv.year,
                    lv.country, 
                    lv.item,
                    production,
                    (CASE 
                        WHEN production >= LAG(production, 1) OVER (PARTITION BY country, item ORDER BY year)
                        THEN 1
                        ELSE 0 
                    END) change, 
                    (SELECT mean(production)
                        FROM livestock
                        WHERE 1=1
                            AND item=lv.item
                            AND country=lv.country) mean,
                    (SELECT stddev_pop(production)
                         FROM livestock
                        WHERE 1=1
                            AND item=lv.item
                            AND country=lv.country) stddev
                FROM livestock lv
                ''')

# Add a zscore to each crop for each year of production
livestock_norm = df.withColumn("zscore", (df.production - df.mean) / df.stddev)
livestock_norm.createOrReplaceTempView("livestock")

# Create dataframe with mean and stddev columns for weather specifically precipitation
print("Getting Weather Data...\n")
df = spark.sql('''
                SELECT 
                    year,
                    country, 
                    PRCP,
                    (SELECT mean(PRCP)
                        FROM weather
                        WHERE 1=1
                            AND country=w.country) mean,
                    (SELECT stddev_pop(PRCP)
                         FROM weather
                        WHERE 1=1
                            AND country=w.country) stddev
                FROM weather w
                ''')
# Add a zscore to each country for each year of weather
weather_norm = df.withColumn("zscore", (df.PRCP - df.mean) / df.stddev)
weather_norm.createOrReplaceTempView("weather")

# Create dataframe with discrete income levels for gdp table
gdp_norm = spark.sql('''
                SELECT
                    country,
                    year,
                    amount,
                    (ntile(5) OVER (PARTITION BY year ORDER BY amount)) class
                FROM gdp
                ''')
gdp_norm.createOrReplaceTempView("gdp")


df = spark.sql('''
                SELECT 
                    livestock.country,
                    livestock.year,
                    livestock.item,
                    livestock.change,
                    weather.zscore,
                    gdp.class
                FROM livestock
                JOIN weather ON 1=1
                    AND livestock.country=weather.country
                    AND livestock.year=weather.year
                JOIN gdp ON 1=1
                    AND livestock.country=gdp.country
                    AND livestock.year=gdp.year
                
                UNION
                
                SELECT 
                    crops.country,
                    crops.year,
                    crops.item,
                    crops.change,
                    weather.zscore,
                    gdp.class
                FROM crops
                JOIN weather ON 1=1
                    AND crops.country=weather.country
                    AND crops.year=weather.year
                JOIN gdp ON 1=1
                    AND crops.country=gdp.country
                    AND crops.year=gdp.year
                ''')

df.createTempView("full_data")



print("Processing...")
# Get list of items
items = spark.sql('''SELECT DISTINCT item FROM crops UNION SELECT DISTINCT item FROM livestock''').collect()
results_list = []
insufficient_data = []
print("Dropping original tables from RAM...")

# Iterate through items run NaiveBayes on each with change and prcp as featues gdp as label
for item in items:
    print(".")
    # Get transactions for each item
    query = "SELECT * FROM full_data WHERE item = \"{}\"".format(item['item'])
    df = spark.sql(query)
    df.createOrReplaceTempView("temp")

    # Get counts of countries and transactions for each item
    countries = len(spark.sql('''SELECT DISTINCT country FROM temp''').collect())
    transactions = df.count()

    # Restrict items that have less then 25 countries producing or less than 1000 transactions in history
    if countries < 50 or transactions < 1000:
        insufficient_data.append(item['item'])
        continue

    data = []
    # Split transactions into labels and features (features: change, zscore.  label: class)
    for row in df.toLocalIterator():
        label = row[5]
        features = row[3:4]
        lp = LabeledPoint(label, features)
        data.append(lp)

    # Train model on transactions
    dataframe = sc.parallelize(data)
    training, test = dataframe.randomSplit([0.7, 0.3])
    model = NaiveBayes.train(training)

    # Find accuracy and determine max
    predictionAndLabel = test.map(lambda p: (model.predict(p.features), p.label))
    accuracy = 1.0 * predictionAndLabel.filter(lambda pl: pl[0] == pl[1]).count() / test.count()

    # Manual garbage collection ftw
    df.unpersist()
    dataframe.unpersist()
    # Add to results list
    results_list.append([item['item'], accuracy, transactions, countries])

print("\n\n")
results = spark.createDataFrame(results_list, schema=['item', 'accuracy', 'transactions', 'countries'])
results.createTempView('results')

# Show some results
spark.sql('''SELECT * FROM results''').show(n=100, truncate=False)
# Show the items with insufficient data
print("Insufficient data for ...\n", insufficient_data)
# Write csv to disk
results.write.csv(os.path.join('data', 'results_df'), header=True)
