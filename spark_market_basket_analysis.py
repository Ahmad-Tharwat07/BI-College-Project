# --- HEADER: Import and Initialize Spark ---
from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth

# We must manually create the session here
spark = SparkSession.builder \
    .appName("MarketBasketAnalysis") \
    .master("local[*]") \
    .getOrCreate()

# Reduce the logging noise (optional, but makes output cleaner)
spark.sparkContext.setLogLevel("ERROR")

print("------------------------------------------------")
print("   STARTING MARKET BASKET ANALYSIS APP")
print("------------------------------------------------")

# --- BODY: The Logic ---

# 1. Create dummy data
data = [
    (1, ["Milk", "Bread", "Eggs"]),
    (2, ["Bread", "Eggs", "Cereal"]),
    (3, ["Milk", "Bread", "Eggs", "Cereal"]),
    (4, ["Milk", "Eggs"]),
    (5, ["Bread", "Cereal"]),
    (6, ["Milk", "Bread", "Cereal"])
]

df = spark.createDataFrame(data, ["id", "items"])

# 2. Define the Model
fp_growth = FPGrowth(itemsCol="items", minSupport=0.3, minConfidence=0.6)

# 3. Train the Model
model = fp_growth.fit(df)

# 4. Show Results
print("\n--- Most Frequent Itemsets ---")
model.freqItemsets.show()

print("\n--- Association Rules (If -> Then) ---")
model.associationRules.show()

print("------------------------------------------------")
print("   JOB FINISHED")
print("------------------------------------------------")

# --- FOOTER: Stop the Engine ---
spark.stop()
