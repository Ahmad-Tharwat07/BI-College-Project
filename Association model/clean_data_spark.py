from pyspark.sql import SparkSession
import shutil
import os

# ==========================================
# 🧹 SPARK DATA CLEANING PIPELINE
# ==========================================
# This script demonstrates "Big Data Preprocessing" by using Spark 
# to clean the raw dataset before analysis.
# ==========================================

def run_spark_cleaning():
    print("🚀 Starting Spark Cleaning Job...")
    
    # 1. Initialize Spark
    spark = SparkSession.builder \
        .appName("MarketBasketCleaning") \
        .master("local[*]") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("ERROR")
    
    # 2. Load Raw Data
    # We load as text because it's a "basket" format (no fixed columns)
    raw_path = "dataset.csv"
    output_single_file = "clean_market_data.csv"
    
    print(f"📂 Reading raw data from '{raw_path}'...")
    df = spark.read.text(raw_path)
    
    initial_count = df.count()
    print(f"📊 Initial Row Count: {initial_count}")
    
    # 3. Cleaning Steps
    # A. Remove empty rows (value is null or empty string)
    df_clean = df.filter(df.value.isNotNull() & (df.value != ""))
    
    # B. Remove Duplicate Transactions (if any)
    df_clean = df_clean.dropDuplicates()
    
    final_count = df_clean.count()
    print(f"✨ Cleaned Row Count: {final_count}")
    print(f"🗑️  Dropped {initial_count - final_count} garbage rows.")
    
    # 4. Save Output
    # Spark saves as a directory of parts. We'll coalesce to 1 part for easy CSV reading.
    temp_dir = "temp_clean_output"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        
    df_clean.coalesce(1).write.text(temp_dir)
    
    # 5. Post-Processing: Move the part file to a nice CSV name
    # Find the part-00000... file
    part_file = [f for f in os.listdir(temp_dir) if f.startswith("part-") and f.endswith(".txt")][0]
    
    if os.path.exists(output_single_file):
        os.remove(output_single_file)
        
    shutil.move(os.path.join(temp_dir, part_file), output_single_file)
    shutil.rmtree(temp_dir)
    
    print(f"✅ Data saved to '{output_single_file}'. Ready for Apriori!")
    spark.stop()

if __name__ == "__main__":
    run_spark_cleaning()
