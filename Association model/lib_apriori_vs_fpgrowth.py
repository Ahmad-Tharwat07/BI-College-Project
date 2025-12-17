import pandas as pd
import time
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

# ==========================================
# 1. PREPARE THE DATA
# ==========================================
print("🛒 Step 1: Loading Data...")

# OPTION A: Use this dummy data to test the code immediately
dataset = [
    ['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
    ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
    ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
    ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
    ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs'],
    ['Milk', 'Unicorn', 'Corn', 'Yogurt'],
    ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
    ['Corn', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs'],
    ['Milk', 'Donut', 'Eggs'],
    ['Milk', 'Onion', 'Corn', 'Kidney Beans', 'Yogurt', 'Eggs'],
    ['Apple', 'Cigarettes', 'Kidney Beans', 'Eggs'],
    ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
]

# OPTION B: If you download a dataset from Kaggle, uncomment lines below:
# df_kaggle = pd.read_csv('your_dataset.csv')
# dataset = df_kaggle.values.tolist() # Convert dataframe to list of lists

# --- Preprocessing ---
# The algorithms need a "One-Hot Encoded" format (True/False for every item)
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

print(f"✅ Data processed! We have {len(dataset)} transactions and {len(df.columns)} unique items.")
print("-" * 50)


# ==========================================
# 2. RUN ALGORITHM 1: APRIORI
# ==========================================
print("🐢 Step 2: Running Apriori Algorithm...")

start_time = time.time()

# min_support=0.3 means an itemset must appear in at least 30% of transactions
frequent_itemsets_apriori = apriori(df, min_support=0.3, use_colnames=True)

apriori_time = time.time() - start_time
print(f"✅ Apriori finished in {apriori_time:.6f} seconds.")
print("-" * 50)


# ==========================================
# 3. RUN ALGORITHM 2: FP-GROWTH
# ==========================================
print("🐇 Step 3: Running FP-Growth Algorithm...")

start_time = time.time()

frequent_itemsets_fp = fpgrowth(df, min_support=0.3, use_colnames=True)

fpgrowth_time = time.time() - start_time
print(f"✅ FP-Growth finished in {fpgrowth_time:.6f} seconds.")
print("-" * 50)


# ==========================================
# 4. GENERATE RULES & EVALUATE
# ==========================================
print("📊 Step 4: Generating Association Rules (from Apriori results)...")

# We look for rules with a minimum 'confidence' of 70%
rules = association_rules(frequent_itemsets_apriori, metric="confidence", min_threshold=0.7)

# Let's filter columns to make it readable
view_rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

print("\nTop 5 Strongest Rules Discovered:")
print(view_rules.head())
print("\nInterpretation Example:")
print("If the first row says {Eggs} -> {Kidney Beans}, it means people who buy Eggs likely buy Kidney Beans.")


# ==========================================
# 5. VISUALIZE COMPARISON
# ==========================================
print("\n📉 Step 5: Visualizing Performance...")

algorithms = ['Apriori', 'FP-Growth']
times = [apriori_time, fpgrowth_time]

plt.figure(figsize=(8, 5))
plt.bar(algorithms, times, color=['salmon', 'skyblue'])
plt.ylabel('Execution Time (Seconds)')
plt.title('Algorithm Comparison: Apriori vs FP-Growth')
plt.show()

print("\n💡 RECOMMENDATION:")
if fpgrowth_time < apriori_time:
    print("Suggest using **FP-Growth**. It was faster and is generally more memory efficient for large datasets.")
else:
    print("Suggest using **Apriori**. On this tiny dataset, it was faster, but for Big Data, FP-Growth usually wins.")
