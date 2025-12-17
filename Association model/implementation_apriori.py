# ==========================================
# Source: https://www.kaggle.com/datasets/shwetabh123/market-basket-optimization
# ==========================================

import itertools
import csv
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth

# ==========================================
# 1. SETTINGS & DATASET
# ==========================================
MIN_SUPPORT = 0.05     # Adjusted for real datasets
MIN_CONFIDENCE = 0.2
CSV_FILE = "clean_market_data.csv"
RESULTS_DIR = "results"

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def load_dataset_from_csv(filename):
    """
    Reads a CSV file where each row is a transaction.
    """
    dataset = []
    if not os.path.exists(filename):
        print(f"❌ Error: File '{filename}' not found.")
        print("💡 Please run 'clean_data_spark.py' first to generate this file!")
        exit(1)

    print(f"📂 Loading data from '{filename}'...")
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            clean_row = [item for item in row if item.strip()]
            if clean_row:
                dataset.append(clean_row)
    print(f"✅ Loaded {len(dataset)} transactions.")
    return dataset

# ==========================================
# 2. HELPER FUNCTIONS (Apriori Scratch)
# ==========================================

def calculate_support(itemset, dataset):
    count = 0
    itemset_set = set(itemset)
    for transaction in dataset:
        if itemset_set.issubset(set(transaction)):
            count += 1
    return count / len(dataset)

def get_unique_items(dataset):
    unique_items = set()
    for transaction in dataset:
        for item in transaction:
            unique_items.add(item)
    return sorted(list(unique_items))

# ==========================================
# 3. ALGORITHM 1: APRIORI (FROM SCRATCH)
# ==========================================

def run_apriori_scratch(dataset, min_support):
    print("\n Starting Apriori Algorithm (From Scratch)...")
    start_time = time.time()

    all_frequent_itemsets = []

    # Phase 1: Frequent k=1
    print("\n📋 TABLE: Frequent 1-Itemsets (Candidates)")
    print(f"{'Itemset':<40} | {'Support':<10}")
    print("-" * 55)
    candidates = get_unique_items(dataset)
    current_frequent_items = [] # A list for items that pass the support test

    for item in candidates:
        item_tuple = (item,)
        support = calculate_support(item_tuple, dataset)
        if support >= min_support:
            current_frequent_items.append(item_tuple)
            all_frequent_itemsets.append((item_tuple, support))
            print(f"{str(item_tuple):<40} | {support:.4f}")

    print(f"✅ Found {len(current_frequent_items)} frequent single items.")

    # Phase 2: Frequent k=2+
    k = 2
    while True:
        if len(current_frequent_items) == 0:
            break

        unique_items_in_layer = set()
        for itemset in current_frequent_items:
            for item in itemset:
                unique_items_in_layer.add(item)
        sorted_pool = sorted(list(unique_items_in_layer))
        candidates = list(itertools.combinations(sorted_pool, k))

        new_frequent_layer = []

        print(f"\n📋 TABLE: Frequent {k}-Itemsets")
        print(f"{'Itemset':<40} | {'Support':<10}")
        print("-" * 55)

        for candidate in candidates:
            support = calculate_support(candidate, dataset)
            if support >= min_support:
                new_frequent_layer.append(candidate)
                all_frequent_itemsets.append((candidate, support))
                print(f"{str(candidate):<40} | {support:.4f}")

        if not new_frequent_layer:
            print("[No more frequent itemsets found]")
            break

        print(f"✅ Found {len(new_frequent_layer)} frequent {k}-itemsets.")
        current_frequent_items = new_frequent_layer
        k += 1

    duration = time.time() - start_time
    print(f"\n✅ Apriori (Scratch) finished in {duration:.4f} seconds.")
    return all_frequent_itemsets, duration

# ==========================================
# 4. ALGORITHM 2: FP-GROWTH (LIBRARY)
# ==========================================

def run_fpgrowth_library(dataset, min_support):
    print("\n🐇 Starting FP-Growth Algorithm (Using Library)...")
    start_time = time.time()
    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_sets_fp = fpgrowth(df, min_support=min_support, use_colnames=True)
    duration = time.time() - start_time
    print(f"✅ FP-Growth (Library) finished in {duration:.4f} seconds.")
    return frequent_sets_fp, duration

# ==========================================
# 5. GENERATE RULES (Manual)
# ==========================================

def generate_rules(frequent_itemsets, dataset, min_confidence):
    print("\n📜 Evaluating Rules (Accepted vs Rejected)...")

    rules_found = []

    # Header for the verbose output
    print(f"{'Status':<10} | {'Rule':<50} | {'Conf':<6}")
    print("-" * 75)

    for itemset, support_itemset in frequent_itemsets:
        if len(itemset) > 1:
            permutations = list(itertools.permutations(itemset))
            for perm in permutations:
                for i in range(1, len(perm)):
                    antecedent = perm[:i]
                    consequent = perm[i:]

                    support_antecedent = calculate_support(antecedent, dataset)

                    # Avoid division by zero
                    if support_antecedent == 0:
                        continue

                    confidence = support_itemset / support_antecedent

                    rule_str = f"{list(antecedent)} -> {list(consequent)}"

                    if confidence >= min_confidence:
                        print(f"✅ [PASS]   | {rule_str:<50} | {confidence:.2f}")

                        support_consequent = calculate_support(consequent, dataset)
                        lift = confidence / support_consequent

                        rules_found.append({
                            'rule': rule_str,
                            'confidence': confidence,
                            'lift': lift,
                            'support': support_itemset
                        })
                    else:
                        print(f"❌ [FAIL]   | {rule_str:<50} | {confidence:.2f}")

    # Sort by Lift
    rules_found.sort(key=lambda x: x['lift'], reverse=True)

    print("\n⭐ Final Strong Rules Table:")
    print(f"{'Rule':<60} | {'Conf':<8} | {'Lift':<8}")
    print("-" * 85)
    for r in rules_found[:15]:
        print(f"{r['rule']:<60} | {r['confidence']:.2f}     | {r['lift']:.2f}")

    return rules_found

# ==========================================
# 6. VISUALIZATION & SAVING
# ==========================================

def plot_rules_and_save(rules):
    if not rules:
        return

    top_rules = rules[:10]
    names = [r['rule'] for r in top_rules]
    confidences = [r['confidence'] for r in top_rules]

    plt.figure(figsize=(12, 6))
    plt.barh(names, confidences, color='skyblue')
    plt.xlabel('Confidence')
    plt.title('Top 10 Strongest Rules (Apriori)')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    filename = os.path.join(RESULTS_DIR, "top_rules_confidence.png")
    plt.savefig(filename)
    print(f"\n💾 Saved Rule Plot to: {filename}")
    plt.show()

def plot_comparison_and_save(t_apriori, t_fpgrowth):
    algorithms = ['Apriori (Scratch)', 'FP-Growth (Lib)']
    times = [t_apriori, t_fpgrowth]

    plt.figure(figsize=(8, 5))
    colors = ['#FF6B6B', '#4ECDC4'] # Red vs Teal
    bars = plt.bar(algorithms, times, color=colors)

    plt.ylabel('Seconds (Lower is Better)')
    plt.title('Performance Comparison')

    # Add values on top
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.4f}s", va='bottom', ha='center')

    filename = os.path.join(RESULTS_DIR, "performance_comparison.png")
    plt.savefig(filename)
    print(f"� Saved Comparison Plot to: {filename}")
    plt.show()

# ==========================================
# 7. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Load Data
    data = load_dataset_from_csv(CSV_FILE)

    # 2. Run Apriori (Scratch)
    freq_items_ap, time_ap = run_apriori_scratch(data, MIN_SUPPORT)

    # 3. Reference: Run FP-Growth (Library) for comparison
    freq_items_fp, time_fp = run_fpgrowth_library(data, MIN_SUPPORT)

    # 4. Generate Rules (using Apriori results)
    final_rules = generate_rules(freq_items_ap, data, MIN_CONFIDENCE)

    # 5. Visualizations
    plot_rules_and_save(final_rules)
    plot_comparison_and_save(time_ap, time_fp)

    # 6. Final Recommendation
    print("\n" + "="*50)
    print("🏆 FINAL RECOMMENDATION")
    print("="*50)

    speedup = time_ap / time_fp
    print(f"Metrics Used: Execution Time & Scalability")
    print(f"1. Apriori Time:   {time_ap:.4f}s")
    print(f"2. FP-Growth Time: {time_fp:.4f}s")
    print(f"⚡ Speedup Factor: FP-Growth is {speedup:.2f}x faster!")

    if time_fp < time_ap:
        print("\n✅ RESULT: We recommend **FP-Growth**.")
        print("   Reason: It is significantly faster because it avoids Candidate Generation")
        print("   and uses an efficient FP-Tree structure, requiring only 2 passes over the data.")
    else:
        print("\n✅ RESULT: Apriori was faster (likely due to small dataset overhead).")

    print("\n✅ Analysis Complete. Check the 'results' folder for images.")