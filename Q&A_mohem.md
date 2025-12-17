# ❓ Important Project Q&A (Q&A_mohem)

### 1. How did we calculate the Confidence?
**Formula**: Confidence(X -> Y) = $\frac{\text{Support}(X \cup Y)}{\text{Support}(X)}$
*   **Meaning**: Out of all the transactions that contain X, what percentage also contain Y?
*   **In Code**: We calculated the support of the combined itemset (X and Y) and divided it by the support of the antecedent (X) alone.

### 2. What are our "Rules"?
*   **Definition**: An Association Rule is a pattern that suggests that if certain items are bought (Antecedent), other specific items are likely to be bought as well (Consequent).
*   **Format**: `{Item A, Item B} -> {Item C}`
*   **Selection**: We generated all possible permutations from our frequent itemsets and kept only those where the **Confidence > Minimum Threshold**.

### 3. How did we choose our Minimum Support Threshold (0.05)?
*   **In Exams (Like the Image)**: The teacher gives you a high number (e.g., 40%) because the dataset is tiny (5 rows). If you used 5% on 5 rows, everything would be a pattern!
*   **In Our Project**: **WE chose it.**
    *   **Reasoning**: Our dataset has **7,500 transactions**.
    *   If we used 40% (like the slide), an item would need to appear 3,000 times! almost nothing is bought that often.
    *   We chose **0.05 (5%)** so we capture items bought at least 375 times. It's a "Data-Driven" decision based on the dataset size.

### 4. How did we choose the Minimum Confidence Threshold (0.2)?
*   **In Exams**: Given by the professor to ensure everyone gets the same answer.
*   **In Our Project**: **WE chose it via Trial & Error.**
    *   We started with **0.2 (20%)**.
    *   **Why?**: Retail data is "noisy". Human behavior is unpredictable. A 20% consistent pattern is actually quite strong in a grocery store.
    *   If we set it to 70% (like the slide), we found **Zero Rules**. So we lowered it until valid patterns emerged.

### 5. Why do we sort the final results by "Lift"?
*   **Reasoning**: 
    *   **Confidence** tells us the *reliability* of the rule (how often Y happens given X).
    *   **Lift** tells us the *strength of the correlation*.
        *   Lift > 1: Positive correlation (Buying X makes Y *more* likely).
        *   Lift = 1: No correlation (Independent events).
        *   Lift < 1: Negative correlation (Buying X makes Y *less* likely).
*   **Decision**: We sort by Lift to show the most **meaningful** connections first, rather than just the most frequent ones (which might just be popular items like "Mineral Water" that everyone buys anyway).

### 6. Why did we use Apache Spark for cleaning?
*   **Context**: The dataset (7,500 rows) is small enough for Excel or Pandas.
*   **Decision**: We used **PySpark** `clean_data_spark.py` to demonstrate **Big Data Engineering skills**. In a real-world scenario with millions of transactions (Walmart/Amazon scale), Pandas would crash, but Spark would handle it perfectly. We wanted to build a "scalable pipeline".

### 7. Why compare Apriori with FP-Growth?
*   **Reasoning**: Apriori is the classic algorithm, but it can be slow because it scans the database many times.
*   **Decision**: We added **FP-Growth** (Frequent Pattern Growth) as a benchmark.
    *   It uses a Tree structure (FP-Tree) and is generally much faster.
    *   By running both, we prove our "Scratch" implementation is correct (results match) and visually demonstrate the performance difference in the bar chart.

### 8. Final Recommendation: Which Algorithm is Best?
*   **Metric**: **Execution Time** (Speed).
*   **Winner**: **FP-Growth**.
*   **Why?**:
    *   **Apriori** must verify candidates at every step (scanning the database $k$ times).
    *   **FP-Growth** scans the database only **twice** (once to find frequencies, once to build the tree).
    *   Our results show FP-Growth is significantly faster (often 4x-5x faster) on this dataset.

### 9. Does FP-Growth give "better" rules (higher accuracy) than Apriori?
*   **Answer**: **NO.** The results are **Identical**.
*   **Reasoning**: Both algorithms are mathematically "Exact".
    *   If Apriori finds that `{Milk} -> {Eggs}` has 45% confidence, FP-Growth **MUST** find the exact same rule with 45% confidence.
    *   They are just two different *methods* to solve the same math problem.
    *   **Analogy**: It's like calculating `24 * 5`. You can do it in your head (FP-Growth) or write it down on paper (Apriori). The answer is 120 either way. The only difference is **Speed**.
*   **Metric**: Therefore, we **cannot** use accuracy or confidence as a comparison metric. We **must** use Execution Time.
