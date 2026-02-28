# 🌳 Decision Trees 
---

## 1. What is a Decision Tree?

A **Decision Tree** is a supervised machine learning model that makes predictions using a sequence of **IF–ELSE rules** learned from data.

It mimics human decision-making:

**Example: Loan Approval**

```
IF Income > 60k:
    IF CreditScore > 700:
        IF Debt < 10k:
            APPROVE
        ELSE:
            REVIEW
    ELSE:
        REJECT
ELSE:
    REJECT
```

The model automatically learns:

* Which feature to check first
* What threshold to use
* In what order to apply rules

---

## 2. Tree Structure & Terminology

![Image](https://images.openai.com/static-rsc-3/4I3x11hOXIvss3gkDFVqChfolOGVph1-N96UfzBU0Oy0P_XVfQVSoex6TMmXGxZ_DZJALr3WFW26oTcqz8nIoOjQ1EquVAy1hGDlVt-AbAQ?purpose=fullsize\&v=1)

![Image](https://mljar.com/blog/visualize-decision-tree/output_15_0.svg)

![Image](https://images.openai.com/static-rsc-3/1dkR9kvb99D09fHyKjL7H2YFbnX6gAHa1O_9e_52ynR4nHgns1FIDnHwiAkBycIOlaAOEEePpkUmhs18uyDUCZZ5uWcpS0faRuoVdtfQCc0?purpose=fullsize\&v=1)

![Image](https://images.openai.com/static-rsc-3/nhVh6siWmMZZxcSBavk-FVSqCEC6xWAdIOsOP9-nvAucbRKmXpECXvNaXv27t_qKBPfyP7jkAFC1kZMEJ-4Evh9Ms11q5xNrHlESqsgUGdA?purpose=fullsize\&v=1)

| Component     | Meaning                                   |
| ------------- | ----------------------------------------- |
| Root Node     | First split (most important feature)      |
| Decision Node | Internal node where data is split further |
| Branch        | Outcome of a condition                    |
| Leaf Node     | Final prediction                          |

### What a node contains (scikit-learn style)

Example:

```
CreditScore <= 715
gini = 0.469
samples = 8
value = [3, 5]
class = Yes
```

Meaning:

* Split condition
* Impurity of the node
* Number of samples
* Class distribution
* Predicted class (majority)

---

## 3. How Prediction Works

A new sample travels from **root → leaf**.

Example:

Input:

* Income = 72k
* CreditScore = 710
* Debt = 12k

Path:

```
Income > 60k → Yes
CreditScore > 700 → Yes
Debt < 10k → No
→ Prediction: REVIEW
```

### Probabilities

Leaf stores class distribution:

* If leaf has 18 Yes, 2 No
  → P(Yes) = 0.9

`predict_proba()` returns these probabilities.

---

## 4. Running Example Dataset

Loan default prediction:

| ID | CreditScore | Income_k | Debt_k | Default |
| -- | ----------- | -------- | ------ | ------- |
| 1  | 780         | 90       | 5      | No      |
| 2  | 760         | 85       | 8      | No      |
| 3  | 720         | 70       | 12     | No      |
| 4  | 710         | 65       | 18     | Yes     |
| 5  | 690         | 60       | 20     | Yes     |
| 6  | 680         | 55       | 25     | Yes     |
| 7  | 650         | 50       | 30     | Yes     |
| 8  | 630         | 45       | 35     | Yes     |

Observation:

* High credit score → No default
* Low credit score → Yes default

A good split separates these groups.

---

## 5. How Trees Decide Splits (Core Idea)

Goal: Create **pure nodes**

Pure = only one class
Impure = mixed classes

The algorithm:

1. Try all features
2. Try multiple thresholds
3. Calculate impurity after split
4. Choose split with **maximum impurity reduction**

---

## 6. Impurity Measures

### 6.1 Gini Impurity (Default)

[
Gini = 1 - \sum p_i^2
]

Example: 8 Yes, 2 No

[
p_{yes}=0.8,; p_{no}=0.2
]
[
Gini = 1 - (0.64 + 0.04) = 0.32
]

Interpretation:

* 0 → pure
* 0.5 → maximum impurity (binary)

---

### 6.2 Entropy

[
Entropy = -\sum p_i \log_2(p_i)
]

Same example:

[
Entropy \approx 0.72
]

Properties:

* 0 → pure
* 1 → maximum impurity (binary)

### Gini vs Entropy

![Gini vs Entropy Graph](./gini_vs_entropy.png)

*(Note: In the graph above, Entropy is often scaled by half to easily compare its curve shape directly with Gini Impurity. Both reach their minimum at probability 0.0 and 1.0 (pure nodes), and their maximum when classes are perfectly balanced.)*

| Metric  | Max value | properties |
| ------- | --------- | ---------- |
| Gini    | 0.5       | Faster to calculate (no logs). Slightly favors larger partitions (tends to isolate the most frequent class). |
| Entropy | 1         | Slower (logarithm). Slightly penalizes unbalanced splits, preferring more balanced trees. |

**Which one to use?**
In practice, they perform very similarly (~2% difference in resulting trees). Gini is the default in scikit-learn simply because it's computationally cheaper.

---

## 7. Information Gain

The tree chooses splits that reduce impurity.

[
Information\ Gain = Impurity(parent) - Weighted\ impurity(children)
]

Weighted impurity:

[
\sum \frac{n_k}{n} \times Impurity(child_k)
]

### Example: Categorical Target & Categorical Split Feature

Understanding how a decision tree splits data when both the feature and the target are categorical.
**Scenario**: Predict if a customer will **Buy** (Yes/No) based on their **Device** (Mobile, Desktop, Tablet).

**Step 1: The Root Node (Before Split)**
* Dataset: 15 Customers (10 Buy=Yes, 5 Buy=No)
* Gini Impurity of Root:
  \[ P(Yes) = 10/15 = 0.67 \]
  \[ P(No) = 5/15 = 0.33 \]
  \[ Gini_{root} = 1 - (0.67^2 + 0.33^2) = 1 - (0.444 + 0.109) = 0.447 \]

**Step 2: The Split by "Device"**
The tree divides the 15 customers based on their device.
* **Mobile**: 7 customers (6 Yes, 1 No)
* **Desktop**: 6 customers (3 Yes, 3 No)
* **Tablet**: 2 customers (1 Yes, 1 No)

**Step 3: Calculating Impurity for Each Branch**
* **Mobile Node**: \[ Gini = 1 - ((6/7)^2 + (1/7)^2) = 0.245 \]
* **Desktop Node**: \[ Gini = 1 - ((3/6)^2 + (3/6)^2) = 0.500 \] (Perfectly mixed)
* **Tablet Node**: \[ Gini = 1 - ((1/2)^2 + (1/2)^2) = 0.500 \] (Perfectly mixed)

**Step 4: Calculate Weighted Impurity & Information Gain**
* Weighted Impurity = \[ (7/15 \times 0.245) + (6/15 \times 0.500) + (2/15 \times 0.500) = 0.114 + 0.200 + 0.067 = 0.381 \]
* **Information Gain** = \[ 0.447 (Root) - 0.381 (Children) = \mathbf{0.066} \]

**Visual Flow of the Split:**

![Visual Flow of Categorical Split](https://kroki.io/mermaid/svg/eNqNj00LgkAQhu_9igEvBhraatFSgrnQqQ7hJaJDW2suSiu69AH--Kw1KfHgHIb38jzvzCU_ZjGEZADV-PutEBKDbcGOFQa4sBFzmnsrfuUYrJHjTA9gml5J2I2fGCxgLShPWQlLXaUKOLM3MlEGu2UYO-5QVf17CCsSKbISAr2OjQkpE2qZXMvqNIVHmjJZAtFVajx290WN57MK-aye8CHiaYq1aBYZhcxFwrCGEKqzeednGeNx9vhBljVCI9oXCb4ttDdCeiIv5HiOQQ==)

*The tree will compare this Information Gain (0.066) against other potential feature splits, choosing the one with the highest gain.*

---

## 8. How Thresholds are Chosen (Numeric Features)

<!-- ![Image](https://miro.medium.com/1%2AlGvZjpsekqvpdyKf90AsLw.png)

![Image](https://substackcdn.com/image/fetch/%24s_%21zeTN%21%2Cf_auto%2Cq_auto%3Agood%2Cfl_progressive%3Asteep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fee49b32b-b01a-4281-8299-84a7834e7d4c_2333x1019.png)

![Image](https://explained.ai/decision-tree-viz/images/knowledge-TD-3-X.png)

![Image](https://explained.ai/decision-tree-viz/images/boston-TD-AGE.svg) -->

Steps:

1. **Sort** the data by the numerical feature (e.g., Credit Score).
2. **Find Midpoints**: Calculate the midpoints between every consecutive pair of values. These midpoints become the "candidate thresholds".
3. **Split & Evaluate**: For every candidate threshold, split the data into `< threshold` and `>= threshold`. Calculate the weighted Gini impurity for the resulting children.
4. **Select Best Threshold**: Choose the threshold that results in the highest Information Gain (lowest weighted child impurity).

### Example: Numeric Split Feature & Categorical Target

**Scenario**: Predict Default (Yes/No) based on Credit Score.
Values (sorted): 630 (Yes), 650 (Yes), 680 (Yes), 690 (Yes), 710 (Yes), 720 (No), 760 (No), 780 (No)

**Step 1: Calculate Candidates**
Candidate thresholds: 640, 665, 685, 700, 715, 740, 770.

**Step 2: Try a threshold (e.g., 715)**
* Root: 5 Yes, 3 No (Gini: 0.469)
* **Left Child (Score <= 715)**: Includes {630, 650, 680, 690, 710}. Contains 5 Yes, 0 No.
  \[ Gini_{left} = 1 - (1^2 + 0^2) = 0 \] (Perfectly Pure!)
* **Right Child (Score > 715)**: Includes {720, 760, 780}. Contains 0 Yes, 3 No.
  \[ Gini_{right} = 1 - (0^2 + 1^2) = 0 \] (Perfectly Pure!)

**Step 3: Information Gain**
* Weighted Impurity = (5/8 * 0) + (3/8 * 0) = 0
* Information Gain = 0.469 - 0 = 0.469. This is the maximum possible gain!

**Visual Flow of the Split:**

![Visual Flow of Numerical Split](https://kroki.io/mermaid/svg/eNpLL0osyFAIceFSAALH6KD8_BIrBVOFyNRiHQVjBb98m6QiO_fMvEwrBQM9EzPLWAVdXbsa56LUlMwSheDk_KJUBRtbBXND0xoFJw2f1LQSoJaUVJAmqBkGaGYYaEJswjTHDmKMs0ZQZnoGwhwD7G6BmgMmiksqc1KBRqZl5uRYKadZpukUlxTlZ6daKRsbG0PZuuWZKSUZVkYFFUhanKBaktKSiNXiTKQWANygZvk=)

*Since splitting at 715 produces entirely pure nodes, the decision tree stops here for this branch.*

---

## 9. Regression Trees

Used when target is numeric.

Instead of Gini/Entropy → minimize **Mean Squared Error (MSE)**.

Leaf prediction = **mean of values** in that leaf.

Example:

Values: 100, 120 → mean = 110
Prediction for that region = 110

### Why mean?

Because mean minimizes squared error:

[
\sum (y_i - c)^2
]

is minimized when (c = \bar{y}).

---

## 10. Decision Boundaries

![Image](https://i.sstatic.net/FgdfC.jpg)

![Image](https://substackcdn.com/image/fetch/f_auto%2Cq_auto%3Agood%2Cfl_progressive%3Asteep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa64d9bf8-c9ec-4012-8134-1ccd2e2affc6_2664x2212.png)

![Image](https://framerusercontent.com/images/EcXajfuaF3vdq95dsHp75rGE.png)

![Image](https://365datascience.com/resources/blog/x671k7dla1f-overfitting-vs-underfitting-classification-examples.png)

Trees create **axis-aligned splits**:

* Feature1 > threshold → vertical line
* Feature2 > threshold → horizontal line

Each leaf = rectangular region.

---

## 11. Overfitting in Decision Trees

Problem:

* Deep trees memorize noise
* Leaves may contain only 1 sample
* High training accuracy, poor test accuracy

### Signs

* Very deep tree
* Many small regions

---

## 12. Controlling Overfitting (Regularization)

Key hyperparameters:

| Parameter             | Meaning                          |
| --------------------- | -------------------------------- |
| max_depth             | Limit tree depth                 |
| min_samples_split     | Minimum samples to split         |
| min_samples_leaf      | Minimum samples in leaf          |
| max_leaf_nodes        | Limit number of leaves           |
| min_impurity_decrease | Split only if gain is meaningful |

---

## 13. Pruning (Cost-Complexity)

Tree complexity penalty:

[
R_\alpha(T) = R(T) + \alpha |T|
]

Where:

* (R(T)) = training error
* (|T|) = number of leaves
* (\alpha) = `ccp_alpha`

Effect:

* α = 0 → large tree
* Higher α → smaller tree

Removes branches that don’t improve performance enough.

---

## 14. Categorical Features

Handling methods:

1. One-Hot Encoding (common)
2. Native category splits (some implementations)

Important:

* High-cardinality features may cause overfitting
* Ordinal encoding only if natural order exists

---

## 15. Multi-Class Case

Gini and Entropy generalize:

[
Gini = 1 - \sum p_i^2
]

Example:
Classes: [0.5, 0.3, 0.2]

[
Gini = 1 - (0.25 + 0.09 + 0.04) = 0.62
]

---

## 16. Why Trees Don’t Need Feature Scaling

Trees use comparisons:

```
Age ≤ 30
```

Scaling (Age × 100) does not change ordering → splits remain valid.

---

## 17. Advantages of Decision Trees

* Easy to interpret
* Handles non-linear relationships
* No feature scaling required
* Works with numerical and categorical data
* Fast training

---

## 18. Limitations

* High variance (overfitting)
* Sensitive to small data changes
* Piecewise constant predictions (regression)
* Axis-aligned splits only

---

## 19. When to Use Decision Trees

Good for:

* Tabular business data
* Rule-based decision systems
* Explainable models
* Baseline modeling

