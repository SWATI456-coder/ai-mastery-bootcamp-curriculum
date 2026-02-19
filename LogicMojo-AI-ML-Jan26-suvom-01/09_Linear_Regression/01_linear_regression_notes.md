# Linear Regression

---

## Contents (Quick)

- 0. Theory: Why ML? Why Linear Regression?
- 1. What is Linear Regression?
- 2. Visual Intuition: Best Fit Line
- 3. Cost Function (MSE)
- 4. Training = Optimization
- 5. Gradient Descent
- 6. Vectorized Gradient
- 7. Assumptions of Linear Regression
- 8. Normality of Residuals
- 9. Overfitting vs Underfitting
- 10. Why Overfitting Happens
- 11. Regularization
- 12. Ridge Regression (L2 Regularization)
- 13. Lasso Regression (L1 Regularization)
- 14. Ridge vs Lasso
- 15. Practical Tips
- 16. Evaluation Metrics
- 17. Real-World Use Cases
- 18. Final Summary

---

## Notation (Symbols used)

*   \(X\): feature matrix (inputs)
*   \(y\): actual target values (labels)
*   \(\hat{y}\): predicted target values
*   \(\theta\): model parameters (weights + intercept)
*   \(m\): number of training examples
*   \(\alpha\): learning rate (gradient descent step size)
*   \(\lambda\): regularization strength (penalty weight)

---

## 0. Theory: Why ML? Why Linear Regression?

### Concept
Machine Learning (ML) is a subset of artificial intelligence that focuses on building systems that learn from data, rather than being explicitly programmed. In traditional programming, you provide rules and data to get answers. In ML, you provide data and answers (labels) to learn the rules.

### Definition
**Machine Learning** is the study of computer algorithms that improve automatically through experience and by the use of data.
**Supervised Learning** is a type of ML where the model is trained on labeled data (input-output pairs).

### Notes / Intuition
In many real-world problems we do **not** know the exact formula that maps inputs to outputs. For instance, predicting a house price involves many complex factors (location, size, condition) that are hard to capture with hand-written if-else rules. ML algorithms approximate this function by finding patterns in historical data.

*   **Features (X)**: measurable inputs (area, rooms, ad spend)
*   **Target (y)**: value to predict (price, sales, demand)

### Example
**Traditional Programming:**
`Price = Area * 1000 + Rooms * 5000` (Hardcoded rules)

**Machine Learning:**
Input: (Area=1500, Price=200k), (Area=2000, Price=300k)...
Model learns: `Price ≈ 130 * Area + 10000` (Derived from data)

---

### Why Linear Regression?

Linear Regression is a fundamental algorithm in statistics and machine learning used for predictive analysis. It assumes a **linear relationship** between the input variables (\(X\)) and the output variable (\(y\)).

**Definition**: **Linear Regression** is a linear approach for modeling the relationship between a scalar response and one or more explanatory variables (dependent vs independent variables).

**Why it’s often the first model**:

*   **Simple and fast**: computationally inexpensive.
*   **Highly interpretable**: weights (\(\theta\)) directly show how features affect the prediction.
*   **Strong baseline**: try this first before moving to complex models.

Use when the relationship is **approximately linear**.

---

## 1. What is Linear Regression?

### Concept
The core idea is to find a line (or hyperplane in higher dimensions) that best fits the data points. The model predicts the dependent variable as a weighted sum of the independent variables plus a bias term (intercept).

### Definition
The hypothesis function $h_\theta(x)$ is defined as:
$$
\hat{y} = h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n
$$

Where:
*   $\hat{y}$: predicted value
*   $\theta_0$: bias or intercept (value of $y$ when all $x$ are 0)
*   $\theta_1, ..., \theta_n$: weights or coefficients (how much $y$ changes for a unit change in $x_i$)
*   $x_1, ..., x_n$: input features

### Notes / Intuition
Think of $\theta$ as "knobs" that we turn to adjust the line.
*   If $\theta_1$ is positive, $x_1$ increases $y$.
*   If $\theta_1$ is negative, $x_1$ decreases $y$.
*   $\theta_0$ shifts the line up or down.

### Example
Predicting exam score based on hours studied:
$$ \text{Score} = 10 + 5 \times (\text{Hours}) $$
*   $\theta_0 = 10$: Even with 0 hours, you might get 10 marks per base knowledge.
*   $\theta_1 = 5$: Every hour of study increases score by 5 points.

---

## 2. Visual Intuition: Best Fit Line

### Concept
In 2D (one feature), the model is a straight line. In 3D (two features), it's a plane. The "best fit" means the line that passes as close as possible to all data points.

### Definition
**Line of Best Fit**: A straight line that best represents the data on a scatter plot. This line may pass through some of the points, none of them, or all of them.

### Notes / Intuition
*   The line represents the model’s prediction.
*   Vertical gaps between points and line are **residuals (errors)**.
*   Goal: choose a line that minimizes total error (the sum of squared residuals).

![Image](https://miro.medium.com/v2/resize%3Afit%3A1200/1%2AJiuVb5EdbE1ROPUZVk3Rlw.png)
![Image](https://www.researchgate.net/publication/340837542/figure/fig2/AS%3A883111357317129%401587561563861/Scatter-plot-of-predicted-value-vs-actual-value-from-RSM-design.jpg)
![Image](https://www.researchgate.net/publication/360460360/figure/fig4/AS%3A1175393159659522%401657246975521/The-scatter-plot-between-the-actual-and-predicted-values-according-to-a-multiple-linear.png)

---

## 3. Cost Function (MSE)

### Concept
To find the best parameters $\theta$, we need a way to measure how "wrong" the model is. This is done using a cost function.

### Definition
**Mean Squared Error (MSE)** is the average of the squares of the errors—that is, the average squared difference between the estimated values and the actual value.
$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2
$$
*   $m$: number of training examples
*   The factor $\frac{1}{2}$ is often added to simplify derivative calculation (it cancels out the exponent 2).

### Notes / Intuition
**Why squared?**
1.  **Removes negative signs**: Error of -5 and +5 are both bad; squaring makes them both 25.
2.  **Penalizes large errors**: An error of 10 becomes 100, while an error of 1 becomes 1. This forces the model to focus on fixing big mistakes.
3.  **Differentiable and Convex**: The squared function forms a nice bowl shape (parabola), ensuring a single global minimum that is easy to find with optimization algorithms.

### Example
Actual: [2, 4, 6]
Predicted: [3, 3, 7]
Errors: [-1, +1, -1]
Squared Errors: [1, 1, 1] -> Sum = 3 -> Mean = 1.
(Ideally, we want this to be 0).

---

## 4. Training = Optimization

### Concept
Training a model simply means finding the values of parameters ($\theta$) that result in the minimum possible cost.

### Definition
**Optimization** is the mathematical procedure of selecting the best element from a set of available alternatives. Here, it is finding $\theta$ that minimizes $J(\theta)$.
$$
\theta^* = \arg\min_{\theta} J(\theta)
$$

### Notes / Intuition
Imagine you are at the top of a mountain (high error) and you want to get to the very bottom of the valley (minimum error). You look around and take a step in the direction that goes down the steepest. You keep doing this until you reach the bottom.

---

## 5. Gradient Descent

### Concept
Gradient Descent is an iterative optimization algorithm for finding the local minimum of a differentiable function.

### Definition
**Gradient Descent** updates parameters in the opposite direction of the gradient of the objective function (cost function) w.r.t. to the parameters.

### Notes / Intuition
*   The **gradient** $(\nabla J)$ tells us the direction of steepest ascent (upward slope).
*   We want to go down, so we subtract the gradient.
*   The **learning rate** $(\alpha)$ decides how big of a step we take.

### Cost Surface and Direction
![Image](https://miro.medium.com/v2/resize%3Afit%3A1400/1%2AmsSz99bjOkIbc0gJUzZlTA.png)
![Image](https://adeveloperdiary.com/assets/img/How-to-visualize-Gradient-Descent-using-Contour-plot-in-Python-adeveloperdiary.com-1.webp)
![Image](https://www.researchgate.net/publication/379294688/figure/fig2/AS%3A11431281258211757%401720009758285/Cost-Vs-Iteration-graph-for-Stochastic-Gradient-Descent.ppm)

The bowl represents the cost function. Gradient descent moves step-by-step toward the lowest point (minimum error).

### Learning Rate Behavior (Most Important Visual)
![Image](https://www.jeremyjordan.me/content/images/2018/02/Screen-Shot-2018-02-24-at-11.47.09-AM.png)
![Image](https://miro.medium.com/v2/resize%3Afit%3A1400/0%2A7Ml5AasK2c4-eaVD.png)
![Image](https://miro.medium.com/v2/resize%3Afit%3A1400/0%2AARiY_3OpxNoArNw2)
![Image](https://cdn-images-1.medium.com/max/1440/1%2ADvxLu7yCO6Fxtv8Y6TVhlg.jpeg)

*   **Small learning rate**: Baby steps. Safe but very slow.
*   **Large learning rate**: Giant leaps. Might overshoot the minimum and diverge (error increases).
*   **Optimal learning rate**: Fast and stable convergence.

### Why the Minus Sign?

Update rule:
$$ \theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta) $$

### Intuition
| Gradient sign (slope) | Meaning | Update effect |
| --- | --- | --- |
| Positive slope (+) | Cost increases to the right | Subtract positive → move left (decrease $\theta$) |
| Negative slope (-) | Cost decreases to the right | Subtract negative → add positive → move right (increase $\theta$) |

This ensures we always move **toward the minimum**, regardless of which side of the valley we are on.

---

## 6. Vectorized Gradient

### Concept
Instead of using for-loops to update each weight individually, we use Linear Algebra (matrix operations) to compute updates for all weights simultaneously. This leverages modern CPU/GPU SIMD instructions for speed.

### Definition
$$
\nabla J(\theta) = \frac{1}{m} X^T (X\theta - y)
$$
Where $X$ is the design matrix (rows are samples), $\theta$ is the weight vector, and $y$ is the target vector.

### Notes / Intuition
*   $X\theta$: Predictions for all samples at once.
*   $(X\theta - y)$: Errors for all samples.
*   $X^T (...)$: Dot product sums up the errors weighted by feature values, effectively calculating the derivative for all $\theta$s in one shot.

---

## 7. Assumptions of Linear Regression

### Concept
Linear Regression is a parametric model. Ideally, the data must satisfy certain conditions for the model's estimates (and statistical tests like p-values) to be valid and unbiased.

### Explanation of Assumptions
1.  **Linearity**: The relationship between X and y is linear. If the data is curved (parabolic), a straight line will underfit.
2.  **Independence**: Sample observations are independent. (No autocorrelation, important in time-series).
3.  **Homoscedasticity (Equal Variance)**: The "spread" or variance of residuals is constant across all predicted values. (No "funnel" shape where errors get bigger as predictions get bigger).
4.  **No Multicollinearity**: Features should not be highly correlated with each other (e.g., using both "kg" and "lbs" as features). This makes the matrix inversion unstable.
5.  **Normality of Residuals**: The errors follow a Normal (Gaussian) distribution.

![Image](https://res.cloudinary.com/jerrick/image/upload/c_scale%2Cf_jpg%2Cq_auto/649e6c7b6bcb42001deb4af8.png)
![Image](https://miro.medium.com/v2/resize%3Afit%3A1400/0%2AjQ7I-M3BhknLRMXw.jpg)
![Image](https://metricgate.com/blogs/assets/blogs/collinearity-vs-multicollinearity.png)
![Image](https://editor.analyticsvidhya.com/uploads/95472heatmap.png)

---

## 8. Normality of Residuals

### Concept
While the model can predict without this assumption, it is crucial for **statistical inference** (calculating confidence intervals and hypothesis testing).

### Definition
The errors $\epsilon = y - \hat{y}$ should follow: $\epsilon \sim \mathcal{N}(0, \sigma^2)$.

### Notes / Intuition
If residuals are not normal (e.g., skewed), it means the model is consistently under-predicting or over-predicting for certain ranges, or that outliers are influencing the fit.

### Diagnostics
*   **Histogram**: Should look like a bell curve centered at 0.
*   **Q-Q Plot**: Points should lie on the 45-degree diagonal line.

![Image](https://www.researchgate.net/publication/322832910/figure/fig6/AS%3A609092750041089%401522230436713/The-histogram-of-the-residuals-with-a-normal-curve-superimposed.png)
![Image](https://d2o2utebsixu4k.cloudfront.net/ChatGPT%20Image%20May%2014%2C%202025%2C%2003_53_37%20PM-3ad29590faf04dc0abea7c25e4325c62.png)
![Image](https://miro.medium.com/1%2A30tdeThF77AQHxUlZoLJQg.png)

---

## 9. Overfitting vs Underfitting

### Concept
The bias-variance tradeoff is a central problem in supervised learning. It relates to the model's ability to generalize to new, unseen data.

### Definition
*   **Bias**: Error due to overly simplistic assumptions (Underfitting).
*   **Variance**: Error due to excessive sensitivity to small fluctuations in the training set (Overfitting).

### Notes / Intuition
*   **Underfitting (High Bias)**: The model is "too dumb". It fails to capture the underlying trend. (e.g., fitting a straight line to a parabola).
*   **Overfitting (High Variance)**: The model is "too smart". It memorizes the training data, including noise and outliers. It performs great on training data but fails on new data.
*   **Good Fit**: Captures the signal, ignores the noise.

![Image](https://media.licdn.com/dms/image/v2/C5112AQFBL0RbaQlYpQ/article-cover_image-shrink_600_2000/article-cover_image-shrink_600_2000/0/1580724772581?e=2147483647\&t=-WB-qnmoRsHdvDbPthQDTfujfX8KgholIAZx6YYWXkE\&v=beta)
![Image](https://images.openai.com/static-rsc-3/ctwE7UEz6KYzHi11mg7hdg4BJK39tgPkzLQPxZhL1MFI-cLv6p32nCbzDk1NCItPas1Hl_W4q-QA-iJ54o8Q7DFCmld8Yqmlvoab5MS3ptA?purpose=fullsize\&v=1)
![Image](https://codingnomads.com/images/f8c470c9-c714-48e6-0397-7a9a4e57b500/public)

---

## 10. Why Overfitting Happens

### Concept
Overfitting occurs when the model has more **capacity** (complexity) than justified by the amount of data.

### Explanation factors:
1.  **Too many features**: If you have 100 features for only 50 data points, the model can find chance correlations that don't exist in reality.
2.  **High-degree polynomial**: Allowing the model to curve too much ($x^5, x^{10}$) lets it wiggle to hit every single point.
3.  **Small dataset**: Not enough examples to smooth out the noise.
4.  **Multicollinearity**: Redundant features confuse the model.

### Example
Imagine trying to learn a language.
*   **Underfitting**: You only learn "Hello".
*   **Overfitting**: You memorize the specific sentences in the textbook, but can't speak a new sentence.
*   **Generalization**: You learn the grammar rules and can form new sentences.

---

## 11. Regularization

### Concept
Regularization is a technique to discourage learning a more complex or flexible model, so as to avoid the risk of overfitting.

### Definition
It adds a penalty term to the cost function to constrain the magnitude of the coefficients ($\theta$).
$$
\text{Cost} = \text{MSE} + \lambda \times \text{Penalty}
$$

### Notes / Intuition
We want to minimize the error, *but also* keep the weights ($\theta$) small. Large weights usually indicate overfitting (sensitivity to small changes).
*   $\lambda$ (Lambda): The regularization hyperparameter. It controls how much we penalize complexity.

### Effect of $\lambda$
| $\lambda$ value | Result | Explanation |
| --- | --- | --- |
| 0 | Overfitting | Same as standard linear regression. |
| Very large | Underfitting | Forces all $\theta \approx 0$. Model becomes a flat line. |
| Optimal | Generalization | Balances fit and complexity. |

---

## 12. Ridge Regression (L2 Regularization)

### Concept
Ridge regression adds a penalty equal to the square of the magnitude of coefficients.

### Definition
$$
J(\theta) = \text{MSE} + \lambda \sum_{j=1}^{n} \theta_j^2
$$

### Notes / Intuition
*   **Shrinks weights**: It forces coefficients to be small but rarely zero.
*   **Handles Multicollinearity**: If two features are correlated, Ridge will distribute the coefficient weight between them equally.
*   **L2 Norm**: Geometrically, it constrains the coefficients within a circle (hypersphere).

### Example
If feature 1 and feature 2 are identical, Ridge gives both $\theta_1=0.5, \theta_2=0.5$ (instead of one being 1 and the other 0).

---

## 13. Lasso Regression (L1 Regularization)

### Concept
Least Absolute Shrinkage and Selection Operator (Lasso) adds a penalty equal to the absolute value of the magnitude of coefficients.

### Definition
$$
J(\theta) = \text{MSE} + \lambda \sum_{j=1}^{n} |\theta_j|
$$

### Notes / Intuition
*   **Sparsity**: L1 regularization has a unique property where it can force some coefficients to be **exactly zero**.
*   **Feature Selection**: Because it zeros out irrelevant features, it performs automatic feature selection.
*   **L1 Norm**: Geometrically, it constrains coefficients within a diamond shape. Corner solutions (zeros) are likely.

### Example
If you have 100 features but only 5 are important, Lasso will likely set the other 95 weights to 0.

---

## 14. Ridge vs Lasso

### Comparison Table

| Feature | Ridge (L2) | Lasso (L1) |
| --- | --- | --- |
| **Penalty** | Squared Magnitude ($\theta^2$) | Absolute Magnitude ($|\theta|$) |
| **Effect on Weights** | Approaches zero asymptotically | Can reach exactly zero |
| **Feature Selection** | No (keeps all features) | Yes (removes irrelevant features) |
| **Best Used When** | Many features, all somewhat useful | Many features, only a few useful (sparse) |
| **Differentiable?** | Always | No (at 0), needs subgradient methods |

---

## 15. Practical Tips

### Feature Scaling
**Theory**: Gradient descent converges much faster if all features are on a similar scale (e.g., all between 0 and 1).
**Practice**: Apply **StandardScaler** (Z-score normalization) or **MinMaxScaler** before training.
*   Without scaling: Cost function is a skewed "taco" shape (slow to crawl, oscillates).
*   With scaling: Cost function is a nice symmetric bowl (fast direct path to bottom).

### Choosing $\lambda$ (Hyperparameter Tuning)
**Theory**: You cannot learn $\lambda$ from the training data (it will always choose 0 to minimize error).
**Practice**: Use **Cross-Validation**. Try a range of values (e.g., $10^{-4}, 10^{-3}, ..., 10, 100$) and see which one gives the lowest error on the validation set.

### Note on Intercept
**Do not regularize $\theta_0$ (intercept).** The intercept adjusts the height of the line; penalizing it would force the line to pass through the origin (0,0), which might not be appropriate for the data.

---

## 16. Evaluation Metrics

### Theory
We need standardized numbers to report how well the model performs.

### 1. RMSE (Root Mean Squared Error)
$$ \text{RMSE} = \sqrt{\frac{1}{m} \sum (\hat{y} - y)^2} $$
*   **Explanation**: It describes the average error in the **same units** as the target variable.
*   **Example**: "Our house price prediction is off by roughly $5,000 on average."

### 2. MAE (Mean Absolute Error)
$$ \text{MAE} = \frac{1}{m} \sum |\hat{y} - y| $$
*   **Explanation**: More robust to outliers than RMSE (doesn't square the big errors). Gives the direct average distance.

### 3. R-Squared ($R^2$)
$$ R^2 = 1 - \frac{SS_{residuals}}{SS_{total}} $$
*   **Theory**: Proportion of variance in the dependent variable that is predictable from the independent variable(s).
*   **Interpretation**:
    *   $1.0$: Perfect fit.
    *   $0.0$: Worst possible fit (equivalent to just predicting the mean value for everyone).
    *   Example: $R^2 = 0.8$ means "80% of the variation in house prices is explained by our features (size, location)."

### 4. Adjusted R-Squared ($R^2_{adj}$)

Adjusted \(R^2\) penalizes adding extra (unnecessary) features, so it is better for comparing models with different numbers of features.

If \(m\) = number of samples and \(n\) = number of features (excluding intercept), then:

$$
R^2_{adj} = 1 - (1 - R^2)\frac{m - 1}{m - n - 1}
$$

*   **Key point**: \(R^2\) never decreases when you add features, but \(R^2_{adj}\) can decrease if the new feature does not help.

---

## 17. Real-World Use Cases

### Theory
Linear regression is arguably the most widely used statistical technique in business and science.

### Examples
1.  **Real Estate**: Predicting House Price based on sq ft, number of rooms, zip code.
2.  **Marketing**: Predicting Sales based on Ad Spend (TV, Radio, Social Media).
    *   *Interpretation*: "For every $1 spent on TV, sales increase by $5."
3.  **Medical**: Predicting blood pressure based on age, weight, and BMI.
4.  **Agriculture**: Predicting crop yield based on rainfall and temperature.
5.  **Economics**: Trend forecasting (GDP growth, stock market trends).

---

## 18. Final Summary

1.  **Linear Regression** maps inputs to continuous outputs using a linear equation.
2.  **Cost Function (MSE)** measures standard error; **Gradient Descent** minimizes this error iteratively.
3.  **Assumptions** (Linearity, Independence, Normality, Homoscedasticity) must be checked for reliable results.
4.  **Overfitting** happens when the model is too complex; **Underfitting** when too simple.
5.  **Regularization** (Ridge/Lasso) adds a penalty to weights to prevent overfitting.
    *   **Ridge** shrinks weights (good for multicollinearity).
    *   **Lasso** removes features (good for sparsity).
6.  Always **scale your data** and check **metrics (RMSE, $R^2$, Adjusted $R^2$)** to validate performance.
