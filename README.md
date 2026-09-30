# NumPy Machine Learning Functions

This file contains implementations of common **Machine Learning mathematical functions** using NumPy.

The examples cover:

* Sigmoid Function
* ReLU Function
* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)
* Min-Max Normalization
* Standardization
* One-Hot Encoding

---

## 📌 Requirements

Install NumPy:

```bash
pip install numpy
```

Import NumPy:

```python
import numpy as np
```

---

# 1. Sigmoid Function

The **Sigmoid** function converts values into a range between **0 and 1**.

### Formula

```text
Sigmoid(x) = 1 / (1 + e^(-x))
```

### Implementation

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

### Example

```python
x = np.array([-10, -5, 0, 5, 10])

print(sigmoid(x))
```

### Output

```text
[4.53978687e-05 6.69285092e-03 5.00000000e-01
 9.93307149e-01 9.99954602e-01]
```

### Key Point

* Large negative values → close to `0`
* `0` → `0.5`
* Large positive values → close to `1`

---

# 2. ReLU Function

**ReLU** stands for **Rectified Linear Unit**.

### Formula

```text
ReLU(x) = max(0, x)
```

It converts all negative values to `0` while keeping positive values unchanged.

### Implementation

```python
def Relu(x):
    return np.maximum(0, x)
```

### Example

```python
r = np.array([-5, -2, 0, 3, 7])

print(Relu(r))
```

### Output

```text
[0 0 0 3 7]
```

### Key Point

```text
Negative → 0
Positive → unchanged
```

---

# 3. Mean Squared Error (MSE)

**MSE** measures the average squared difference between actual and predicted values.

### Formula

```text
MSE = mean((actual - predicted)²)
```

### Implementation

```python
def MSE(actual, predicted):
    return np.mean((actual - predicted) ** 2)
```

### Example

```python
actual = np.array([10, 20, 30, 40])
predicted = np.array([12, 18, 33, 35])

print(MSE(actual, predicted))
```

### Output

```text
10.5
```

### Key Point

A smaller MSE generally means the predictions are closer to the actual values.

---

# 4. Mean Absolute Error (MAE)

**MAE** calculates the average absolute difference between actual and predicted values.

### Formula

```text
MAE = mean(|actual - predicted|)
```

### Correct Implementation

```python
def MAE(actual, predicted):
    return np.mean(np.abs(actual - predicted))
```

### Example

```python
actual = np.array([10, 20, 30, 40])
predicted = np.array([12, 18, 33, 35])

print(MAE(actual, predicted))
```

### Output

```text
3.0
```

### ⚠️ Important

Your original version:

```python
def MAE(actual, predicted):
    return np.mean(actual - predicted)
```

is **not MAE** because it does not take the absolute value.

Use:

```python
np.abs(actual - predicted)
```

---

# 5. Min-Max Normalization

Min-Max normalization converts values to a range between **0 and 1**.

### Formula

```text
x_normalized = (x - min(x)) / (max(x) - min(x))
```

### Implementation

```python
def x_normalized(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))
```

### Example

```python
x = np.array([10, 20, 30, 40, 50])

print(x_normalized(x))
```

### Output

```text
[0.   0.25 0.5  0.75 1.  ]
```

### Key Point

The minimum value becomes:

```text
0
```

The maximum value becomes:

```text
1
```

---

# 6. Standardization

Standardization converts data into **Z-scores**.

### Formula

```text
z = (x - mean) / standard_deviation
```

### Implementation

```python
def Standardization(x):
    return (x - np.mean(x)) / np.std(x)
```

### Example

```python
x = np.array([10, 20, 30, 40, 50])

print(Standardization(x))
```

### Output

```text
[-1.41421356 -0.70710678  0.
  0.70710678  1.41421356]
```

### Key Point

After standardization:

```text
Mean ≈ 0
Standard Deviation ≈ 1
```

---

# 7. One-Hot Encoding

One-hot encoding converts categorical labels into binary vectors.

Given:

```python
labels = np.array([0, 2, 1, 2, 0])
```

We want:

```text
[1 0 0]
[0 0 1]
[0 1 0]
[0 0 1]
[1 0 0]
```

### Implementation using NumPy Indexing

```python
labels = np.array([0, 2, 1, 2, 0])

one_hot = np.zeros((len(labels), 3), dtype=int)

one_hot[np.arange(len(labels)), labels] = 1

print(one_hot)
```

### Output

```text
[[1 0 0]
 [0 0 1]
 [0 1 0]
 [0 0 1]
 [1 0 0]]
```

### How the indexing works

```python
np.arange(len(labels))
```

produces:

```text
[0 1 2 3 4]
```

And:

```python
labels
```

contains:

```text
[0 2 1 2 0]
```

Therefore:

```python
one_hot[np.arange(len(labels)), labels] = 1
```

sets these positions to `1`:

```text
(0,0)
(1,2)
(2,1)
(3,2)
(4,0)
```

---

# 🧠 Concepts Practiced

This practice covers several important NumPy concepts:

| Concept               | NumPy Function       |
| --------------------- | -------------------- |
| Exponential           | `np.exp()`           |
| Maximum               | `np.maximum()`       |
| Mean                  | `np.mean()`          |
| Absolute value        | `np.abs()`           |
| Minimum               | `np.min()`           |
| Maximum value         | `np.max()`           |
| Standard deviation    | `np.std()`           |
| Array creation        | `np.zeros()`         |
| Array indexing        | `array[row, column]` |
| Range generation      | `np.arange()`        |
| Vectorized operations | NumPy operations     |

---

# 🤖 Machine Learning Connection

These functions are commonly encountered when learning Machine Learning:

```text
NumPy
  ↓
Mathematical Operations
  ↓
Data Preprocessing
  ↓
Machine Learning Algorithms
```

Examples:

* **Sigmoid** → Logistic Regression / Neural Networks
* **ReLU** → Neural Networks / Deep Learning
* **MSE** → Regression model evaluation
* **MAE** → Regression model evaluation
* **Min-Max Scaling** → Feature preprocessing
* **Standardization** → Feature preprocessing
* **One-Hot Encoding** → Categorical data preprocessing

---

# 🚀 Practice Goal

The main goal of this practice is to understand how common Machine Learning operations can be implemented using **NumPy without loops**.

Focus especially on:

```python
np.mean()
np.abs()
np.exp()
np.maximum()
np.min()
np.max()
np.std()
np.zeros()
np.arange()
```

These functions are fundamental for working with numerical data in Python and Machine Learning.
