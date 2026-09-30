import numpy as np
# Sigmod Function 
# Implement sigmoid without using a loop:

def sigmoid(x):
    return 1 /(1+np.exp(-(x)))     

# Test it with:

# x = np.array([-10, -5, 0, 5, 10])
x = np.array([-10, -5, 0, 5, 10])
print(sigmoid(x))
# 59. ReLU

# Implement:

# ReLU(x) = max(0, x)
def Relu (x) :
    return np.maximum(0,x)
# using NumPy.

# Example:

r = np.array([-5, -2, 0, 3, 7])

# Expected:
print(Relu(r))
# [0, 0, 0, 3, 7]
# 60. Mean Squared Error

# Implement:

# MSE = mean((actual - predicted)²)
def MSE (actual , predicated) :
    return np.mean((actual -predicated) **2)
# without using a loop.

# 61. Mean Absolute Error

# Implement:

# MAE = mean(|actual - predicted|)
def MAE (actual , predicated) :
    return np.mean((actual -predicated))
# using NumPy.

# 62. Min-Max Normalization

# Given:

# x = np.array([10, 20, 30, 40, 50])
x = np.array([10, 20, 30, 40, 50])
# implement:

# x_normalized = (x - min(x)) / (max(x) - min(x))

def x_normalized(x) :
    return (x - np.min(x)) / (np.max(x) - np.min(x))
# Expected range:
print(x_normalized(x))
# 0 → 1
# 63. Standardization

# Implement:

# z = (x - mean) / standard_deviation
def Standardization(x):
    return (x - np.mean(x)) / np.std(x)
# using NumPy.

# 64. One-hot encoding

# Given:

# labels = np.array([0, 2, 1, 2, 0])
labels = np.array([0, 2, 1, 2, 0])
# convert it into:

# [1 0 0]
# [0 0 1]
# [0 1 0]
# [0 0 1]
# [1 0 0]
one_hot = np.zeros((len(labels), 3), dtype=int)
one_hot[np.arange(len(labels)), labels] = 1
# Try doing it using NumPy indexing.
print(one_hot)