# Task 2 - Linear Regression

## Table of Contents

- [Overview](#overview)
- [Optimisation Methods](#optimisation-methods)
- [Functions](#functions)
- [Running the Tasks](#running-the-tasks)

## Overview

Linear Regression can be understood geometrically as a regression line that minimizes the square root of the sum of squared distances from the data points to the line. 
Functionally, it aims to minimize a cost or loss function. There are different types of Linear Regression: simple, multiple, and logistic.
For this assignment, the chosen type was the **Multiple Linear Regression** to predict apartment prices in a fictional urban area. 

The regression function is defined as:
$$\space h_θ(x) = θ_0 + θ_1 * x_1 + θ_2 * x_2 + ... + θ_n * x_n + ε$$

Where:
- $$h_θ(x)$$ is the predicted value based on input features $$(x_1, x_2, ..., x_n)$$.
- $$θ_0$$ is the intercept (value of $$h_θ(x)$$ when all features are 0).
- $$θ_1, ..., θ_n$$ are the weights or coefficients of the model.
- $$ε$$ is the prediction error (the difference between predicted and actual value).

These coefficients define how well the model generalises to new, unseen data. The goal is to find θ values that minimise the prediction error.

To measure the performance of the model, a cost function **J(θ)** is defined as follows:

$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_{\theta}(x^{(i)}) - y^{(i)} \right)^2
$$

Where:
- $$m$$ is the number of training examples.
- $$x^{(i)}$$ is the input vector for the i-th training example.
- $$y^{(i)}$$ is the actual output for the i-th training example.
- $$h_θ(x^{(i)})$$ is the model's prediction for the i-th input.

The cost function computes the average squared difference between predicted and actual values. Minimising this function improves the accuracy of the model.

## Optimisation Methods

There are two main optimisation algorithms to determine the model coefficients in linear regression:

### 1. Gradient Descent Method

- Gradient Descent is a general iterative optimisation algorithm used to minimise convex functions
- In the context of linear regression, it is applied to minimize the cost function $$\( J(\theta) \)$$
- Since the cost function is convex and has a unique global minimum, any local minimum is also a global minimum
- The method updates the parameters $$\( \theta \)$$ based on the gradient of the cost function and a learning rate $$\( \alpha \in \mathbb{R} \)$$

In MATLAB notation, the gradient vector is:
 ∇J(θ) = [
$$\frac{∂J}{∂θ_1}(\theta)$$
...
$$\frac{∂J}{∂θ_n}(\theta)$$
]'

Where each partial derivative is computed as:  $$\frac{∂J}{∂θ_j} = \frac{1}{m} \sum_{i=1}^{m} \left( h_{\theta}(x^{(i)}) - y^{(i)} \right) * x_j^{(i)}$$

Update rule:  $$θ_j := θ_j - \alpha * \frac{∂J}{∂θ_j}(\theta)$$

### 2. Normal Equation Method

- This is a closed-form solution for computing the coefficients θ directly, without iteration
- It is effective for small datasets only, due to matrix inversion
- To address this, the **Conjugate Gradient Method** is used as an efficient alternative for solving large systems

Fundamental formula: $$θ_j = (X^{T} * X)^{-1} * X^{T} * Y$$

Where:
- $$X \in \mathbb{R}^{m \times n}$$ is the matrix of feature vectors.
- $$Y \in \mathbb{R}^{m \times 1}$$ is the column vector of actual values.
- $$θ \in \mathbb{R}^{n \times 1}$$ is the column vector of model coefficients.


## Functions



## Running the Tasks
