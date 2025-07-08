# Task 2 - Linear Regression

## Table of Contents

- [Overview](#overview)
- [Optimisation Methods](#optimisation-methods)
- [Regularisation Methods](#regularisation-methods)
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

Fundamental formula: $$θ = (X^{T} * X)^{-1} * X^{T} * Y$$

Where:
- $$X \in \mathbb{R}^{m \times n}$$ is the matrix of feature vectors.
- $$Y \in \mathbb{R}^{m \times 1}$$ is the column vector of actual values.
- $$θ \in \mathbb{R}^{n \times 1}$$ is the column vector of model coefficients.

## Regularisation Methods

In the field of Machine Learning, **regularisation** is a technique used to make a model more generalisable. 
This means it helps the model reduce its variance error when encountering new data after training.

The assignment features two types of regularisation techniques:

### 1. L2 Regularisation (Ridge Regression)

L2 regularisation focuses on finding a regression line that fits the data while introducing a small **bias**. 
This means the line does not perfectly minimise the sum of squared distances between the predicted values and the training data points.

The main idea is to **shrink the model coefficients** $$θ_0, θ_1, \dots, θ_n$$ towards zero. 
This weakens the dependency between the output $$y^{(i)}$$ and certain input features  $$x_1, x_2, \dots, x_n$$, leading to a model that depends less on individual predictors.

The L2 regularised cost function is defined as:

$$
J_{L2}(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2 + \lambda \sum_{j=1}^{n} \theta_j^2
$$

Where:
- $$\lambda \sum_{j=1}^{n} \theta_j^2 $$ is the **L2 regularization term**
- $$\lambda \in \mathbb{R}_+ $$ controls the **strength of the regularization**
- $$\lambda$$ is usually selected via **cross-validation**, but will be provided in implementation for this case

This approach helps prevent overfitting by discouraging overly complex models with large coefficients.

### L1 Regularisation (Lasso Regression)

L1 Regularisation is similar in purpose to L2 regularisation, aiming to reduce the complexity of the machine learning model. 
However, Lasso Regression has a distinct effect: some of the model’s coefficients $$θ_0, θ_1, \dots, θ_n$$ can become exactly **zero**.

This means that **certain predictors can be entirely removed** from the model, effectively performing **feature selection**. 
The result is a simpler model that still aims to generalise well to new data.

The L1 regularised cost function is defined as:

$$
J_{L1}(\theta) = \frac{1}{m} \sum_{i=1}^{m} \left( y^{(i)} - h_\theta(x^{(i)}) \right)^2 + \lambda ||\theta||_1
$$

Where:
- $$||\theta||_1 = |\theta_0| + |\theta_1| + \cdots + |\theta_n|$$ is the **L1 norm** of the coefficient vector
- $$\lambda \in \mathbb{R}_+$$ is the **regularisation parameter** that controls the strength of the penalty

L1 regularisation is especially useful when the dataset contains many irrelevant or weak features, as it helps the model focus only on the most important ones.

## Functions

Now that the theoretical background is fully documented, below are the necessary MATLAB functions to perform the tasks:



## Running the Tasks
