# Task 3 - MNIST

## Table of Contents

- [Overview](#overview)
- [Theoretical References](#theoretical-references)
- [Functions](#functions)
- [Running the Tasks](#running-the-tasks)

## Overview

After exploring numerical algorithms and prediction using linear regression in the second task, the final one takes a final step to adapt these algorithms and optimisation methods for other problems, such as **classification**.
The goal is to classify images of handwritten digits (0 to 9) using an appropriate classification model. Since this is a multi-class classification problem, a small neural network is chosen, containing:

- An input layer of 400 neurons (corresponding to the 20×20 pixel values of each image)
- An output layer of 10 neurons (one for each digit class)
- A hidden layer with 25 neurons to increase model complexity and improve classification performance

## Theoretical References

### Logistic Regression

The basic principle behind making predictions using linear regression is that the desired result is a **linear combination** of a set of given parameters (features). 
A similar model can be adapted for **classification** problems with a finite number of classes.

To begin, consider a simple **binary classification** problem. Like linear regression, binary classification uses an input vector of parameters along with a **label** (the target output), which in this case takes values in the set: $$y \in \{0, 1\}$$

However, a key issue arises when applying linear regression to classification: linear regression predicts real-valued outputs (possibly outside the range [0,1]), which are **not appropriate** for a binary decision.

To address this, we need to **introduce a non-linearity** that maps the output of the linear combination into the interval [0, 1]. 
Instead of using the linear hypothesis $$h_θ(x) = θ^T x$$, we use a **modified hypothesis** of the form: 

$$h_θ(x) = \sigma(θ^T x)$$, where $$\sigma : \mathbb{R} \rightarrow [0, 1]$$ is a non-linear **activation function**.
The most commonly used function for this purpose is the **sigmoid function**, defined as:

$$
\sigma : \mathbb{R} \rightarrow [0, 1],\space \sigma(x) = \frac{1}{1 + e^{-x}}
$$

With the new form of our hypothesis, we also need to **redefine the cost function**, since the **mean squared error** used in linear regression is no longer suitable. 
We want an **untrained model** to incur a high penalty if its prediction differs significantly from the true class label.

To better capture the **extreme cases** where prediction and reality diverge, we introduce a new cost function called **cross-entropy loss**, defined for a single example as:

$$
cost^{(i)} = -y^{(i)} \cdot \log(h_θ(x^{(i)})) - (1 - y^{(i)}) \cdot \log(1 - h_θ(x^{(i)}))
$$

For the entire training set, the cost function becomes:

$$
J(θ) = \frac{1}{m} \sum_{i=1}^{m} \left[ -y^{(i)} \cdot \log(h_θ(x^{(i)})) - (1 - y^{(i)}) \cdot \log(1 - h_θ(x^{(i)})) \right]
$$

This form of the cost function is especially effective for binary classification problems because:

- It penalises confident wrong predictions heavily.
- It maps predicted probabilities close to the actual class labels.

To **optimise** this cost function, we can use the same techniques already introduced for linear regression, such as **Gradient Descent** or a modified **Conjugate Gradient method**, resulting in models with strong performance for basic classification tasks.

**Logistic Regression** is a supervised learning technique particularly effective for **simple classification problems**, especially when the number of features is small. However, it comes with some notable limitations:

- **Multiclass limitation**:  
  Classic logistic regression is designed for **binary classification**. Extending it to more than two classes typically requires training a **separate model for each class** (a strategy called **one-vs-all classification**)

- **Scalability issues**:  
  Logistic regression does not scale well to more **complex classification tasks**, such as those encountered in **Computer Vision** (e.g., object detection or image recognition). These problems require more sophisticated classifiers, such as **neural networks** or **deep learning models**, capable of capturing higher-dimensional, non-linear relationships.

### Extending Logistic Regression to Neural Network

Logistic regression can be interpreted as a very simple neural network which consists of the following key components:

- **Neurons (nodes)**:  
  Each node in the network represents a computational unit, also known as a **neuron** or **perceptron**.

- **Connections (edges) with weights**:  
  The links between neurons indicate how much one neuron's output **contributes** to the next layer. Each connection has an associated **weight** that influences the calculation.

- **Activation function**:  
  This introduces **non-linearity** into the model. The most common activation functions are:
  - **Sigmoid**: $$\sigma(x) = \frac{1}{1 + e^{-x}}\space$$  --> the chosen one in this context
  - **ReLU**: $$\max(0, x)$$
  - **Hyperbolic Tangent**: $$\tanh(x)$$

- **Network structure**:  
  The model includes:
  - An **input layer** with multiple input neurons (each corresponding to a feature),
  - A **single output unit**, which represents the predicted class label (binary classification: 0 or 1).

Thus, logistic regression is structurally equivalent to a **single-layer neural network** with a sigmoid output.
This simple design can be **extended** by introducing:

- **Hidden layers** with intermediate neurons,
- An **increased number of output units** to match the number of classes in the classification task.

This leads to a **fully connected neural network** architecture.

#### Network Architecture Overview

- The network consists of **three layers**:
  - **Input layer**: size \( s_1 \)
  - **Hidden layer**: size \( s_2 \)
  - **Output layer**: size \( s_3 = K \), where \( K \) is the number of output classes

> In this specific case:
> - Input: 400 neurons (corresponding to the 20×20 grayscale image pixels)
> - Hidden: 25 neurons
> - Output: 10 neurons (for digits 0 through 9)

#### Key Concepts and Notation

- **Activation values**:
  - For the **input layer**, activations are the actual **input data**: the 400 pixel values.
  - For the **output layer**, activations are the **predictions** (probabilities for each class).
  - For **hidden and output layers**, activations are computed based on the activations from the **previous layer** (fully connected).

- **Weights (parameters)** between layers:
  - Between input and hidden layer:  
    \( \Theta^{(1)} \in \mathbb{R}^{s_2 \times (s_1 + 1)} \)
  - Between hidden and output layer:  
    \( \Theta^{(2)} \in \mathbb{R}^{s_3 \times (s_2 + 1)} \)

These matrices store the **learned parameters** (weights) of the network and include an extra column to handle the **bias unit**.


## Functions

Now that the theoretical background is fully documented, here are the necessary MATLAB functions:

#### 1. 

## Running the Tasks

- Check the `run_all_tasks.m` file and change marked parameters if desired (input file name, dataset split percentage, etc)
- Ensure that the input file exists at the specified path and is well-formatted
- From the MATLAB/GNU Octave Command Window, enter `run_all_tasks` and inspect the output
