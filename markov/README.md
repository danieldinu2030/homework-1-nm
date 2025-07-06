# Task 1 - Markov

## Table of Contents

- [Overview](#overview)
- [Theoretical References](#theoretical-references)

## Overview

The problem focuses on navigating a robot through a labyrinth, starting from a given position. The goal is to determine the probability that the robot 
will reach a **winning exit**, given that at each step it randomly selects one of the available directions (up, down, left, right — no diagonal moves allowed). 
The robot avoids revisiting previously visited cells.

There are two types of exits in the labyrinth:

- **Winning exits** (marked green): Located on the **top and bottom** edges of the labyrinth. Reaching one of these exits means the
  robot has successfully "escaped" with a winning probability of **1**.
- **Losing exits** (marked red): Found on the **left and right** edges. Reaching one of these ends the game in a loss, with a probability of **0**.

Walls may exist between adjacent cells, blocking movement between them. For example, moving directly from cell (1, 1) to (1, 2) might not be possible due to such walls.
The idea is to model the problem probabilistically and then apply heuristic methods (e.g., greedy algorithms based on computed probabilities) to guide the 
robot more efficiently toward a winning exit.

## Theoretical References

To model the given scenario, **Markov chains** are used—probabilistic models particularly valuable in fields such as economics, dynamic system reliability and 
artificial intelligence algorithms. One notable application of Markov chains is the **Google PageRank algorithm**.

In a Markov chain, the system is represented as a **directed graph**, where:
- **Nodes (states)** correspond to possible positions or conditions of the system.
- **Edges (transitions)** indicate non-zero probabilities of moving from one state to another.
- For any state, the sum of all outgoing transition probabilities is **1**.

In this problem:
- Each **labyrinth cell** corresponds to a **unique state** in the Markov chain.
- Two additional special states are introduced:
  - **WIN state**: The robot has reached a winning exit. It is an absorbing state (no further transitions).
  - **LOSE state**: The robot exits through a losing side. Also an absorbing state.

Thus, the labyrinth becomes a probabilistic model:
- **States** represent positions in the maze.
- **Transitions** represent possible valid moves with equal probabilities.
- The **Markov chain** is stored as a **weighted directed graph**, with weights corresponding to movement probabilities.

This abstraction allows us to analyze the robot's behavior using probability theory and efficiently calculate its chances of winning or losing.

