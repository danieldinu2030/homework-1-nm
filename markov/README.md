# Task 1 - Markov

## Table of Contents

- [Overview](#overview)
- [Theoretical References](#theoretical-references)

## Overview

The problem focuses on navigating a robot through a labyrinth, starting from a given position. The goal is to determine the probability that the robot 
will reach a **winning exit**, given that at each step it selects one of the available directions (up, down, left, right — no diagonal moves allowed). 
The robot avoids revisiting previously visited cells.

There are two types of exits in the labyrinth:

- **Winning exits** (marked green): Located on the **top and bottom** edges of the labyrinth. Reaching one of these exits means the
  robot has successfully "escaped" with a winning probability of **1**.
- **Losing exits** (marked red): Found on the **left and right** edges. Reaching one of these ends in a loss, with a probability of **0**.

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

This abstraction allows us to analyse the robot's behavior using probability theory and efficiently calculate its chances of winning or losing.
Below are the most important features of this way of modelling the problem.

### Adjacency Matrix for a Directed Graph

The **adjacency matrix** of a directed graph, similar to that of an undirected graph, is defined as:

$$
A = (A_{ij})_{i,j \in \{1, ..., n\}} \in \{0, 1\}^{n \times n}
$$

Where:

- $$\( A_{ij} = 1 \)$$, if there is a **transition** from state \( i \) to state \( j \),
- $$\( A_{ij} = 0 \)$$, otherwise.

In the context of the labyrinth:

- The submatrix \( A(1:n, 1:n) \) is **symmetric**,
- This symmetry exists because **walls are bidirectional** — if a transition from state \( i \) to state \( j \) is possible, then the reverse transition is also possible.

This matrix structure models **bidirectional movement** between adjacent, non-blocked cells in the maze.

### Link Matrix (Transition Probability Matrix)

The **link matrix** is a more powerful representation than the adjacency matrix. While structurally similar, the key difference lies in the meaning of the values it contains.

In a **link matrix**, each element represents the **transition probability** from one state to another in the Markov chain.
Using the notation $$p_{ij}$$, the matrix is defined as:

- $$L_{ij} = p_{ij}$$, if 0 < $$p_{ij}$$ ≤ 1
- $$L_{ij} = 0$$, otherwise

Notice that \( L \) is a **row-stochastic matrix**: the sum of each row equals 1.

### Reformulating the Problem as a Linear System

In addition to graph-based representations, the Markov chain describing the robot’s movement in a labyrinth can be reformulated as a **system of linear equations**.

Let $$p ∈ ℝ^{m·n}$$ be a vector of winning probabilities for each cell in the labyrinth, where `m` and `n` are the maze's dimensions. 
Each entry $$p_i$$ represents the probability that the robot, starting from state `i`, eventually reaches a winning exit.

For example, suppose the robot in state 1 can:
- Move to state 4 with a probability of 1/2, and
- Move to the WIN state (where the game is won) with a probability of 1/2.

Then the equation becomes:

$$p_1 = (1/2) * p_4 + (1/2) * p_{WIN} = (1/2) * p_4 + 1/2$$

Writing similar equations for all states results in a linear system, where the influence of WIN and LOSE states is encoded in the right-hand side. 

This system takes the form: `p = G·p + c`

- `G` is a matrix capturing transition probabilities between non-terminal states
- `c` is a vector containing contributions from terminal states (e.g., WIN with probability 1)
- `p` is the unknown vector we want to solve for

This form is particularly suitable for **iterative methods** such as the **Jacobi method**. The iteration step is defined as:

$$x_{k+1} = G·x_k + c$$

This iterative scheme efficiently computes the probability of reaching a winning exit from any given state, especially in large, sparse labyrinths.
