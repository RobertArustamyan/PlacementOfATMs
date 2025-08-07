# PlacementOfATMs


[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](#)
[![Last Updated](https://img.shields.io/badge/updated-2025--08--07-orange)](#)

> A modular, extensible framework to solve the **Set Covering Problem (SCP)** using multiple **Branch and Bound** strategies in Python.

---

## Features

- Multiple **Branch and Bound** algorithms: DFS, BFS, Best-First Search
- SCP-specific bounding and pruning techniques
- Structured and modular codebase
- Configurable problem inputs
- Performance benchmarking with metrics like runtime, optimality, and search space reduction

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Algorithms](#algorithms)
- [Project Structure](#project-structure)
- [Benchmarking](#benchmarking)

---

## Overview

This project targets the **Set Covering Problem (SCP)**—a classic NP-hard problem common in resource allocation, logistics, and facility placement. The solution approach is based on **Branch and Bound (B&B)** algorithms, a powerful method for exploring combinatorial solution spaces by intelligently pruning suboptimal branches.

This implementation compares various B&B strategies to evaluate their performance, scalability, and practical applicability to real-world SCP instances.

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/RobertArustamyan/PlacementOfATMs.git
   cd PlacementOfATMs
   ```

2. (Optional but recommended) Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---



## Algorithms

### Implemented Branch and Bound Variants

| Algorithm         | Description                                                             |
|------------------|-------------------------------------------------------------------------|
| **DFS B&B**       | Depth-first traversal, minimal memory usage, deeper solutions faster     |
| **BFS B&B**       | Level-wise traversal, shallow optimal solutions but higher memory usage |
| **Best-First B&B**| Selects the most promising node based on heuristic lower bounds         |

Each algorithm is implemented modularly, allowing customization of:
- Node evaluation strategy
- Bounding function
- Branching order

---

## Project Structure

```
PlacementOfATMs/
├── BranchAndBound/        # Main B&B implementations
├── SCPData/               # Input datasets for SCP instances
├── algorithms/            # Helper functions and heuristics
├── models/                # Problem formulations and constraints
├── utils/                 # Logging, input parsing, visualization
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

---



Expected output (terminal/log):

```
Algorithm: Best-First Branch and Bound
Optimal Cost: 5
Selected Sets: ['B', 'E']
Nodes Expanded: 14
Pruned Branches: 9
Execution Time: 0.038s
```

---

## Benchmarking

Performance is evaluated on:

- Solution quality (optimality gap)
- Number of nodes expanded
- Total runtime
- Pruning effectiveness

Benchmark datasets are included in `SCPData/`. You can extend the tests with larger, real-world instances to analyze scalability.


---
