# Parameter Optimization for a Non-Linear Radar Power Model

[![Course](https://img.shields.io/badge/Course-Optimization_(MYE008)-blue.svg)]()
[![Institution](https://img.shields.io/badge/Institution-University_of_Ioannina-red.svg)]()
[![Language](https://img.shields.io/badge/Language-Python_3-yellow.svg)]()

## Overview
This repository contains the implementation and comparative analysis of various optimization algorithms used to estimate the parameters of a non-linear radar received power model. The project focuses on calibrating the model to match synthetic training data while ensuring robust generalization on unseen test data.

## Problem Formulation

### Parametric Radar Model
The received signal power (PW) is modeled as a function of target range (R), aspect angle (theta), air temperature (T), and atmospheric pressure (P):

$$PW(\beta,R,\theta,T,P)=\beta_{1}\frac{\cos^{2}\theta}{R^{4}}+\beta_{2}\exp(-\beta_{3}(T-15))+\beta_{4}\log(P)+\beta_{5}$$

### Search Space and Constraints
The parameter vector beta = ($\beta_1$, $\beta_2$, $\beta_3$, $\beta_4$, $\beta_5$) is subject to the following physical bounds:
- $\beta_1 \in [10^3, 10^6]$
- $\beta_2 \in [1, 100]$
- $\beta_3 \in [10^{-3}, 10^{-1}]$
- $\beta_4 \in [0.1, 3.0]$
- $\beta_5 \in [-120, -40]$

To improve numerical stability, all parameters are normalized to a [0, 1] interval during optimization.

### Objective Function
The goal is to minimize the Mean Squared Error (MSE) on the training set (N=400):

$$MSE_{train}(\beta)=\frac{1}{N}\sum_{i=1}^{N}(PW(\beta,R_{i},\theta_{i},T_{i},P_{i})-PW_{i})^{2}$$

## Implemented Algorithms
The following optimization methods were implemented from scratch:
1. **Newton Dogleg (NewtonDG):** A trust-region method combining Cauchy and Newton steps.
2. **BFGS with Strong Wolfe Conditions (BFGS-W):** A quasi-Newton method utilizing analytical gradients.
3. **Nelder-Mead Simplex (NM):** A heuristic, derivative-free search method.
4. **Genetic Algorithm (GA):** An evolutionary approach using tournament selection and arithmetic crossover.
5. **Particle Swarm Optimization (PSO):** A swarm intelligence algorithm.

### Computational Budgeting
To ensure parity, all algorithms are limited to **100,000 function evaluations**. Computational costs are tracked as follows:
- Objective function evaluation: 1 hit.
- Gradient vector calculation: 5 hits.
- Hessian matrix calculation: 15 hits.

## Repository Structure
- `algos.py`: Core implementation of the parametric model, budget tracking, and optimization algorithms.
- `analysis.py`: Statistical analysis script (Wilcoxon rank-sum tests, correlation coefficients).
- `radar_train.txt` & `radar_test.txt`: Data files for training and generalization testing.
- `initial_points.txt`: A file containing 30 pre-defined random starting points to ensure reproducibility.

## Usage

### 1. Run the Optimization Algorithms
This script tunes the hyperparameters for GA and PSO (using a small budget) and then executes 30 independent runs for all 5 algorithms using the full 100,000 evaluation budget.
```bash
python algos.py
```
*(Output: Generates text files for each algorithm, e.g., `Newton.txt`, `GA.txt`, containing the MSE scores, cost, and optimal $\beta$ parameters per run).*

### 2. Generate Statistics & Visualizations
Once the optimization phase is complete, run the analysis script to process the results, perform statistical tests, and generate visual plots.
```bash
python analysis.py
```
*(Output: Generates `descriptive_statistics.csv`, `comparison_matrix.csv`, `correlations.csv`, and a high-resolution `boxplots.png`).*

## Key Findings & Results

Based on non-parametric statistical testing (Wilcoxon rank-sum, $\alpha=0.05$) and experimental analysis across 30 independent runs:

* **Best Overall Performer:** The **Newton Dogleg (NewtonDG)** algorithm achieved the best balance. It produced excellent generalization results ($MSE_{test}$) while consuming a fraction of the computational budget compared to population-based methods.
* **Best Generalization Accuracy:** The **Genetic Algorithm (GA)** achieved the lowest median $MSE_{test}$ with minimal variance (extreme stability), though it consistently exhausted the 100,000 evaluation budget.
* **Correlation:** Population-based methods (PSO, NM) and NewtonDG showed a strong positive correlation between training and testing errors, indicating stable generalization without overfitting.
* **Underperforming Algorithm:** The **BFGS-W** algorithm struggled significantly, frequently trapping in local minima far from the optimal solution, highlighting its sensitivity to the highly non-linear nature of this specific problem.

## Author
* **Giannis Fillis**
* **Course:** Optimization (MYE008), 2025-2026
* **Institution:** Department of Computer Engineering and Informatics, University of Ioannina
