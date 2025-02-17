# Recurrence Mimicking Learning (RML)

This repository contains code for **Recurrence Mimicking Learning (RML)** experiments described in our article. The method aims to optimize a global reward (such as the Sharpe Ratio) by **mimicking** how recurrent decisions would unfold over a time series, but without incurring the usual high cost of repeated model executions.

## Method Overview

In brief, RML uses a single feedforward pass to generate actions for each time step _as if_ they were generated recurrently. It does so by stacking the input $X$ multiple times along all possible previous actions $a_{t-1}$. After generating a stacked output, a lightweight re-indexing step ($\phi$-processing) reconstructs a trajectory of decisions that mirrors the recurrent process. This allows a direct calculation of the global reward (e.g., Sharpe Ratio) with only two forward passes, rather than $T$ passes in a traditional offline RRL.


## Repository Structure

- **src/**  
  Source code with modules for data loading/preprocessing, different reinforcement learning methods (offline RRL, online RRL, RML), and separate training scripts.
- **pyproject.toml**  
  Project dependencies.
- **.gitignore**  
  Standard Python and OS ignore patterns.

## Minimal Usage Example

1. Install dependencies:
  ```bash
  bash install.sh
  ```
2. Jak can configure experiments with:
  ```bash
  config.py
  ```
3. To run comparison with Online RRL:
  ```bash
  python experiment.py
  ```
4. To run comparison with Offline RRL:
  ```bash
  python time_experiment.py
  ```

## Reference

For the complete description of the method, mathematical details, and experiments, see our article THE REFERENCE TO ADD