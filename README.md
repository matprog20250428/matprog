# PyTorch Optimization Visualizer

This project implements and visualizes various gradient descent optimization algorithms using PyTorch. It allows users to compare the performance of different optimizers on various 1D and 2D functions and explore the effects of hyperparameter tuning.

## Description

The core of the project lies in implementing several optimization algorithms from scratch, inheriting from a base `Optimiser` abstract class. These optimizers are then used to minimize different mathematical functions, and their paths are visualized in 3D space using Plotly. The project also includes functionality to animate the optimization process and test the impact of varying hyperparameters like learning rate and momentum coefficients.

## Features

### Implemented Optimizers:

* **Vanilla SGD**: Basic Stochastic Gradient Descent.
* **SignSGD**: SGD variant using the sign of the gradient.
* **Momentum SGD**: Incorporates momentum to accelerate convergence.
* **Nesterov Accelerated Gradient (NAG)**: A modification of momentum SGD with improved convergence properties.
* **Quasi-Hyperbolic Momentum (QHM)**: A weighted average of momentum and plain SGD.
* **RMSprop**: Uses a moving average of squared gradients to adapt the learning rate.
* **AdaGrad**: Adapts the learning rate based on the historical sum of squared gradients.
* **AdaDelta**: An extension of AdaGrad that seeks to reduce its aggressive, monotonically decreasing learning rate.
* **Adam**: Combines ideas from RMSprop and Momentum.
* **Nadam**: Adam incorporating Nesterov momentum.
* **AMSGrad**: A variant of Adam aiming to fix convergence issues.
* **Barzilai-Borwein (BB)**: A two-point step size gradient method.
* **Muon**: Uses the Newton-Schulz iteration for matrix inversion approximation within the optimization step (based on arXiv:2502.16982v1).

### Visualization:

* **Static 3D Plots**: Uses Plotly to generate interactive 3D surface plots showing the loss landscape and the paths taken by different optimizers.
* **Animated 3D Plots**: Creates animations showing the step-by-step progression of each optimizer towards the minimum.

### Test Functions:

The optimizers are tested on various functions, including:
* Simple quadratic functions ($f(x) = x^2$, $f(x,y) = x^2 + y^2/2$)
* Functions with oscillations ($f(x) = x^2 + sin(10x)$)
* Himmelblau's function
* Functions with local minima and saddle points

## Dependencies

* Python 3.x
* PyTorch
* NumPy
* Pandas
* Matplotlib (for 1D plots)
* Plotly (for 3D plots and animations)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/matprog20250428/matprog
    cd matprog
    ```
2.  **Install dependencies:**
    ```bash
    pip install torch numpy pandas matplotlib plotly
    ```

## Usage

1.  **Run the Jupyter Notebook:**
    Open and run the `main.ipynb` notebook using Jupyter Lab or Jupyter Notebook or another IDE.
2.  **Define Optimizers:** Instantiate the desired optimizer classes with specific hyperparameters.
3.  **Define a Function:** Create a Python function that takes a PyTorch tensor (representing the parameters) and returns a scalar loss value.
4.  **Run Optimization:** Use the `run_optimization` or `run_optimizations` function to run one or multiple optimizers on the defined function.
    ```python
    # Example: Run multiple optimizers on function f
    optimizers_to_run = {
        "Adam": Adam(ZERO_ZERO, lr=0.1),
        "RMSprop": RMSprop(ZERO_ZERO, lr=0.1)
    }
    results = run_optimizations(f, N=50, optims=optimizers_to_run)
    ```
5.  **Visualize Results:** Use the `plot` function for static 3D visualization or the `animate` function for an animated view.
    ```python
    # Example: Plot results
    plot(results, func=f)

    # Example: Animate results
    animate(results)
    ```
6.  **Hyperparameter Tuning:** Modify the hyperparameters (e.g., learning rate, momentum coefficient) when creating optimizer instances and re-run the optimization and visualization steps to observe the effects.

## Code Structure

* **Imports and Setup**: Imports necessary libraries and sets up the device (CPU/GPU).
* **Optimizer Classes**: Defines the base `Optimiser` class and implementations for various algorithms.
* **1-D Examples**: Simple examples demonstrating optimizer behavior on 1D functions using Matplotlib.
* **Optimization Loop**: Contains the `run_optimization` function to execute the optimization process for a given optimizer and function.
* **Running Multiple Optimizers**: Defines a dictionary of optimizers and the `run_optimizations` function to run and collect results for multiple algorithms.
* **Visualization Functions**: Contains `plot` and `animate` functions using Plotly for 3D visualization.
* **Hyperparameter Experiments**: Sections dedicated to testing different learning rates and momentum values.
* **Testing on Other Surfaces**: Applies the optimizers to more complex functions like Himmelblau's function and functions with saddle points or multiple local minima.
