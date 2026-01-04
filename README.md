# Micro-Grad Engine

An implementation of Andrej Karpathy's micrograd from scratch. This is a minimal neural network and autodiff engine built entirely in Python.

## Project Structure

```
.
├── engine.py # Core autodiff engine (Value class)
├── model.py # Example usage of training loop
├── neural_network.py # Neuron, Layer, and MLP classes
├── README.md
└── requirements.txt # Python dependencies
```

## Features

- **Autograd Engine**
    - Supports addition, multiplication, power, tanh, ReLu, and more
    - Reverse mode automatic differentiation (backpropagation)
- **Neural Network Module**
    - Fully connected layers (`Layer`)
    - Multi-layer perceptrons (`MLP`)
- **Training Loop**
    - Vanilla stochastic gradient descent
    - Uses mean squared error loss

## Installation

1. Clone the repository:
```bash
git clone https://github.com/eko-071/micrograd-engine
cd micrograd-engine
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

- `numpy`

> This is a pure Python project; no heavy ML libraries are required.