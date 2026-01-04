from engine import Value
from neural_network import Neuron, Layer, MLP

x_s = [
    [4.0, 1.0, -2.0],
    [3.0, -1.0, 0.5],
    [0.5, 3.0, 2.0],
    [2.0, 1.0, -1.0]
]
y_s = [1.0, -1.0, -1.0, 1.0]
# Binary classifier

n = MLP(3, [4, 4, 1])

for k in range(30):
    # Forward pass
    y_predictions = [n(x)[0] for x in x_s]
    loss = sum([(Value(y_truth)-y_prediction)**2 for y_truth, y_prediction in zip(y_s, y_predictions)])
    # Backward pass
    n.zero_grad()
    loss.backward()
    # Updating values
    for p in n.parameters():
        p.data += -0.05 * p.grad
    
    print(k, loss.data)

print(y_predictions)
