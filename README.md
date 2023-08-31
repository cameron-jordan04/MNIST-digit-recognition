## MNIST-digit-recognition

Each training image is 28 x 28 pixels (where each pixel has a value between 0 and 255) = 784 total pixels

Each training images can be represented as a row (where each row is 784 columns long) in a matrix: 
$`X \ = \ \begin{bmatrix}x^{(1)}\\ x^{(2)}\\ :\\x^{(n)}\end{bmatrix}`$

We can take the transpose of this matrix, s.t. each column now represents an example: 
$'X^T \ = \ \begin{bmatrix}x^{(1)} & x^{(2)} & ... & x^{(n)}\end{bmatrix}'$

The neural networks of digit classification will have one hidden layers: The input $0th$ layer will have 784 nodes (each pixel maps to a node), the hidden $1st$ layer will have 10 nodes and the output $2nd$ layer will have 10 nodes.

#### Forward propagation: 
$$Input \ Layer: \ A^{(0)} \ = \ \overrightarrow{x}$$ $$ Unactivated \ First \ Layer: \ Z^{(1)} \ = \ W^{(1)} \cdot A^{(0)} \ + \ b^{(1)}$$ $$ where \ W \ is \ a \ weight \ matrix \ and \ b \ is \ a \ bias \ term$$
$$ Activation \ Function: \ A^{(1)} \ = \ g(Z^{(1)}) \ = \ ReLU(Z^{(1)}) $$
$$where \ \begin{equation} ReLU(x) = \begin{cases} x \ if \ x \gt 0 \\ 0 \ if \ x \leq 0 \end{cases} \end{equation}$$

$ReLU$ stands for Rectified Linear Unit and is a common weight function used in neural networks. Other common functions include the sigmoid function and the hyperbolic tangent function.
$$Unactivated \ Second \ Layer: \ Z^{(2)} \ = \ W^{(2)} \cdot A^{(1)} \ + \ b^{(2)}$$
$$Activation \ Function: \ A^{(2)} \ = \ softmax(Z^{(2)})$$
$$where \ softmax(z_i) \ = \ \frac {e^{z_i}} {\sum_{j=1} ^K e^{z_j}}$$
$softmax$ generates a probability vector for each node in the neural network.

#### Backwards Propagation
$$Error \ of \ the \ 2nd \ layer \ Z^{(2)}: \ dZ^{2} \ = \ A^{(2)} \ - \ Y$$
$$ Error \ of \ the \ weight \ matrix \ W^{(2)}: \ dW^{(2)} \ = \ \frac {1} {m} dZ^{(2)} \cdot A^{(1) \ T}$$
$$Error \ of \ the \ bias \ b^{(2)}: \ db^{(2)} \ = \ \frac {1} {m} \sum dZ^{(2)}$$
$$Error \ of \ the \ 1st \ layer \ Z^{(1)}: \ dZ^{(1)} \ = \ W^{(2) \ T} \cdot dZ^{(2)} \ \star \ g'(Z^{(2)})$$
$$Error \ of \ the \ weight \ matrix \ W^{(1)}: \ dW^{(1)} \ = \ \frac {1} {m} dZ^{(1)} \cdot X^T $$
$$Error \ of \ the \ bias \ b^{(1)}: \ db^{(1)} \ = \ \frac {1} {m} \sum dZ^{(1)}$$

#### Update Parameters
$$W^{(1)} \ := \ W^{(1)} - \alpha  \ dW^{(1)}$$
$$b^{(1)} \ := \ b^{(1)} \ - \ \alpha \ db^{(1)}$$
$$W^{(2)} \ := \ W^{(2)} \ - \ \alpha \ dW^{(2)}$$
$$b^{(2)} \ := \ b^{(2)} \ - \ \alpha \ db^{(2)}$$
$$where \ \alpha \ is \ the \ learning \ rate \ of \ the \ neural \ network$$
