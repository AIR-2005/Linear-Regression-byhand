import numpy as np

class LinearRegModel:
    # Initialize the model
    def __init__(self):
        self.w = 0
        self.b = 0
    # Lineer Regression Function
    def predict(self, x):
        return (self.w * x) + self.b
    # Cost Function
    def calc_cost(self, x, y):
        # Number of x values
        m = x.shape[0]
        total = 0
        for i in range(m):
            total += (self.predict(x[i]) - y[i]) ** 2
        return total / (2 * m)
    # Gradient function for w and b
    def calc_grad(self, x, y):
        # Number of x values
        m = x.shape[0]
        # Gradient of w
        total = 0
        for i in range(m):
            total += (self.predict(x[i]) - y[i]) * x[i]
        grad_w = total / m

        # Gradient of b
        total = 0
        for i in range(m):
            total += (self.predict(x[i]) - y[i])
        grad_b = total / m

        # Return gradients
        return grad_w, grad_b

    # Update weight and bias
    def update(self, x, y, lr):
        grad_w, grad_b = self.calc_grad(x, y)
        self.w = self.w - (lr * grad_w)
        self.b = self.b - (lr * grad_b)

    # Train
    def train(self, x, y, lr, step):
        for e in range(step):
            # Update
            self.update(x, y, lr)
            # Calculate cost
            cost = self.calc_cost(x, y)
            if (e + 1) % 25 == 0:
                print(f"Step: {e + 1} | w: {self.w:.2f} | b: {self.b:.2f} | Cost: {cost:.2f}")