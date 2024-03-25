from newton_method import ProblemInterface, NewtonsMethod
from lagrange_multiplier import Problem, LagrangeMultiplier

import numpy as np

class UneqConstraint(ProblemInterface):
    def __init__(self, initial_guess: np.array):
        x, y = initial_guess[0:2]
        self.x = x
        self.y = y
    
    def __repr__(self):
        return "Constraint(y - x + 3 <= 0)"
    
    def eval(self):
        return self.y - self.x + 3
    
    def grad(self):
        return np.array([-1, 1, 0.0])
    
    def hessian(self):
        return np.eye(3)*0.0
    
    def update_x(self, d_x):
        self.x += d_x[0]
        self.y += d_x[1]

if __name__ == "__main__":
    """
    Problem description:
    f(x, y) = x, y.dot(x, y)
    g(x, y) = x^2X[1] - 3
    h(x, y) = y - x + 3

    min{f(x)}
    s.t. g(x) = 0
    h(x) <= 0

    Dual Problem:
    L(x, lamb, miu) = f(x) + lamb * g(x) + miu * h(x)
    nabla_{L(x)} = 0
    s.t. miu >= 0

    Reference:
    f'(x, y) = 2 * x, y
    partial_g(x, y) / partial_x = 2 * x * y
    partial_g(x, y) / partial_y = x^2
    
    partial_h(x, y) / partial_x = -1
    partial_h(x, y) / partial_y = 1
    """
    pb_kkt = LagrangeMultiplier(Problem, UneqConstraint, np.ones(4), True)
    print(pb_kkt.eval())
    print(pb_kkt.grad())
    print(pb_kkt.hessian())
    NewtonsMethod(pb_kkt)