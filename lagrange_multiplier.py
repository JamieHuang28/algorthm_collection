from newton_method import ProblemInterface, NewtonsMethod, GradDescent
import numpy as np

class Problem(ProblemInterface):
    def __init__(self, x: float, y: float, lamb: float):
        self.x = x
        self.y = y
        self.lamb = lamb
        self.x_n = np.array([self.x, self.y, self.lamb])
    
    def eval(self):
        return self.x**2 + self.y **2 + self.lamb * (self.x**2 * self.y - 3)
    
    def grad(self):
        return  np.array([2 * self.x + self.lamb * 2 * self.x * self.y, 2 * self.y + self.lamb * self.x**2, self.x**2*self.y - 3])
    
    def hessian(self):
        return np.array([
            [2 + self.lamb * 2 * self.y, self.lamb * 2 * self.x, 2 * self.x * self.y],
            [self.lamb * 2 * self.x, 2, self.x**2],
            [2 * self.x*self.y, self.x**2, 0]
            ])
    
    def update_x(self, x):
        self.x = self.x + x[0]
        self.y = self.y + x[1]
        self.lamb = self.lamb + x[2]
        self.x_n = np.array([self.x, self.y, self.lamb])

if __name__ == "__main__":
    """
    Problem description:
    f(x) = x^2 + y^2
    g(x) = x^2y - 3

    min{f(x)}
    s.t. g(x) = 0

    Solving method:
    L(x, y, lambda) = f(x, y) + lambda * g(x, y)

    let nabla_{L(x, y)} = 0, which is equivalent to the equation collection
    partial_{L(x, y, lambda)}/partial_{x} = 0
    partial_{L(x, y, lambda)}/partial_{y} = 0
    partial_{L(x, y, lambda)}/partial_{lambda} = 0
    which is
    2x + lambda * 2xy = 0
    2y + lambda * x^2 = 0
    x^2y - 3 = 0
    
    """
    print(__name__)
    problem = Problem(0.1, 0.1, 0.0)
    NewtonsMethod(problem)
