def f(x, a, b, c):
    return a * x ** 2 + b * x + c

def f_prime(x, a, b):
    return 2 * a * x + b

def newtons_method(a, b, c, initial_guess=0, tolerance=1e-7, max_iterations=1000):
    x_n = initial_guess
    for iter in range(max_iterations):
        # Calculate the value of the derivative at x_n
        f_prime_val = f_prime(x_n, a, b)
        
        # If the derivative is very close to zero, then we have found the minimum
        if abs(f_prime_val) < tolerance:
            print(f"Minimum found at x = {x_n}, f(x) = {f(x_n, a, b, c)}, iter = {iter}")
            return x_n
        
        # Since the second derivative of a quadratic function is constant,
        # we can directly calculate the next iteration without actually deriving it again.
        x_n = x_n - f_prime_val / (2 * a)
    
    print(f"Maximum number of iterations reached with x = {x_n}, f(x) = {f(x_n, a, b, c)}")
    return x_n

def example1():
    # Example usage:
    # Quadratic function coefficients: f(x) = x^2 - 4x + 3
    a = 1
    b = -4
    c = 3

    # Initial guess
    initial_guess = 0

    # Find the minimum
    minimum_x = newtons_method(a, b, c, initial_guess)
    return minimum_x

import numpy as np

class QuadFunc:
    def __init__(self):
        pass

    def eval(self, x: np.array):
        return np.inner(x, x)
    
    def partial(self, x: np.array):
        return np.array([2 * a for a in x])

    def partial2(self, x: np.array):
        return np.eye(np.shape(x)[0]) * 2.0

def newtons_method(func : QuadFunc, initial_guess: np.array, tolerance=1e-7, max_iterations=1000):
    x_n = initial_guess
    for iter in range(max_iterations):
        g = func.partial(x_n)
        H = func.partial2(x_n)
        d_x = - np.linalg.inv(H).dot(g)
        print(f"{iter}: d_x = {d_x}")
        if abs(np.linalg.norm(func.partial(x_n))) < tolerance:
            print(f"found at {x_n}, value = {func.eval(x_n)} , iter = {iter}")
            return x_n
        x_n = x_n + d_x
    
    print(f"maximum reached at {x_n}, value = {func.eval(x_n)}")
    return x_n

def example2():
    qf = QuadFunc()
    X = np.array([1, 2])
    print(qf.eval(X))
    print(qf.partial(X))
    print(qf.partial2(X))
    print(qf.partial(X) / qf.partial2(X))
    newtons_method(qf, X)

from abc import ABC, abstractclassmethod

class ProblemInterface(ABC):
    
    @abstractclassmethod
    def eval(self):
        pass

    @abstractclassmethod
    def grad(self):
        pass

    @abstractclassmethod
    def hessian(self):
        pass

    @abstractclassmethod
    def update_x(self, d_x: np.array):
        pass

def NewtonsMethod(problem : ProblemInterface, tolerance=1e-7, max_iterations=1000):
    for iter in range(max_iterations):
        grad = problem.grad()
        print(f"{iter}: grad = {grad}")
        if abs(np.linalg.norm(problem.grad())) < tolerance:
            x_n = problem.x_n
            print(f"found at {x_n}, value = {problem.eval()} , iter = {iter}")
            return x_n
        d_x = - np.linalg.inv(problem.hessian()).dot(problem.grad())
        problem.update_x(d_x)
    
    print(f"maximum reached at {problem.x_n}, value = {problem.eval()}")
    return problem.x_n

class QuadProblem(ProblemInterface):
    def __init__(self, initial_guess: np.array):
        self.x_n = initial_guess
        self.func = QuadFunc()
    
    def eval(self):
        return self.func.eval(self.x_n)
    
    def grad(self):
        return self.func.partial(self.x_n)
    
    def hessian(self):
        return self.func.partial2(self.x_n)
    
    def update_x(self, d_x: np.array):
        self.x_n = self.x_n + d_x


def exampleNewtons():
    problem = QuadProblem(np.array([1.0, 2.0]))
    print(problem.eval())
    print(problem.grad())
    print(problem.hessian())
    print(np.linalg.inv(problem.hessian()))
    print(problem.x_n)
    print(problem.update_x(np.ones(2)))
    print(problem.x_n)
    NewtonsMethod(problem)

def GradDescent(problem: ProblemInterface, step, tolerance=1e-7, max_iterations=1000):
    for iter in range(max_iterations):
        grad = problem.grad()
        print(f"{iter}: grad = {grad}")
        print(f"x_n = {problem.x_n}")
        if abs(np.linalg.norm(problem.grad())) < tolerance:
            x_n = problem.x_n
            print(f"found at {x_n}, value = {problem.eval()} , iter = {iter}")
            return x_n
        d_x = -problem.grad() * step
        problem.update_x(d_x)
    
    print(f"maximum reached at {problem.x_n}, value = {problem.eval()}")
    return problem.x_n

def exampleGradDescent():
    problem = QuadProblem(np.array([1.0, 2.0]))
    GradDescent(problem, 1e-1)
    
if __name__ == "__main__":
    exampleNewtons()
        