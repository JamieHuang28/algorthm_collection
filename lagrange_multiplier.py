from newton_method import ProblemInterface, NewtonsMethod, GradDescent
import numpy as np

class FunctionQuadratic(ProblemInterface):
    def __init__(self, initial_guess: np.array):
        x, y = initial_guess[0:2]
        self.x = x
        self.y = y
    
    def __repr__(self) -> str:
        return "FunctionQuadratic("+str(self.x)+", "+str(self.y)+")"

    def eval(self):
        return self.x**2 + self.y**2
    
    def grad(self):
        return np.array([2.0*self.x, 2.0*self.y])
    
    def hessian(self):
        return np.eye(2) * 2.0
    
    def update_x(self, d_x):
        self.x += d_x[0]
        self.y += d_x[1]

class EqConstraint(ProblemInterface):
    def __init__(self, initial_guess: np.array):
        x, y = initial_guess[0:2]
        self.x = x
        self.y = y
    
    def __repr__(self) -> str:
        return "EqConstraint(x^2-3=0)"
    
    def eval(self):
        return self.x**2*self.y -3
    
    def grad(self):
        return np.array([2*self.x*self.y, self.x**2])
    
    def hessian(self):
        return np.array([
            [2*self.y, 2*self.x],
            [2*self.x, 0]
        ])
    
    def update_x(self, d_x):
        self.x += d_x[0]
        self.y += d_x[1]

class LagrangeMultiplier(ProblemInterface):
    """
    L(x, l) = f(x) + l*g(x)
    grad_L = [grad_f(x) + l*grad_g(x), g(x)]
    hessian_L = [
        [hessian_f(x)+l*hessian_g(x), grad_g(x)'],
        [grad_g(x), [0.0]]
    ]
    """
    def __init__(self, f_x: ProblemInterface, g_x: ProblemInterface, initial_guess: np.array, is_uneq_constr=False):
        self.f_x = f_x(initial_guess)
        self.g_x = g_x(initial_guess)
        self.lamb = initial_guess[-1]
        self.is_uneq_constr = is_uneq_constr
    
    def __repr__(self) -> str:
        return "L = {"+self.f_x.__repr__() + "+ lambda("+str(self.lamb)+")*"+self.g_x.__repr__()+"}"
    
    def eval(self):
        return self.f_x.eval() + self.lamb*self.g_x.eval()
    
    def grad(self):
        return np.concatenate([self.f_x.grad()+self.lamb*self.g_x.grad(), [self.g_x.eval()]])
    
    def hessian(self):
        num_raw_f_x = np.shape(self.f_x.hessian())[0]
        return np.block([
                [self.f_x.hessian()+self.lamb*self.g_x.hessian(), self.g_x.grad().reshape((num_raw_f_x, -1))],
                [self.g_x.grad(), 0.0]
            ])
    
    def update_x(self, d_x):
        self.f_x.update_x(d_x)
        self.g_x.update_x(d_x)
        if self.is_uneq_constr:
            self.lamb = max(0.0, self.lamb + d_x[-1]) # ensure that lamb >= 0
        else:
            self.lamb += d_x[-1]
    
class Problem(ProblemInterface):
    def __init__(self, initial_guess: np.array):
        self.object_func = FunctionQuadratic(initial_guess)
        self.eq_constr = EqConstraint(initial_guess)
        self.lamb = initial_guess[-1]
    
    def __repr__(self) -> str:
        return self.object_func.__repr__() +"+ lambda("+str(self.lamb)+")*"+self.eq_constr.__repr__()
    
    def eval(self):
        return self.object_func.eval() + self.lamb*self.eq_constr.eval()
    
    def grad(self):
        return np.concatenate((self.object_func.grad() + self.lamb*self.eq_constr.grad(), [self.eq_constr.eval()]))
    
    def hessian(self):
        # print(self.object_func.hessian() + self.lamb*self.eq_constr.hessian())
        # print(self.eq_constr.grad().transpose())
        # print(self.eq_constr.grad().reshape(2,1))
        return np.block([
            [self.object_func.hessian() + self.lamb*self.eq_constr.hessian(), self.eq_constr.grad().reshape(2,1)],
            [self.eq_constr.grad(), np.zeros(1)]
        ])
    
    def update_x(self, d_x: np.array):
        self.object_func.update_x(d_x)
        self.eq_constr.update_x(d_x)
        self.lamb += d_x[2]

class ProblemLegacy(ProblemInterface):
    def __init__(self, initial_guess: np.array):
        x, y, lamb = initial_guess[0:3]
        self.x = x
        self.y = y
        self.lamb = lamb
    
    def __repr__(self) -> str:
        return "[ProblemLegacy] "+"x:"+str(self.x)+",y:"+str(self.y)+",lamb:"+str(self.lamb)
    
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
    pb = Problem(np.ones(3))
    pb_legacy = ProblemLegacy(np.ones(3))
    print(pb.eval())
    print("legacy", pb_legacy.eval())
    print(pb.grad())
    print("legacy", pb_legacy.grad())
    print(pb.hessian())
    print("legacy", pb_legacy.hessian())
    pb.update_x(np.ones(3))
    pb_legacy.update_x(np.ones(3))
    print(pb.eval())
    print("legacy", pb_legacy.eval())

    NewtonsMethod(pb)

    print("----")
    pb_l = LagrangeMultiplier(FunctionQuadratic, EqConstraint, np.ones(3), False)
    print(pb_l.eval())
    print(pb_l.grad())
    print(pb_l.hessian())
    NewtonsMethod(pb_l)
