from newton_method import ProblemInterface
import numpy as np
import scipy.linalg

class SystemModel:
    def __init__(self, s_k: np.array, s_k_plus_one: np.array):
        delta_t = 0.1
        self.A = np.eye(2) + np.array([
            [0, 1.0],
            [0.0, 0.0]
        ])*delta_t
        self.B = np.array([0, 1.0]).reshape(2, 1)*delta_t
        self.setS(s_k, s_k_plus_one)
    
    def getABmat(self):
        return np.block([self.A, self.B])

    def getSubstractMat(self):
        return np.block([-1.0*np.eye(2), np.zeros(2).reshape((2,1))])
    
    def validate(self, sub_problem_idx = 0):
        """
        sub_problem_idx is the partial object, i.e x, v.
        Note that input is not constraint, so sub_problem_idx < dim(X)
        """
        # print(self.u)
        # print(self.getStates())
        # self.u = np.ones(1)
        # print(self.getStates())
        return (self.getABmat().dot(self.s_k.reshape(self.s_k.size, 1))\
            +self.getSubstractMat().dot(self.s_k_plus_one.reshape(self.s_k_plus_one.size,1)))[sub_problem_idx]

    def grad(self, sub_problem_idx = 0):
        # not necessarily blocked beforehand
        return self.getABmat()[sub_problem_idx], self.getSubstractMat()[sub_problem_idx]
    
    def hessian(self, sub_problem_idx = 0):
        # luckily its all zero, but not inversable
        return np.zeros(18).reshape(2,3,3)[sub_problem_idx], np.zeros(18).reshape(2,3,3)[sub_problem_idx]
    
    def setS(self, s_k: np.array, s_k_plus_one: np.array):
        self.s_k = s_k
        self.s_k_plus_one = s_k_plus_one

class SystemModelConstraints(ProblemInterface):
    """
    model(x_k, x_{k+1}) is the elemental relation.
    for single dimension such as x, the expected output should be [[0],[0],...],
    but it doesn't matter.
    for calculation, the expected gradient w.r.t state s is
    [grad(s_0, model_0)+grad(s_0, model_1), grad(s_1, model_1), grad(s_1, model_2), ...]
    = [(model(s_0,s_1).grad()[1]+model(s_1,s_2).grad()[0]), (model(s_1,s_2).grad()[1]+model(s_2,s_3).grad()[0]), ...]
    = [[0,0]^T, [0,0]^T, ...] (function group in 2x1)
    """
    def __init__(self, kNumStates=2, sub_problem_idx = 0):
        self.sub_problem_idx = sub_problem_idx
        self.kNumStates = kNumStates
        self.ss = []

        ss_list = []
        for i in range(self.kNumStates):
            ss_list.append(np.ones(3))
        self.ss = np.array(ss_list)
        self.models = []
        for i in range(1, self.kNumStates):
            self.models.append(SystemModel(self.ss[i-1], self.ss[i]))
    
    def eval(self):
        return np.array([model.validate(self.sub_problem_idx) for model in self.models])
    
    def grad(self):
        grads = []
        grads.append(self.models[0].grad(self.sub_problem_idx)[0])
        for i in range(1, len(self.models)):
            model_0 = self.models[i-1]
            model_1 = self.models[i]
            grads.append(model_0.grad(self.sub_problem_idx)[1]+model_1.grad(self.sub_problem_idx)[0])
        grads.append(self.models[-1].grad(self.sub_problem_idx)[1])
        # return np.array(grads)
        return scipy.linalg.block_diag(*grads)
    
    def hessian(self):
        hessians = []
        hessians.append(self.models[0].hessian(self.sub_problem_idx)[0])
        for i in range(1, len(self.models)):
            model_0 = self.models[i-1]
            model_1 = self.models[i]
            hessians.append(model_0.hessian(self.sub_problem_idx)[1]+model_1.hessian(self.sub_problem_idx)[0])
        hessians.append(self.models[-1].hessian(self.sub_problem_idx)[1])
        
        hessians_blocked = [scipy.linalg.block_diag(*np.array([block, np.eye(block.shape[0])*0.0])) for block in hessians]
        return np.array(hessians_blocked)
    
    def update_x(self, d_x: np.array, sub_problem_idx = 0):
        self.ss[:, sub_problem_idx] += d_x
        for i in range(1, len(self.models)):
            self.models[i].setS(self.ss[i-1], self.ss[i])

class ElementMaxConstraint(ProblemInterface):
    """
    value function is in the form of []
    """
    def __init__(self, max: float, x = 0.0):
        self.x = x
    
    def eval(self):
        return self.x - max
    
    def grad(self):
        return 1.0
    
    def hessian(self):
        return 0.0
    
    def update_x(self, d_x: float):
        self.x += d_x

class ElementMinConstraint(ProblemInterface):
    def __init__(self, min: float, x = 0.0):
        self.x = x
    
    def evel(self):
        return -self.x + min
    
    def grad(self):
        return -1.0
    
    def hessian(self):
        return 0.0
    
    def update_x(self, d_x: float):
        self.x += d_x


if __name__ == "__main__":
    # s_0 = np.ones(3)
    # s_1 = np.ones(3)
    # sys_model = SystemModel(s_0, s_1)
    # print(sys_model.A)
    # print(sys_model.B)
    # print(sys_model.validate())
    # print(sys_model.grad())
    # print(sys_model.hessian())
    model_constr = SystemModelConstraints(2, 0)
    print(model_constr.eval())
    print("----grad of all xs")
    print(model_constr.grad())
    print(model_constr.grad().dot(model_constr.ss.reshape(-1,1)))
    print("expected form is [[grad_s_0(x_0)], [grad_s_0_partial(x_1)]]")

    print("----update_x at xs")
    print(model_constr.ss)
    model_constr.update_x(np.ones(2)*0.2, 0)
    print(model_constr.ss)
    print(model_constr.grad())
    print("----hessian")
    print(model_constr.hessian())
    print(model_constr.hessian().dot(model_constr.ss.reshape(-1,1)))
    print("expected form is [[hessian_s_0(s_0)], [hessian_s_0(s_1)]]")