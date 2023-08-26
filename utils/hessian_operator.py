from termios import VSTART
import dolfin as dl
import numpy as np

class HessianOperator():
    cgiter = 0
    def __init__(self, R, Wmm, C, A, adj_A, W, Wum, bc_adj, gauss_newton_approx=False):
        self.R = R
        self.Wmm = Wmm
        self.C = C
        self.A = A
        self.adj_A = adj_A
        self.W = W
        self.Wum = Wum
        self.gauss_newton_approx = gauss_newton_approx
        self.bc_adj = bc_adj
    

        # incremental state
        self.du = dl.Vector()
        self.A.init_vector(self.du,0)
        
        # incremental adjoint
        self.dp = dl.Vector()
        self.adj_A.init_vector(self.dp,0)
        
        # auxiliary vectors
        self.CT_dp = dl.Vector()
        self.C.init_vector(self.CT_dp, 1)
        self.Wum_du = dl.Vector()
        self.Wum.init_vector(self.Wum_du, 1)
        
    def init_vector(self, v, dim):
        # print('init:')
        # print(v)
        # print(dim)
        self.R.init_vector(v,dim)

    # Hessian performed on v, output as generic vector y
    def mult(self, v, y):
        # print('mult')
        # print(v.get_local())
        # print(y)
        self.cgiter += 1
        y.zero()
        if self.gauss_newton_approx:
            self.mult_GaussNewton(v,y)
        else:
            self.mult_Newton(v,y)
        # print('after')
        # print(v)
        # print(y)
            
    # define (Gauss-Newton) Hessian apply H * v
    def mult_GaussNewton(self, v, y):
        
        # incremental forward
        rhs = -(self.C * v)
        self.bc_adj.apply(rhs)
        dl.solve (self.A, self.du, rhs)
        
        # incremental adjoint
        rhs = - (self.W * self.du)
        self.bc_adj.apply(rhs)
        dl.solve (self.adj_A, self.dp, rhs)
        
        # Reg/Prior term
        self.R.mult(v,y)
        
        # Misfit term
        self.C.transpmult(self.dp, self.CT_dp)
        y.axpy(1, self.CT_dp)
        
    # define (Newton) Hessian apply H * v
    def mult_Newton(self, v, y):
        # incremental forward
        rhs = -(self.C * v)
        self.bc_adj.apply(rhs)
        dl.solve (self.A, self.du, rhs)
        
        # incremental adjoint
        rhs = -(self.W * self.du) -  self.Wum * v
        self.bc_adj.apply(rhs)
        dl.solve (self.adj_A, self.dp, rhs)
        
        # Reg/Prior term
        self.R.mult(v,y)
        y.axpy(1., self.Wmm*v)
        
        # Misfit term
        self.C.transpmult(self.dp, self.CT_dp)
        y.axpy(1., self.CT_dp)
        self.Wum.transpmult(self.du, self.Wum_du)
        y.axpy(1., self.Wum_du)

class HessianOperatorNBC():
    cgiter = 0
    def __init__(self, R, Wmm, C, A, adj_A, W, Wum, gauss_newton_approx=False):
        self.R = R
        self.Wmm = Wmm
        self.C = C
        self.A = A
        self.adj_A = adj_A
        self.W = W
        self.Wum = Wum
        self.gauss_newton_approx = gauss_newton_approx
        
        
        # incremental state
        self.du = dl.Vector()
        self.A.init_vector(self.du,0)
        
        # incremental adjoint
        self.dp = dl.Vector()
        self.adj_A.init_vector(self.dp,0)
        
        # auxiliary vectors
        self.CT_dp = dl.Vector()
        self.C.init_vector(self.CT_dp, 1)
        self.Wum_du = dl.Vector()
        self.Wum.init_vector(self.Wum_du, 1)
        
    def init_vector(self, v, dim):
        self.R.init_vector(v,dim)

    # Hessian performed on v, output as generic vector y
    def mult(self, v, y):
        self.cgiter += 1
        y.zero()
        if self.gauss_newton_approx:
            self.mult_GaussNewton(v,y)
        else:
            self.mult_Newton(v,y)
            
    # define (Gauss-Newton) Hessian apply H * v
    def mult_GaussNewton(self, v, y):
        
        # incremental forward
        rhs = -(self.C * v)
#         bc_adj.apply(rhs)
        dl.solve (self.A, self.du, rhs)
        
        # incremental adjoint
        rhs = - (self.W * self.du)
#         bc_adj.apply(rhs)
        dl.solve (self.adj_A, self.dp, rhs)
        
        # Reg/Prior term
        self.R.mult(v,y)
        
        # Misfit term
        self.C.transpmult(self.dp, self.CT_dp)
        y.axpy(1, self.CT_dp)
        
    # define (Newton) Hessian apply H * v
    def mult_Newton(self, v, y):
        
        # incremental forward
        rhs = -(self.C * v)
        dl.solve (self.A, self.du, rhs)
        
        # incremental adjoint
        rhs = -(self.W * self.du) -  self.Wum * v
#         bc_adj.apply(rhs)
        dl.solve (self.adj_A, self.dp, rhs)
        
        # Reg/Prior term
        self.R.mult(v,y)
        y.axpy(1., self.Wmm*v)
        
        # Misfit term
        self.C.transpmult(self.dp, self.CT_dp)
        y.axpy(1., self.CT_dp)
        self.Wum.transpmult(self.du, self.Wum_du)
        y.axpy(1., self.Wum_du)