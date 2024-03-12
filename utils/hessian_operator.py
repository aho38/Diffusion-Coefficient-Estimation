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
        
    # define (Newton) Hessian apply H * v = y
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

    def eval(self, v, y):
        # incremental forward
        rhs = -(self.C * v)
        self.bc_adj.apply(rhs)
        dl.solve (self.A, self.du, rhs)
        
        # incremental adjoint
        rhs = -(self.W * self.du) -  self.Wum * v
        self.bc_adj.apply(rhs)
        dl.solve (self.adj_A, self.dp, rhs)
        
        # Reg/Prior term
        self.R.mult(v,y) # regularization
        y.axpy(1., self.Wmm*v) # e^m \tild{m} \hat{m} \nabla u * \nabla p
        
        # Misfit term
        self.C.transpmult(self.dp, self.CT_dp)
        y.axpy(1., self.CT_dp)
        self.Wum.transpmult(self.du, self.Wum_du)
        y.axpy(1., self.Wum_du)

    def inner(self, x, y):
        '''
        reutnr x^T * H(m) * y
        '''
        Hy = dl.Vector()
        self.A.init_vector(Hy,0)
        self.eval(y, Hy)
        return x.inner(Hy)

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
        dl.solve (self.A, self.du, rhs)
        
        # incremental adjoint
        rhs = - (self.W * self.du)
        dl.solve (self.adj_A, self.dp, rhs)
        
        # Reg/Prior term
        self.R.mult(v,y)
        y.axpy(1.,self.Wmm*v)
        
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
        dl.solve (self.adj_A, self.dp, rhs)
        
        # Reg/Prior term
        self.R.mult(v,y)
        
        # Misfit term
        self.C.transpmult(self.dp, self.CT_dp)
        y.axpy(1., self.CT_dp)
        self.Wum.transpmult(self.du, self.Wum_du)
        y.axpy(1., self.Wum_du)

class HessianOperator_comb():
    cgiter = 0
    def __init__(self, R, W, Wmm1, C1, A1, adj_A1, Wum1, Wmm2, C2, A2, adj_A2, Wum2, bc_adj, gauss_newton_approx=False):
        self.R = R
        self.W = W
        self.gauss_newton_approx = gauss_newton_approx
        self.bc_adj = bc_adj

        self.Wmm1 = Wmm1
        self.C1 = C1
        self.A1 = A1
        self.adj_A1 = adj_A1
        
        self.Wum1 = Wum1
        # self.gauss_newton_approx = gauss_newton_approx
        

        self.Wmm2 = Wmm2
        self.C2 = C2
        self.A2 = A2
        self.adj_A2 = adj_A2

        self.Wum2 = Wum2
        # self.gauss_newton_approx = gauss_newton_approx
    

        # incremental state
        self.du1 = dl.Vector()
        self.du2 = dl.Vector()
        self.A1.init_vector(self.du1,0)
        self.A2.init_vector(self.du2,0)
        
        # incremental adjoint
        self.dp1 = dl.Vector()
        self.dp2 = dl.Vector()
        self.adj_A1.init_vector(self.dp1,0)
        self.adj_A2.init_vector(self.dp2,0)
        
        # auxiliary vectors
        self.CT_dp = dl.Vector()
        self.C1.init_vector(self.CT_dp, 1)
        self.Wum_du1 = dl.Vector()
        self.Wum_du2 = dl.Vector()
        self.Wum1.init_vector(self.Wum_du1, 1)
        self.Wum2.init_vector(self.Wum_du2, 1)
        
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
        
    # define (Newton) Hessian apply H * v
    def mult_GaussNewton(self, v, y):
        
        # incremental forward
        rhs = -(self.C1 * v)
        self.bc_adj.apply(rhs)
        dl.solve (self.A1, self.du1, rhs)

        rhs = -(self.C2 * v)
        dl.solve (self.A2, self.du2, rhs)
        
        # incremental adjoint
        rhs = - (self.W * self.du1)
        self.bc_adj.apply(rhs)
        dl.solve (self.adj_A1, self.dp1, rhs)

        rhs = -(self.W * self.du2) #-  self.Wum2 * v
        dl.solve (self.adj_A2, self.dp2, rhs)

        # Reg/Prior term
        self.R.mult(v,y)
        y.axpy(1.,self.Wmm2*v)
        
        # Misfit term
        self.C1.transpmult(self.dp1, self.CT_dp)
        y.axpy(1, self.CT_dp)

        self.C2.transpmult(self.dp2, self.CT_dp)
        y.axpy(1., self.CT_dp)

        # self.Wum2.transpmult(self.du2, self.Wum_du2)
        # y.axpy(1., self.Wum_du2)

    def mult_Newton(self, v, y):
        # incremental forward
        rhs = -(self.C1 * v)
        self.bc_adj.apply(rhs)
        dl.solve (self.A1, self.du1, rhs)

        rhs = -(self.C2 * v)
        dl.solve (self.A2, self.du2, rhs)
        
        # incremental adjoint
        rhs = -(self.W * self.du1) -  self.Wum1 * v
        self.bc_adj.apply(rhs)
        dl.solve (self.adj_A1, self.dp1, rhs)

        rhs = -(self.W * self.du2) -  self.Wum2 * v
        dl.solve (self.adj_A2, self.dp2, rhs)
        
        # Reg/Prior term
        self.R.mult(v,y)
        y.axpy(1., self.Wmm1*v)
        
        # Misfit term
        self.C1.transpmult(self.dp1, self.CT_dp)
        y.axpy(1., self.CT_dp)
        self.C2.transpmult(self.dp2, self.CT_dp)
        y.axpy(1., self.CT_dp)
        self.Wum1.transpmult(self.du1, self.Wum_du1)
        y.axpy(1., self.Wum_du1)
        self.Wum2.transpmult(self.du2, self.Wum_du2)
        y.axpy(1., self.Wum_du2) 