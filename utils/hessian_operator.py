from termios import VSTART
import dolfin as dl
import numpy as np
from petsc4py import PETSc

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
        # apply 0 bc to mhat or v.
        # self.bc_adj.apply(v)
        
        # incremental forward
        C = self.C.copy()
        self.bc_adj.apply(C)

        rhs = -(C * v)
        self.bc_adj.apply(rhs)
        self.bc_adj.apply(self.A)
        dl.solve (self.A, self.du, rhs)
        
        # incremental adjoint
        W = self.W.copy()
        self.bc_adj.apply(W)

        rhs = - (W * self.du)
        self.bc_adj.apply(rhs)
        self.bc_adj.apply(self.adj_A)
        dl.solve (self.adj_A, self.dp, rhs)
        
        # Reg/Prior term
        self.R = self.apply_symm_bc_2_matrix(self.R)
        self.R.mult(v,y)
        
        # Misfit term
        C.transpmult(self.dp, self.CT_dp)
        y.axpy(1, self.CT_dp)
        
    # define (Newton) Hessian apply H * v = y
    def mult_Newton(self, v, y):
        # apply 0 bc to mhat or v.
        # self.bc_adj.apply(v)

        # incremental forward
        C = self.C.copy()
        self.bc_adj.apply(C)# this feels unnecessary here but is needed when do C.transpmult(dp, CT_dp) since we need it to be 0 on the boundary and dp satisfies the homogeneous boundary condition

        rhs = -(C * v)
        self.bc_adj.apply(rhs)
        self.bc_adj.apply(self.A) # this is not necessary because A is coming from fwd solver which applys bc already
        dl.solve (self.A, self.du, rhs)
        
        # incremental adjoint
        W = self.W.copy()
        self.bc_adj.apply(W)
        Wum = self.Wum.copy()
        self.bc_adj.apply(Wum)

        rhs = -(W * self.du) -  Wum * v
        self.bc_adj.apply(rhs)
        self.bc_adj.apply(self.adj_A)
        dl.solve (self.adj_A, self.dp, rhs)
        
        
        # Reg/Prior term
        RpWmm = (self.R + self.Wmm)
        RpWmm = self.apply_symm_bc_2_matrix(RpWmm)
        RpWmm.mult(v, y)
        # y.axpy(1., Wmm*v)
        
        # Misfit term
        C.transpmult(self.dp, self.CT_dp)
        y.axpy(1., self.CT_dp)
        Wum.transpmult(self.du, self.Wum_du)
        y.axpy(1., self.Wum_du)

    def apply_symm_bc_2_matrix(self, df_mat):
        '''
        Since dl keeps throwing me error I will just apply the boundary condition that preserve the symmetry by hand through PETSc
        '''
        df_mat_array = df_mat.array()

        df_mat_array[:,0] = df_mat_array[:,-1] = df_mat_array[0,:] = df_mat_array[-1,:] = 0.0

        df_mat_array[0,0] = df_mat_array[-1,-1] = 1.0
        
        pet_mat = PETSc.Mat()
        pet_mat.createDense(df_mat_array.shape, array = df_mat_array)
        return dl.PETScMatrix(pet_mat)

    def eval(self, v, y): # solve Av = y
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

    def array(self):
        H_array = self.R.array() + self.Wmm.array()
        matrix_of_du = -np.linalg.inv(self.A.array()) @ self.C.array()
        matrix_of_dp = np.linalg.inv(self.adj_A.array()) @ (-self.W.array()@ matrix_of_du - self.Wum.array())
        H_array += self.C.array().T @ matrix_of_dp + self.Wum.array().T @ matrix_of_du
        return H_array



    def inner(self, x, y):
        '''
        reutnr x^T * H(m) * y
        '''
        Hy = dl.Vector()
        self.A.init_vector(Hy,0)
        if self.gauss_newton_approx:
            self.mult_GaussNewton(y,Hy)
        else:
            self.mult_Newton(y,Hy)
        return x.inner(Hy)
    
    def hess_sym_check(self,Vm):
        from utils.random import generate_random_vector
        '''
        Creat x and y vector and compute y^T H x and x^T H y. See if they are equal
        '''
        xx = dl.Function(Vm).vector()
        yy = dl.Function(Vm).vector()
        xx = generate_random_vector(xx,0.,1,1)
        yy = generate_random_vector(yy,0.,1,2)

        xtHy = self.inner(xx, yy)
        ytHx = self.inner(yy, xx)

        symm_diff = abs(xtHy-ytHx)

        return symm_diff

class HessianOperatorNBC():
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
        C = self.C.copy()

        rhs = -(C * v)
        dl.solve (self.A, self.du, rhs)
        
        # incremental adjoint
        W = self.W.copy()

        rhs = - (W * self.du)
        dl.solve (self.adj_A, self.dp, rhs)
        
        # Reg/Prior term
        RpWmm = (self.R + self.Wmm)
        RpWmm = self.apply_symm_bc_2_matrix(RpWmm)
        RpWmm.mult(v,y)
        
        # Misfit term
        C.transpmult(self.dp, self.CT_dp)
        y.axpy(1, self.CT_dp)
        
    # define (Newton) Hessian apply H * v
    def mult_Newton(self, v, y):
        
        # incremental forward
        C = self.C.copy()

        rhs = -(C * v)
        dl.solve (self.A, self.du, rhs)
        
        # incremental adjoint
        W = self.W.copy()
        Wum = self.Wum.copy()

        rhs = -(W * self.du) -  Wum * v
        dl.solve (self.adj_A, self.dp, rhs)
        
        # Reg/Prior term
        RpWmm = (self.R + self.Wmm)
        RpWmm = self.apply_symm_bc_2_matrix(RpWmm)
        RpWmm.mult(v,y)
        
        # Misfit term
        C.transpmult(self.dp, self.CT_dp)
        y.axpy(1., self.CT_dp)
        Wum.transpmult(self.du, self.Wum_du)
        y.axpy(1., self.Wum_du)

    def apply_symm_bc_2_matrix(self, df_mat):
        '''
        Since dl keeps throwing me error I will just apply the boundary condition that preserve the symmetry by hand through PETSc
        '''
        df_mat_array = df_mat.array()

        df_mat_array[:,0] = df_mat_array[:,-1] = df_mat_array[0,:] = df_mat_array[-1,:] = 0.0

        df_mat_array[0,0] = df_mat_array[-1,-1] = 1.0
        
        pet_mat = PETSc.Mat()
        pet_mat.createDense(df_mat_array.shape, array = df_mat_array)
        return dl.PETScMatrix(pet_mat)
    
    def eval(self, v, y):
        
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
    
    def inner(self, x, y):
        '''
        reutnr x^T * H(m) * y
        '''
        Hy = dl.Vector()
        self.A.init_vector(Hy,0)
        if self.gauss_newton_approx:
            self.mult_GaussNewton(y,Hy)
        else:
            self.mult_Newton(y,Hy)
        return x.inner(Hy)
    
    def hess_sym_check(self,Vm):
        from utils.random import generate_random_vector
        '''
        Creat x and y vector and compute y^T H x and x^T H y. See if they are equal
        '''
        xx = dl.Function(Vm).vector()
        yy = dl.Function(Vm).vector()
        xx = generate_random_vector(xx,0.,1,1)
        yy = generate_random_vector(yy,0.,1,2)

        xtHy = self.inner(xx, yy)
        ytHx = self.inner(yy, xx)

        symm_diff = abs(xtHy-ytHx)

        return symm_diff

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
        self.CT_dp1 = dl.Vector()
        self.CT_dp2 = dl.Vector()
        self.C1.init_vector(self.CT_dp1, 1)
        self.C2.init_vector(self.CT_dp2, 1)
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
        C1 = self.C1.copy()
        self.bc_adj.apply(C1)
        rhs = -(C1 * v)
        A1 = self.A1.copy()
        self.bc_adj.apply(rhs)
        self.bc_adj.apply(A1)
        dl.solve (A1, self.du1, rhs)

        C2 = self.C2.copy()
        self.bc_adj.apply(C2)
        rhs = -(self.C2 * v)
        self.bc_adj.apply(rhs)
        A2 = self.A2.copy()
        self.bc_adj.apply(A2)
        dl.solve (A2, self.du2, rhs)
        
        # incremental adjoint
        W = self.W.copy()
        self.bc_adj.apply(W)
        rhs = - (W * self.du1)
        self.bc_adj.apply(rhs)
        self.bc_adj.apply(self.adj_A1)
        dl.solve (self.adj_A1, self.dp1, rhs)

        rhs = -(self.W * self.du2) #-  self.Wum2 * v
        self.bc_adj.apply(rhs)
        adj_A2 = self.adj_A2.copy()
        self.bc_adj.apply(adj_A2)
        dl.solve (adj_A2, self.dp2, rhs)

        # Reg/Prior term
        R = self.R
        self.bc_adj.apply(R)
        R.mult(v,y)
        Wmm2 = self.Wmm2.copy()
        self.bc_adj.apply(Wmm2)
        y.axpy(1.,Wmm2*v)
        
        # Misfit term
        C1 = self.C1.copy()
        self.bc_adj.apply(C1)
        C1.transpmult(self.dp1, self.CT_dp1)
        y.axpy(1., self.CT_dp1)

        C2 = self.C2.copy()
        self.bc_adj.apply(C2)
        C2.transpmult(self.dp2, self.CT_dp2)
        y.axpy(1., self.CT_dp2)

        # self.Wum2.transpmult(self.du2, self.Wum_du2)
        # y.axpy(1., self.Wum_du2)

    def Newton_DBC(self, v, y):
        # incremental forward
        C1 = self.C1.copy()
        self.bc_adj.apply(C1)# this feels unnecessary here but is needed when do C.transpmult(dp, CT_dp) since we need it to be 0 on the boundary and dp satisfies the homogeneous boundary condition
        rhs = -(C1 * v)
        self.bc_adj.apply(rhs)
        self.bc_adj.apply(self.A1)
        dl.solve (self.A1, self.du1, rhs)

        # incremental adjoint
        W = self.W.copy()
        self.bc_adj.apply(W)
        Wum1 = self.Wum1.copy()
        self.bc_adj.apply(Wum1)
        rhs = -(W * self.du1) -  Wum1 * v
        self.bc_adj.apply(rhs)
        self.bc_adj.apply(self.adj_A1)
        dl.solve (self.adj_A1, self.dp1, rhs)

        

        # Misfit term
        C1.transpmult(self.dp1, self.CT_dp1)
        y.axpy(1., self.CT_dp1)
        Wum1.transpmult(self.du1, self.Wum_du1)
        y.axpy(1., self.Wum_du1)

    def Newton_NBC(self, v, y):
        # incremental forward
        C2 = self.C2.copy()
        rhs = -(C2 * v)
        dl.solve (self.A2, self.du2, rhs)

        # incremental adjoint
        W = self.W.copy()
        Wum2 = self.Wum2.copy()
        rhs = -(W * self.du2) -  Wum2 * v
        dl.solve (self.adj_A2, self.dp2, rhs)

        # Misfit term
        C2.transpmult(self.dp2, self.CT_dp2)
        Wum2.transpmult(self.du2, self.Wum_du2)
        y.axpy(1., self.CT_dp2)
        y.axpy(1., self.Wum_du2)

    def mult_Newton(self, v, y):
        # self.W.init_vector(y,0)
        self.Newton_DBC(v, y)
        self.Newton_NBC(v, y)

        # Reg/Prior term
        RpWmm = self.R + self.Wmm1 + self.Wmm2
        RpWmm = self.apply_symm_bc_2_matrix(RpWmm)
        y.axpy(1.,RpWmm*v)

    def inner(self, x, y, Hess_type='exact', m=None):
        '''
        return x^T * H(m) * y
        '''
        Hy = dl.Vector()
        self.R.init_vector(Hy,0)
        if Hess_type == 'exact':
            self.mult_Newton(y, Hy)
        elif Hess_type == 'approx':
            self.mult_GaussNewton(y, Hy)
        elif Hess_type == 'eval':
            Hy= self.eval(m, y)
            return x.inner(Hy)
        return x.inner(Hy)


    def apply_symm_bc_2_matrix(self, df_mat):
        '''
        Since dl keeps throwing me error I will just apply the boundary condition that preserve the symmetry by hand through PETSc
        '''
        df_mat_array = df_mat.array()

        df_mat_array[:,0] = df_mat_array[:,-1] = df_mat_array[0,:] = df_mat_array[-1,:] = 0.0

        df_mat_array[0,0] = df_mat_array[-1,-1] = 1.0
        
        pet_mat = PETSc.Mat()
        pet_mat.createDense(df_mat_array.shape, array = df_mat_array)
        return dl.PETScMatrix(pet_mat)
    
    
    def hess_sym_check(self,Vm):
        from utils.random import generate_random_vector
        '''
        Creat x and y vector and compute y^T H x and x^T H y. See if they are equal
        '''
        xx = dl.Function(Vm).vector()
        yy = dl.Function(Vm).vector()
        xx = generate_random_vector(xx,0.,1,1)
        yy = generate_random_vector(yy,0.,1,2)

        xtHy = self.inner(xx, yy)
        ytHx = self.inner(yy, xx)

        symm_diff = abs(xtHy-ytHx)

        return symm_diff