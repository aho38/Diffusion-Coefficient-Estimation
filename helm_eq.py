## ============= imports =============
from cgitb import reset
from re import L
import dolfin as dl
import pandas as pd
import ufl
import numpy as np
import sys
import os
from hippylib import *
import json
from pathlib import Path
import logging
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from utils.general import increment_path, DboundaryL, DboundaryR, Dboundary, NboundaryL, NboundaryR, data_normalization

dl.set_log_active(False)


class single_data_run():
    def __init__(self, 
                nx, #"node number"
                a, #'left boundary',
                b, #'right boundary',
                bc_type, #'NBC or DBC',
                omega, # exchange rate
                oce_val, # ocean value
                normalized_mean, #'mean for normalization
                func='Lagrange',
                ) -> None:
        self.a, self.b, self.nx = a, b, nx
        self.mesh = dl.IntervalMesh(nx, a, b)
        self.Vm = self.Vu = dl.FunctionSpace(self.mesh, func, 1)

        self.u_trial, self.p_trial, self.m_trial = dl.TrialFunction(self.Vu), dl.TrialFunction(self.Vu), dl.TrialFunction(self.Vm)
        self.u_test, self.p_test, self.m_test = dl.TestFunction(self.Vu), dl.TestFunction(self.Vu), dl.TestFunction(self.Vm)
        self.bc_type = bc_type

        self.omega_val = dl.Constant(omega)
        self.omega = omega
        self.u_oce_val = dl.Constant(f'{oce_val - normalized_mean}')
        self.u_oce = oce_val
        self.normalized_mean = normalized_mean
    

    def state_adj_str_setup(self, a_state_str, L_state_str, a_adj_str, L_adj_str):
        self.a_state_str, self.L_state_str = a_state_str, L_state_str
        self.a_adj_str, self.L_adj_str = a_adj_str, L_adj_str
        
    def misfit_reg_setup(self,gamma):
        self.gamma = gamma
        # weak for for setting up the misfit and regularization compoment of the cost
        W_equ   = ufl.inner(self.u_trial, self.u_test) * ufl.dx
        R_equ   = gamma * ufl.inner(ufl.grad(self.m_trial), ufl.grad(self.m_test)) * ufl.dx

        self.W = dl.assemble(W_equ)
        self.R = dl.assemble(R_equ)

    def f_setup(self, f_type #'type of forcing: str or ndarray'
                ):
        if isinstance(f_type, str):
            f = dl.Expression(f_type, degree=5)
            f = dl.interpolate(f,self.Vm)
        if isinstance(f_type, np.ndarray):
            f = dl.Function(self.Vu)
            f.vector().set_local(f_type[::-1])
        self.f = f
    
    def extra_f_setup(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def mtrue_setup(self,mtrue=None):
        self.mtrue = mtrue
        
    def BC_setup(self,u0L, u0R):
        if self.bc_type == 'DBC':
            self.DBC_setup(u0L, u0R)
        elif self.bc_type == 'NBC':
            self.NBC_setup(u0L, u0R)

    def DBC_setup(self,u0L, u0R):
        bc_state = [dl.DirichletBC(self.Vu, u0L, DboundaryL), dl.DirichletBC(self.Vu, u0R, DboundaryR)]
        bc_adj = dl.DirichletBC(self.Vu, dl.Constant(0.), Dboundary)
        self.bc_state, self.bc_adj = bc_state, bc_adj

    def NBC_setup(self,u0L, u0R):
        boundaries = dl.MeshFunction("size_t", self.mesh, self.mesh.topology().dim()-1)
        # boundaries = dl.FacetFunction()
        left = NboundaryL()
        right = NboundaryR()
        # mark left and right boundary
        left.mark(boundaries, 0)
        right.mark(boundaries, 1)
        # define boundary
        ds = dl.Measure('ds')(subdomain_data=boundaries)
        j0 = dl.Constant(f'{u0L}') # left boundary
        j1 = dl.Constant(f'{u0R}') # right boundary
        self.ds, self.j0, self.j1 = ds, j0, j1
    
    def data_setup(self, PATH = None, ud_array=None, ud = None, data_index=2, normalize=False):
        '''
        data_index = {'temperature': 2, 'salinity': 3,'oxygem': 4}
        '''
        if PATH is not None:
            df_dict = pd.read_excel(r"lake_data/2008_10_14_initialData.xls", header=None)
            data_value = df_dict[data_index].to_numpy()
        elif ud_array is not None:
            data_value = ud_array
        
        if normalize:
            data_value = data_normalization(data_value, self.normalized_mean, data_value.ptp()) #ptp point-to-point, a.k.a. range

        ud = dl.Function(self.Vu)
        ud.vector().set_local(data_value[::-1])
        self.ud = ud
    
    
    def fwd_solve(self, 
            m, #'Fenics function of diffusion coefficient',
            ):
        u = dl.Function(self.Vu)
        if self.bc_type == 'NBC':
            a_state = eval(self.a_state_str)
            L_state = eval(self.L_state_str)

            state_A, state_b = dl.assemble_system (a_state, L_state)
            dl.solve (state_A, u.vector(), state_b)
        
        if self.bc_type == 'DBC':
            a_state = eval(self.a_state_str)
            L_state = eval(self.L_state_str)

            state_A, state_b = dl.assemble_system (a_state, L_state, self.bc_state)
            dl.solve (state_A, u.vector(), state_b)

        return u, state_A, state_b

    def adj_solve(self,
                m, #'diffusion coefficietn',
                u, #'fwd solution',
                ):
        p = dl.Function(self.Vu)
        # weak form for setting up the adjoint equation
        a_adj = eval(self.a_adj_str)
        L_adj = eval(self.L_adj_str)

        if self.bc_type == 'NBC':
            adjoint_A, adjoint_RHS = dl.assemble_system(a_adj, L_adj)
            dl.solve(adjoint_A, p.vector(), adjoint_RHS)

        if self.bc_type == 'DBC':
            adjoint_A, adjoint_RHS = dl.assemble_system(a_adj, L_adj, self.bc_adj)
            dl.solve(adjoint_A, p.vector(), adjoint_RHS)
        
        return p, adjoint_A, adjoint_RHS


    def opt_loop(self, 
                m, #'diffusion coef',
                tol, #'tolerance of grad',
                c, #'constant of armijo linesearch',
                maxiter, #'maximum newton"s step',
                plot_opt_step = False,
                ):
        # initialize iter counters
        iter = 1
        total_cg_iter = 0
        alpha_iter = 0
        converged = False
        g, m_delta = dl.Vector(), dl.Vector()
        m_prev = dl.Function(self.Vm)

        # initial guess values
        u, state_A, state_b = self.fwd_solve(m)
        [cost_old, misfit_old, reg_old] = self.cost(u, m)
        
        
        self.R.init_vector(m_delta,0)
        self.R.init_vector(g,0)

        print ("Nit   CGit   cost          misfit        reg           sqrt(-G*D)    ||grad||       alpha  tolcg")
        gradnorm_list = []
        self.eigval_list = []
        # try:
        while iter <  maxiter and not converged:

            p, adjoint_A, _ = self.adj_solve(m, u)

            # assemble matrix C
            Wum_equ, C_equ, Wmm_equ, M_equ = self.wf_matrices_setup(m,u,p)
            C =  dl.assemble(C_equ)

            # assemble matrix M
            if iter == 1:
                self.M = dl.assemble(M_equ) 

            # assemble W_ua and R
            Wum = dl.assemble (Wum_equ)
            Wmm = dl.assemble (Wmm_equ)

            # evaluate the  gradient
            CT_p = dl.Vector()
            C.init_vector(CT_p,1)
            C.transpmult(p.vector(), CT_p)
            # print(np.linalg.norm(CT_p.get_local()))
            MG = CT_p + self.R * m.vector()
            dl.solve(self.M, g, MG)

            # calculate the norm of the gradient
            grad2 = g.inner(MG)
            gradnorm = math.sqrt(grad2)

            # set the CG tolerance (use Eisenstat–Walker termination criterion)
            if iter == 1:
                gradnorm_ini = gradnorm
            tolcg = min(0.5, math.sqrt(gradnorm/(gradnorm_ini+1e-15)))

            # define the Hessian apply operator (with preconditioner)
            if self.bc_type == 'DBC':
                from utils.hessian_operator import HessianOperator
                Hess_Apply = HessianOperator(self.R, Wmm, C, state_A, adjoint_A, self.W, Wum, self.bc_adj, gauss_newton_approx=(iter<6) )
                self.Hess = Hess_Apply
            if self.bc_type == 'NBC':
                from utils.hessian_operator import HessianOperatorNBC as HessianOperator
                Hess_Apply = HessianOperator(self.R, Wmm, C, state_A, adjoint_A, self.W, Wum, gauss_newton_approx=(iter<6) )
                self.Hess = Hess_Apply
            P = self.R + self.gamma * self.M
            Psolver = dl.PETScKrylovSolver("cg", amg_method())
            Psolver.set_operator(P)

            solver = CGSolverSteihaug()
            solver.set_operator(Hess_Apply)
            solver.set_preconditioner(Psolver)
            solver.parameters["rel_tolerance"] = tolcg
            solver.parameters["zero_initial_guess"] = True
            solver.parameters["print_level"] = -1

            # Plotting the Eigenvalues
            k = self.nx
            p = self.nx

            solver.solve(m_delta, -MG)
            # print(np.linalg.norm(m_delta.get_local()))
            total_cg_iter += Hess_Apply.cgiter

            # linesearch
            alpha = 1
            descent = 0
            no_backtrack = 0
            m_prev.assign(m)
            while descent == 0 and no_backtrack < 10:
                m.vector().axpy(alpha, m_delta )

                # solve the state/forward problem
                u, state_A, state_b = self.fwd_solve(m)
                dl.solve(state_A, u.vector(), state_b)

                # evaluate cost
                [cost_new, misfit_new, reg_new] = self.cost(u, m)

                # check if Armijo conditions are satisfied
                if cost_new < cost_old + alpha * c * MG.inner(m_delta):
                    misfit_old = misfit_new
                    reg_old = reg_new
                    cost_old = cost_new
                    descent = 1
                else:
                    no_backtrack += 1
                    alpha *= 0.5
                    m.assign(m_prev)  # reset a

            # calculate sqrt(-G * D)
            graddir = math.sqrt(- MG.inner(m_delta) )

            if plot_opt_step:
                from utils.plot import plot_step_single
                plot_step_single(u.compute_vertex_values(), self.ud.compute_vertex_values(), 
                          m.compute_vertex_values(), self.mtrue.compute_vertex_values(), self.gamma)
                print ("Nit   CGit   cost          misfit        reg           sqrt(-G*D)    ||grad||       alpha  tolcg")


            sp = ""
            gradnorm_list += [gradnorm]
            print( "%2d %2s %2d %3s %8.5e %1s %8.5e %1s %8.5e %1s %8.5e %1s %8.5e %1s %1.2e %1s %5.3e" % \
                (iter, sp, Hess_Apply.cgiter, sp, cost_old, sp, misfit_old, sp, reg_old, sp, \
                graddir, sp, gradnorm, sp, alpha, sp, tolcg) )

            if alpha < 1e-2:
                alpha_iter += 1
            if alpha_iter > 4:
                print('Alpha too small: ')
                break

            # check for convergence
            if gradnorm < tol and iter > 1:
                converged = True
                print( "Newton's method converged in ",iter,"  iterations")
                print( "Total number of CG iterations: ", total_cg_iter)

            iter += 1
        if not converged:
            print( "Newton's method did not converge in ", maxiter, " iterations")

        self.save_dict = {'gamma_val': self.gamma,
                    'misfit': misfit_old,
                    'reg': reg_old,
                    'u_sol': u.compute_vertex_values().tolist(),
                    'u_d': self.ud.compute_vertex_values().tolist(),
                    'm_sol': m.compute_vertex_values().tolist(),
                    'forcing': self.f.compute_vertex_values(self.mesh).tolist(),
                    'omega': self.omega,
                    'gradnorm': gradnorm,
                    'u_oce': self.u_oce,
                    'gradnorm_list': gradnorm_list,
                    }
        if hasattr(self,'mtrue'):
            self.save_dict['m_true'] = self.mtrue.compute_vertex_values().tolist()
        # except:
        #     print('Something Went Wrong')
        
        return m, u

    
    def wf_matrices_setup(self,m, u, p):
        # weak form for setting up matrices
        Wum_equ = ufl.inner(ufl.exp(m) * self.m_trial * ufl.grad(self.p_test), ufl.grad(p)) * ufl.dx
        C_equ   = ufl.inner(ufl.exp(m) * self.m_trial * ufl.grad(u), ufl.grad(self.u_test)) * ufl.dx
        Wmm_equ = ufl.inner(ufl.exp(m) * self.m_trial * self.m_test *  ufl.grad(u),  ufl.grad(p)) * ufl.dx
        M_equ   = ufl.inner(self.m_trial, self.m_test) * ufl.dx
        return Wum_equ, C_equ, Wmm_equ, M_equ
    
    def cost(self, u, m):
        diff = u.vector() - self.ud.vector()
        reg = m.vector().inner(self.R*m.vector() ) 
        misfit = diff.inner(self.W * diff)
        return [0.5 * reg + 0.5 * misfit, misfit, reg]
    
    def eigenvalue_request(self, m, p=20):
        k = self.nx
        P = self.R + self.gamma * self.M
        Psolver = dl.PETScKrylovSolver("cg", amg_method())
        Psolver.set_operator(P)

        import hippylib as hp
        Omega = hp.MultiVector(m.vector(), k + p)
        hp.parRandom.normal(1., Omega)
        lmbda, evecs = hp.doublePassG(self.Hess, P, Psolver, Omega, k)
        return lmbda, evecs
    
class dual_data_run():
    def __init__(self, 
                nx, 
                a, 
                b,
                gamma1,
                gamma2,
                omega1,
                omega2,
                oce_val1,
                oce_val2,
                normalized_mean1,
                normalized_mean2,
                func='Lagrange',
                save_path=None,
                ):
        self.a, self. b, self.nx = a, b, nx
        self.mesh = dl.IntervalMesh(nx, a, b)
        self.Vm = self.Vu = dl.FunctionSpace(self.mesh, func, 1)

        self.u_trial, self.p_trial, self.m_trial = dl.TrialFunction(self.Vu), dl.TrialFunction(self.Vu), dl.TrialFunction(self.Vm)
        self.u_test, self.p_test, self.m_test = dl.TestFunction(self.Vu), dl.TestFunction(self.Vu), dl.TestFunction(self.Vm)

        self.omega_val1 = dl.Constant(omega1)
        self.omega_val2 = dl.Constant(omega2)
        self.omega1 = omega1
        self.omega2 = omega2
        self.u_oce_val1 = dl.Constant(f'{oce_val1 - normalized_mean1}')
        self.u_oce1 = oce_val1
        self.u_oce_val2 = dl.Constant(f'{oce_val2 - normalized_mean2}')
        self.u_oce2 = oce_val2
        self.normalized_mean1 = normalized_mean1
        self.normalized_mean2 = normalized_mean2

        self.gamma1 = gamma1
        self.gamma2 = gamma2
        ## ===== path for saving results and optimization data ======
        self.save_path = save_path


    def gamma_setup(self, beta1, beta2, gamma=None):
        if gamma is not None:
            self.gamma = gamma
        else:
            self.gamma = np.exp(beta1*np.log(self.gamma1) + beta2*np.log(self.gamma2))

    def misfit_reg_setup(self, beta1, beta2, gamma = None):
        # constant in from of the misfit and weakform
        self.gamma_setup(beta1, beta2, gamma)
        self.beta1 = dl.Constant(beta1)
        self.beta2 = dl.Constant(beta2)
        # weak for for setting up the misfit and regularization compoment of the cost
        W_equ   = ufl.inner(self.u_trial, self.u_test) * ufl.dx(self.mesh)
        R_equ   = self.gamma * ufl.inner(ufl.grad(self.m_trial), ufl.grad(self.m_test)) * ufl.dx(self.mesh)

        self.W = dl.assemble(W_equ)
        self.R = dl.assemble(R_equ)

    def state_adj_str_setup(self, a1_state_str, L1_state_str, a1_adj_str, L1_adj_str, a2_state_str, L2_state_str, a2_adj_str, L2_adj_str):
        self.a1_state_str = a1_state_str
        self.L1_state_str = L1_state_str
        self.a1_adj_str = a1_adj_str
        self.L1_adj_str = L1_adj_str

        self.a2_state_str = a2_state_str
        self.L2_state_str = L2_state_str
        self.a2_adj_str = a2_adj_str
        self.L2_adj_str = L2_adj_str

    def fwd_solve(self,m):
        u1 = dl.Function(self.Vu)
        u2 = dl.Function(self.Vu)

        ## ======= Dirichlet BC ========
        a1_state = eval(self.a1_state_str)
        L1_state = eval(self.L1_state_str)

        state_A1, state_b1 = dl.assemble_system (a1_state, L1_state, self.bc_state)
        dl.solve (state_A1, u1.vector(), state_b1)

        ## ======= Neumann BC ========
        a2_state = eval(self.a2_state_str)
        L2_state = eval(self.L2_state_str)

        state_A2, state_b2 = dl.assemble_system (a2_state, L2_state)
        dl.solve (state_A2, u2.vector(), state_b2)
        
        
        return u1, u2, state_A1, state_b1, state_A2, state_b2
    
    def adj_solve(self,m, u1, u2):
        p1 = dl.Function(self.Vu)
        p2 = dl.Function(self.Vu)
        
        ## ====== Dirichelt ========
        a_adj1 = eval(self.a1_adj_str)
        L1_adj = eval(self.L1_adj_str)

        adjoint_A1, adjoint_RHS1 = dl.assemble_system(a_adj1, L1_adj, self.bc_adj)
        dl.solve(adjoint_A1, p1.vector(), adjoint_RHS1)
        ## ====== Neumann   ========
        a_adj2 = eval(self.a2_adj_str)
        L2_adj = eval(self.L2_adj_str)

        adjoint_A2, adjoint_RHS2 = dl.assemble_system(a_adj2, L2_adj)
        dl.solve(adjoint_A2, p2.vector(), adjoint_RHS2)

        return p1, p2, adjoint_A1, adjoint_RHS1, adjoint_A2, adjoint_RHS2

    def mtrue_setup(self,mtrue=None):
        self.mtrue = mtrue

    def BC_setup(self, u1L_Dirichlet, u1R_Dirichlet, u2L_Neumann, u2R_Neumann):
        self.DBC_setup(u1L_Dirichlet, u1R_Dirichlet)
        self.NBC_setup(u2L_Neumann, u2R_Neumann)
    
    def DBC_setup(self,u0L, u0R):
        bc_state = [dl.DirichletBC(self.Vu, u0L, DboundaryL), dl.DirichletBC(self.Vu, u0R, DboundaryR)]
        bc_adj = dl.DirichletBC(self.Vu, dl.Constant(0.), Dboundary)
        self.bc_state, self.bc_adj = bc_state, bc_adj

    def NBC_setup(self,u0L, u0R):
        boundaries = dl.MeshFunction("size_t", self.mesh, self.mesh.topology().dim()-1)
        # boundaries = dl.FacetFunction()
        left = NboundaryL()
        right = NboundaryR()
        # mark left and right boundary
        left.mark(boundaries, 0)
        right.mark(boundaries, 1)
        # define boundary
        ds = dl.Measure('ds')(subdomain_data=boundaries)
        j0 = dl.Constant(f'{u0L}') # left boundary
        j1 = dl.Constant(f'{u0R}') # right boundary
        self.ds, self.j0, self.j1 = ds, j0, j1

    def data_setup(self, ud1_array, ud2_array, normalize =False):
        if normalize:
            data1 = data_normalization(ud1_array, self.normalized_mean1, ud1_array.ptp())
            data2 = data_normalization(ud2_array, self.normalized_mean2, ud2_array.ptp())
        else:
            data1, data2 = ud1_array, ud2_array

        ud1 = dl.Function(self.Vu)
        ud1.vector().set_local(data1[::-1])
        ud2 = dl.Function(self.Vu)
        ud2.vector().set_local(data2[::-1])

        self.ud1 = ud1
        self.ud2 = ud2
    
    def f_setup(self, f_type #'type of forcing: str or ndarray'
                ):
        if isinstance(f_type, str):
            f = dl.Expression(f_type, degree=5)
            f = dl.interpolate(f,self.Vm)
        if isinstance(f_type, np.ndarray):
            f = dl.Function(self.Vu)
            f.vector().set_local(f_type[::-1])
        self.f = f

    def extra_f_setup(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def opt_loop(self, 
                m, #'diffusion coef',
                tol, #'tolerance of grad',
                c, #'constant of armijo linesearch',
                maxiter, #'maximum newton"s step',
                save_opt_log = True,
                plot_opt_step = False,
                plot_eigval = False,
                ):
        # initialize iter counters
        iter = 1
        total_cg_iter = 0
        alpha_iter = 0
        converged = False
        g, m_delta = dl.Vector(), dl.Vector()
        m_prev = dl.Function(self.Vm)

        # initial guess values
        u1, u2, state_A1, state_b1, state_A2, state_b2 = self.fwd_solve(m)
        [cost_old, misfit_old, reg_old] = self.cost(u1, u2, m)
        
        
        self.R.init_vector(m_delta,0)
        self.R.init_vector(g,0)

        ## create log path for optimization
        if save_opt_log:
            log_path = self.save_path / 'log_opt/'
            if not log_path.exists():
                log_path.mkdir()
            
            opt_csv_name = f'{self.gamma:1.3e}_opt_log.csv'
            csv_path = log_path / opt_csv_name
            if os.path.exists(csv_path):
                pass
            else:
                import csv
                with open(csv_path, 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Nit', 'CGit', 'cost', 'misfit', 'reg', 'sqrt(-G*D)', '||grad||', 'alpha', 'toldcg'])

        print ("Nit   CGit   cost          misfit        reg           sqrt(-G*D)    ||grad||       alpha  tolcg")
        gradnorm_list = []
        # try:
        while iter <  maxiter and not converged:
            p1, p2, adjoint_A1, _, adjoint_A2, _ = self.adj_solve(m, u1, u2)

            # assemble matrix C
            Wum_equ1, C_equ1, Wmm_equ1, Wum_equ2, C_equ2, Wmm_equ2, M_equ = self.wf_matrices_setup(m,u1, u2, p1, p2)
            C1 =  dl.assemble(C_equ1)
            C2 =  dl.assemble(C_equ2)

            # print(np.linalg.norm(C1.array()))
            # print(np.linalg.norm(C2.array()))

            # assemble matrix M
            if iter == 1:
                self.M = dl.assemble(M_equ)
            
            # assemble W_ua and R
            Wum1 = dl.assemble (Wum_equ1)
            Wmm1 = dl.assemble (Wmm_equ1)

            Wum2 = dl.assemble (Wum_equ2)
            Wmm2 = dl.assemble (Wmm_equ2)

            # print(np.linalg.norm(Wum1.array()))
            # print(np.linalg.norm(Wmm1.array()))

            # print(np.linalg.norm(Wum2.array()))
            # print(np.linalg.norm(Wmm2.array()))

            # evaluate the  gradient
            CT_p = dl.Vector()
            CT_p1 = dl.Vector()
            CT_p2 = dl.Vector()

            C1.init_vector(CT_p,0)

            C1.transpmult(p1.vector(), CT_p1)
            C2.transpmult(p2.vector(), CT_p2)
            # print(np.linalg.norm(CT_p1.get_local()))
            # print(np.linalg.norm(CT_p2.get_local()))
            CT_p.axpy(1., CT_p1)
            CT_p.axpy(1., CT_p2)

            MG = CT_p + self.R * m.vector()
            dl.solve(self.M, g, MG)

            # calculate the norm of the gradient
            grad2 = g.inner(MG)
            gradnorm = math.sqrt(grad2) 

            # set the CG tolerance (use Eisenstat–Walker termination criterion)
            if iter == 1:
                gradnorm_ini = gradnorm
            
            gradnorm_rel = gradnorm / gradnorm_ini
            tolcg = min(0.5, (gradnorm_rel))

            # define the Hessian apply operator (with preconditioner)
            from utils.hessian_operator import HessianOperator_comb as HessianOperator
            Hess_Apply = HessianOperator(self.R, self.W, Wmm1, C1, state_A1, adjoint_A1, Wum1, 
                                         Wmm2, C2, state_A2, adjoint_A2, Wum2, self.bc_adj, gauss_newton_approx=(iter<6))
            self.Hess = Hess_Apply
        
            # P = self.R + self.gamma * M
            P = self.R + self.gamma * self.M
            Psolver = dl.PETScKrylovSolver("cg", amg_method())
            Psolver.set_operator(P)

            if iter == 1: 
                k = self.nx
                p = 20

                import hippylib as hp
                Omega = hp.MultiVector(m.vector(), k + p)
                hp.parRandom.normal(1., Omega)
                self.lmbda, _ = hp.doublePassG(Hess_Apply, P, Psolver, Omega, k)

            solver = CGSolverSteihaug()
            solver.set_operator(Hess_Apply)
            solver.set_preconditioner(Psolver)
            solver.parameters["rel_tolerance"] = tolcg
            solver.parameters["zero_initial_guess"] = True
            solver.parameters["print_level"] = -1
            
            solver.solve(m_delta, -MG)
            total_cg_iter += Hess_Apply.cgiter

            # linesearch
            alpha = 1
            descent = 0
            no_backtrack = 0
            m_prev.assign(m)

            while descent == 0 and no_backtrack < 10:
                m.vector().axpy(alpha, m_delta )

                # solve the state/forward problem
                
                u1, u2, state_A1, state_b1, state_A2, state_b2 = self.fwd_solve(m)

                # evaluate cost
                [cost_new, misfit_new, reg_new] = self.cost(u1, u2, m)

                # check if Armijo conditions are satisfied
                if cost_new < cost_old + alpha * c * MG.inner(m_delta):
                    misfit_old = misfit_new
                    reg_old = reg_new
                    cost_old = cost_new
                    descent = 1
                else:
                    no_backtrack += 1
                    alpha *= 0.5
                    m.assign(m_prev)  # reset a
                # alpha = 0.5
            
            if plot_eigval:
                lmbda, evec = self.eigenvalue_request(m)
                idx = 10
                # plt.semilogy(lmbda[:idx], '*')
                plt.plot(evec[0])
                plt.show()
            
            ## Plot opt step
            if plot_opt_step:
                from utils.plot import plot_step
                plot_step(u1.compute_vertex_values(), u2.compute_vertex_values(), 
                          self.ud1.compute_vertex_values(), self.ud2.compute_vertex_values(), 
                          m.compute_vertex_values(), self.mtrue.compute_vertex_values(), self.gamma,0.5,0.5)
                print ("Nit   CGit   cost          misfit        reg           sqrt(-G*D)    ||grad||      ||grad_rel||     alpha  tolcg")
            
            ## Optimization of beta1 and beta2
            # diff1 = u1.vector() - self.ud1.vector()
            # diff2 = u2.vector() - self.ud2.vector()
            # diff1_scalar = diff1.inner(self.W * diff1)
            # diff2_scalar = diff2.inner(self.W * diff2)

            # reg_for_beta_opt = dl.assemble(ufl.inner(ufl.grad(m), ufl.grad(m)) * ufl.dx)

            # def betas_func(x):
            #     gamma = np.exp(x[0]*np.log(self.gamma1) + x[1]*np.log(self.gamma2))
            #     misfit_val = x[0] * diff1_scalar + x[1] * diff2_scalar
            #     reg_val = gamma * reg_for_beta_opt
            #     return 0.5 * misfit_val + 0.5 * reg_val
            
            # def constraint(x):
            #     return x[0] + x[1] - 1
            # bounds = [(0.1, 0.9), (0.1, 0.9)]
            # constraints = ({'type': 'eq', 'fun': constraint})
            # x0 = np.array([self.beta1.values()[0], self.beta2.values()[0]])
            # res = minimize(betas_func, x0, tol=1e-12, constraints=constraints, bounds=bounds)
            # beta1, beta2 = res.x
            # print("beta1, beta2: ", res.x)
            # print(sum(res.x))
            # self.misfit_reg_setup(beta1=0.5, beta2=0.5)


            # calculate sqrt(-G * D)
            graddir = math.sqrt(- MG.inner(m_delta) )

            sp = ""
            gradnorm_list += [gradnorm]
            # print opt log and write opt log
            if save_opt_log:
                with open(csv_path, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([iter, self.Hess.cgiter, cost_new, misfit_new, reg_new, graddir, gradnorm, alpha, tolcg])
            print(f"{iter:2d} {sp:2s} {self.Hess.cgiter:3d} {sp:1s} {cost_new:8.5e} {sp:1s} {misfit_new:8.5e} {sp:1s} {reg_new:8.5e} {sp:1s} {graddir:8.5e} {sp:1s} {gradnorm:8.5e} {sp:1s} {gradnorm_rel:8.5e} {sp:1s} {alpha:1.2e} {sp:1s} {tolcg:5.3e}")

            if alpha < 1e-3:
                alpha_iter += 1
            if alpha_iter > 10:
                print('Alpha too small: ')
                break

            # check for convergence
            if gradnorm < tol and iter > 1:
                converged = True
                print( "Newton's method converged in ",iter,"  iterations")
                print( "Total number of CG iterations: ", total_cg_iter)

            iter += 1
        if not converged:
            print( "Newton's method did not converge in ", maxiter, " iterations")
        
        self.save_dict = {'gamma_val': self.gamma,
                    'misfit': misfit_old,
                    'reg': reg_old,
                    'u_sol1': u1.compute_vertex_values().tolist(),
                    'u_d1': self.ud1.compute_vertex_values().tolist(),
                    'u_sol2': u2.compute_vertex_values().tolist(),
                    'u_d2': self.ud2.compute_vertex_values().tolist(),
                    'm_sol': m.compute_vertex_values().tolist(),
                    'forcing': self.f.compute_vertex_values(self.mesh).tolist(),
                    'gradnorm': gradnorm,
                    'gradnorm_list': gradnorm_list,
                    'u_oce1': self.u_oce1,
                    'u_oce2': self.u_oce2,
                    }

        if hasattr(self, 'mtrue'):
            self.save_dict['m_true'] = self.mtrue.compute_vertex_values().tolist()
        
        return m, u1, u2

    def wf_matrices_setup(self,m, u1, u2, p1, p2):
        # weak form for setting up matrices
        Wum_equ1 = self.beta1 * ufl.inner(ufl.exp(m) * self.m_trial * ufl.grad(self.p_test), ufl.grad(p1)) * ufl.dx
        C_equ1   = self.beta1 * ufl.inner(ufl.exp(m) * self.m_trial * ufl.grad(u1), ufl.grad(self.u_test)) * ufl.dx
        Wmm_equ1 = self.beta1 * ufl.inner(ufl.exp(m) * self.m_trial * self.m_test *  ufl.grad(u1),  ufl.grad(p1)) * ufl.dx

        Wum_equ2 = self.beta2 * ufl.inner(ufl.exp(m) * self.m_trial * ufl.grad(self.p_test), ufl.grad(p2)) * ufl.dx
        C_equ2   = self.beta2 * ufl.inner(ufl.exp(m) * self.m_trial * ufl.grad(u2), ufl.grad(self.u_test)) * ufl.dx
        Wmm_equ2 = self.beta2 * ufl.inner(ufl.exp(m) * self.m_trial * self.m_test *  ufl.grad(u2),  ufl.grad(p2)) * ufl.dx

        M_equ   = ufl.inner(self.m_trial, self.m_test) * ufl.dx

        return Wum_equ1, C_equ1, Wmm_equ1, Wum_equ2, C_equ2, Wmm_equ2, M_equ

    def cost(self, u1, u2, m):
        diff1 = u1.vector() - self.ud1.vector()
        diff2 = u2.vector() - self.ud2.vector()

        beta1 = self.beta1.values().item()
        beta2 = self.beta2.values().item()
        reg = m.vector().inner(self.R*m.vector())
        misfit = beta1 * diff1.inner(self.W * diff1) + beta2 * diff2.inner(self.W * diff2)
        return [0.5 * reg + 0.5 * misfit, misfit, reg]

    def eigenvalue_request(self, m, p=20):
        k = self.nx
        P = self.R + self.gamma * self.M * 1e-3
        Psolver = dl.PETScKrylovSolver("cg", amg_method())
        Psolver.set_operator(P)

        import hippylib as hp
        Omega = hp.MultiVector(m.vector(), k + p)
        hp.parRandom.normal(1., Omega)
        lmbda, evecs = hp.doublePassG(self.Hess, P, Psolver, Omega, k)
        return lmbda, evecs