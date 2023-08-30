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
            # a_state = ufl.inner(ufl.exp(m) * ufl.grad(self.u_trial), ufl.grad(self.u_test)) * ufl.dx\
            #         + self.f * self.omega_val * self.u_trial * self.u_test * ufl.dx
            # L_state = self.f * self.omega_val * self.u_oce_val * self.u_test * ufl.dx  + self.j1 * self.u_test * self.ds(1) - self.j0 * self.u_test * self.ds(0)
            a_state = eval(self.a_state_str)
            L_state = eval(self.L_state_str)

            state_A, state_b = dl.assemble_system (a_state, L_state)
            dl.solve (state_A, u.vector(), state_b)
        
        if self.bc_type == 'DBC':
            # a_state = ufl.inner(ufl.exp(m) * ufl.grad(self.u_trial), ufl.grad(self.u_test)) * ufl.dx\
            #          +self.f * self.omega_val * self.u_trial * self.u_test * ufl.dx
            # L_state = self.f * self.omega_val * self.u_oce_val * self.u_test * ufl.dx
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
        # a_adj = ufl.inner(ufl.exp(m) * ufl.grad(self.p_trial), ufl.grad(self.p_test)) * ufl.dx \
        #         + self.f * self.omega_val * self.p_trial * self.p_test * ufl.dx
        # L_adj = -ufl.inner(u - self.ud, self.p_test) * ufl.dx
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
                M = dl.assemble(M_equ) 

            # assemble W_ua and R
            Wum = dl.assemble (Wum_equ)
            Wmm = dl.assemble (Wmm_equ)

            # evaluate the  gradient
            CT_p = dl.Vector()
            C.init_vector(CT_p,1)
            C.transpmult(p.vector(), CT_p)
            # print(np.linalg.norm(CT_p.get_local()))
            MG = CT_p + self.R * m.vector()
            dl.solve(M, g, MG)

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
            if self.bc_type == 'NBC':
                from utils.hessian_operator import HessianOperatorNBC as HessianOperator
                Hess_Apply = HessianOperator(self.R, Wmm, C, state_A, adjoint_A, self.W, Wum, gauss_newton_approx=(iter<6) )
            P = self.R + self.gamma * M
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

            # print( "Double Pass Algorithm. Requested eigenvectors: {0}; Oversampling {1}.".format(k,p) )
            # Omega = MultiVector(m.vector(), k+p)
            # parRandom.normal(1., Omega)
            # lmbda, evecs = doublePassG(Hess_Apply, P, Psolver, Omega, k)
            # plt.plot(range(0,k), lmbda, 'b*', range(0,k+1), np.ones(k+1), '-r')
            # plt.yscale('log')
            # plt.xlabel('number')
            # plt.ylabel('eigenvalue')
            # plt.show()
            # print('Min Eigenvalue: ', min(lmbda))
            # self.eigval_list.append(min(lmbda))

            # solve the Newton system H a_delta = - MG
            # print(np.linalg.norm(MG.get_local()))
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
    
class dual_data_run():
    def __init__(self, 
                nx, 
                a, 
                b,
                gamma,
                omega1,
                omega2,
                oce_val1,
                oce_val2,
                normalized_mean1,
                normalized_mean2,
                beta1,
                beta2,
                func='Lagrange',):
        self.a, self. b, self.nx = a, b, nx
        self.mesh = dl.IntervalMesh(nx, a, b)
        self.Vm = self.Vu = dl.FunctionSpace(self.mesh, func, 1)

        self.u_trial, self.p_trial, self.m_trial = dl.TrialFunction(self.Vu), dl.TrialFunction(self.Vu), dl.TrialFunction(self.Vm)
        self.u_test, self.p_test, self.m_test = dl.TestFunction(self.Vu), dl.TestFunction(self.Vu), dl.TestFunction(self.Vm)

        self.gamma = gamma
        self.omega_val1 = dl.Constant(omega1)
        self.omega_val2 = dl.Constant(omega2)
        self.omega1 = omega1
        self.omega2 = omega2
        self.u_oce_val1 = dl.Constant(f'{oce_val1 - normalized_mean1}')
        self.u_oce = oce_val1
        self.u_oce_val2 = dl.Constant(f'{oce_val2 - normalized_mean2}')
        self.u_oce = oce_val2
        self.normalized_mean1 = normalized_mean1
        self.normalized_mean2 = normalized_mean2

        # constant in from of the misfit and weakform
        self.beta1 = dl.Constant(beta1)
        self.beta2 = dl.Constant(beta2)

        # weak for for setting up the misfit and regularization compoment of the cost
        W_equ   = ufl.inner(self.u_trial, self.u_test) * ufl.dx
        R_equ   = gamma * ufl.inner(ufl.grad(self.m_trial), ufl.grad(self.m_test)) * ufl.dx

        self.W = dl.assemble(W_equ)
        self.R = dl.assemble(R_equ)

    def fwd_solve(self,m):
        u1 = dl.Function(self.Vu)
        u2 = dl.Function(self.Vu)

        ## ======= Dirichlet BC ========
        a1_state = ufl.inner(ufl.exp(m) * ufl.grad(self.u_trial), ufl.grad(self.u_test)) * ufl.dx\
                + self.f * self.omega_val1 * self.u_trial * self.u_test * ufl.dx
        L1_state = self.f * self.omega_val1 * self.u_oce_val1 * self.u_test * ufl.dx

        state_A1, state_b1 = dl.assemble_system (a1_state, L1_state, self.bc_state)
        dl.solve (state_A1, u1.vector(), state_b1)

        ## ======= Neumann BC ========
        a2_state = ufl.inner(ufl.exp(m) * ufl.grad(self.u_trial), ufl.grad(self.u_test)) * ufl.dx\
                + self.f * self.omega_val2 * self.u_trial * self.u_test * ufl.dx
        L2_state = self.f * self.omega_val2 * self.u_oce_val2 * self.u_test * ufl.dx  + self.j1 * self.u_test * self.ds(1) - self.j0 * self.u_test * self.ds(0)

        state_A2, state_b2 = dl.assemble_system (a2_state, L2_state)
        dl.solve (state_A2, u2.vector(), state_b2)
        
        
        return u1, u2, state_A1, state_b1, state_A2, state_b2
    
    def adj_solve(self,m, u1, u2):
        p1 = dl.Function(self.Vu)
        p2 = dl.Function(self.Vu)
        
        ## ====== Dirichelt ========
        a_adj1 = ufl.inner(ufl.exp(m) * ufl.grad(self.p_trial), ufl.grad(self.p_test)) * ufl.dx \
                + self.f * self.omega_val1 * self.p_trial * self.p_test * ufl.dx
        L1_adj = -ufl.inner(u1 - self.ud1, self.p_test) * ufl.dx

        adjoint_A1, adjoint_RHS1 = dl.assemble_system(a_adj1, L1_adj, self.bc_adj)
        dl.solve(adjoint_A1, p1.vector(), adjoint_RHS1)
        ## ====== Neumann   ========
        a_adj2 = ufl.inner(ufl.exp(m) * ufl.grad(self.p_trial), ufl.grad(self.p_test)) * ufl.dx \
                + self.f * self.omega_val2 * self.p_trial * self.p_test * ufl.dx
        L2_adj = -ufl.inner(u2 - self.ud2, self.p_test) * ufl.dx

        adjoint_A2, adjoint_RHS2 = dl.assemble_system(a_adj2, L2_adj)
        dl.solve(adjoint_A2, p2.vector(), adjoint_RHS2)

        return p1, p2, adjoint_A1, adjoint_RHS1, adjoint_A2, adjoint_RHS2

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

    def opt_loop(self, 
                m, #'diffusion coef',
                tol, #'tolerance of grad',
                c, #'constant of armijo linesearch',
                maxiter, #'maximum newton"s step',
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
                M = dl.assemble(M_equ)
            
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
            dl.solve(M, g, MG)

            # calculate the norm of the gradient
            grad2 = g.inner(MG)
            gradnorm = math.sqrt(grad2)

            # set the CG tolerance (use Eisenstat–Walker termination criterion)
            if iter == 1:
                gradnorm_ini = gradnorm
            tolcg = min(0.5, math.sqrt(gradnorm/(gradnorm_ini+1e-15)))

            # define the Hessian apply operator (with preconditioner)
            from utils.hessian_operator import HessianOperator_comb as HessianOperator
            Hess_Apply = HessianOperator(self.R, self.W, Wmm1, C1, state_A1, adjoint_A1, Wum1, 
                                         Wmm2, C2, state_A2, adjoint_A2, Wum2, self.bc_adj, gauss_newton_approx=(iter<6))
            self.Hess_Apply = Hess_Apply
        
            # P = self.R + self.gamma * M
            P = self.R + self.gamma * M
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
            # print( "Double Pass Algorithm. Requested eigenvectors: {0}; Oversampling {1}.".format(k,p) )

            # Omega = MultiVector(m.vector(), k+p)
            # parRandom.normal(1., Omega)
            # lmbda, evecs = doublePassG(Hess_Apply, P, Psolver, Omega, k)
            # self.lmbda_val = lmbda

            # plt.plot(range(0,k), lmbda, 'b*', range(0,k+1), np.ones(k+1), '-r')
            # plt.yscale('log')
            # plt.xlabel('number')
            # plt.ylabel('eigenvalue')
            # plt.show()
            # print(min(lmbda))

            # solve the Newton system H a_delta = - MG
            # print(np.linalg.norm(MG.get_local()))
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
            
            # calculate sqrt(-G * D)
            graddir = math.sqrt(- MG.inner(m_delta) )

            sp = ""
            gradnorm_list += [gradnorm]
            # print ("Nit   CGit   cost          misfit        reg           sqrt(-G*D)    ||grad||       alpha  tolcg")
            print( "%2d %2s %2d %3s %8.5e %1s %8.5e %1s %8.5e %1s %8.5e %1s %8.5e %1s %1.2e %1s %5.3e" % \
                (iter, sp, Hess_Apply.cgiter, sp, cost_new, sp, misfit_new, sp, reg_new, sp, \
                graddir, sp, gradnorm, sp, alpha, sp, tolcg) )

            if alpha < 1e-2:
                alpha_iter += 1
            if alpha_iter > 5:
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
                    }
        
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

        reg = m.vector().inner(self.R*m.vector())
        misfit = self.beta1.values() * diff1.inner(self.W * diff1) + self.beta2.values() * diff2.inner(self.W * diff2)
        return [0.5 * reg + 0.5 * misfit, misfit, reg]