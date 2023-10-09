from helm_eq import single_data_run
from helm_eq import dual_data_run
from utils.general import increment_path
import dolfin as dl
import ufl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from csv import DictWriter
from datetime import datetime

def gaussian(x, mean, std):
    return np.exp(-np.power(x - mean, 2.) / (2 * np.power(std, 2.)))

class single_data_loop_run():
    def __init__(self,
                 gamma_list, 
                 nx,
                 a,b,
                 bc_type,
                 omega,
                 oce_val,):
        self.gamma_list = gamma_list
        self.nx = nx
        self.a, self.b = a, b
        self.BC_type = bc_type
        self.omega = omega
        self.oce_val = oce_val
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d-%H%M%S")
        self.save_path = increment_path(f'./log/TEST_single_data_{bc_type}_{dt_string}', mkdir=True)
        if os.path.exists(self.save_path / 'results.csv'):
            pass
        else:
            with open(self.save_path / 'results.csv', 'w') as f:
                pass

    def gamma_loop(self):
        for gamma in self.gamma_list:
            run = single_data_run(
                self.nx,
                self.a,
                self.b,
                self.BC_type,
                self.omega,
                self.oce_val,
                normalized_mean=self.oce_val
            )
            # Set up gamma
            run.misfit_reg_setup(gamma)

            # set up state and adjoint for fwd solve
            if bc_type == 'DBC':
                a_state_str = 'ufl.inner(ufl.exp(m) * ufl.grad(self.u_trial), ufl.grad(self.u_test)) * ufl.dx+self.f * self.omega_val * self.u_trial * self.u_test * ufl.dx'
                L_state_str = 'self.f * self.omega_val * self.u_oce_val * self.u_test * ufl.dx + self.g * self.u_test * ufl.dx'
                a_adj_str = 'ufl.inner(ufl.exp(m) * ufl.grad(self.p_trial), ufl.grad(self.p_test)) * ufl.dx + self.f * self.omega_val * self.p_trial * self.p_test * ufl.dx'
                L_adj_str = '-ufl.inner(u - self.ud, self.p_test) * ufl.dx'
            elif bc_type == 'NBC':
                a_state_str = 'ufl.inner(ufl.exp(m) * ufl.grad(self.u_trial), ufl.grad(self.u_test)) * ufl.dx + self.f * self.omega_val * self.u_trial * self.u_test * ufl.dx'
                L_state_str = 'self.f * self.omega_val * self.u_oce_val * self.u_test * ufl.dx + self.g * self.u_test * ufl.dx  + self.j1 * self.u_test * self.ds(1) - self.j0 * self.u_test * self.ds(0)'
                a_adj_str = 'ufl.inner(ufl.exp(m) * ufl.grad(self.p_trial), ufl.grad(self.p_test)) * ufl.dx + self.f * self.omega_val * self.p_trial * self.p_test * ufl.dx'
                L_adj_str = '-ufl.inner(u - self.ud, self.p_test) * ufl.dx'
            run.state_adj_str_setup(a_state_str, L_state_str, a_adj_str, L_adj_str)
            

            from numpy import cos, sin, exp, pi
            x = np.linspace(self.a,self.b,self.nx+1)

            y = sin(2* pi * x)
            y_lin = -2 * pi * x +  pi
            y1 = y.copy()
            y2 = y.copy()
            if bc_type == 'DBC':
                y1[x>=0.5] = y_lin[x>=0.5]
            elif bc_type == 'NBC':
                y1[x<=0.5] = y_lin[x<=0.5]
            else:
                raise ValueError(f'Unknown BC type: {bc_type}')
            dy1 = np.gradient(y1,x,edge_order=2)

            # set up forcing term
            from utils.general import normalize_function
            c_1 = 10
            peak_loc = 0.3
            gauss_var = 1/10
            f_str =  f'std::exp(-pow(x[0] - {peak_loc}, 2) / {gauss_var}) * {c_1}'
            f = normalize_function(f_str, run.Vm, ufl.dx)
            forcing_array = f.compute_vertex_values()
            run.f_setup(f.compute_vertex_values())

            # The true and inverted parameter
            def gaussian(x, mean, std):
                return np.exp(-np.power(x - mean, 2.) / (2 * np.power(std, 2.)))

            mtrue_array = gaussian(x, 0.2, 0.1) + gaussian(x, 0.8, 0.1)
            m_true = dl.Function(run.Vm)
            m_true.vector().set_local(mtrue_array[::-1])
            run.mtrue_setup(m_true)

            # set up boundary conditions
            from numpy import cos, pi, sin, exp
            if self.BC_type == 'DBC':
                uL = y1[0]
                uR = y1[-1]
            elif self.BC_type == 'NBC':
                uL = dy1[0] * np.exp(mtrue_array[0])
                uR = dy1[-1] * np.exp(mtrue_array[-1])
            else:
                raise ValueError(f'Unknown BC type: {self.BC_type}')
            run.BC_setup(uL, uR)

            g_array = - np.gradient( np.exp(mtrue_array) * np.gradient(y1, x, edge_order=2) , x, edge_order=2) \
            + forcing_array * omega * (y1 - oce_val)

            g = dl.Function(run.Vu)
            g.vector().set_local(g_array[::-1])
            run.extra_f_setup(g = g)

            np.random.seed(0)
            self.noise_level = 0.03
            ud, goal_A, goal_b = run.fwd_solve(run.mtrue)
            utrue_array = ud.compute_vertex_values()
            from utils.general import apply_noise
            apply_noise(self.noise_level, ud, goal_A)
            run.data_setup(ud_array = ud.compute_vertex_values(),normalize = False)

            #initial guess m
            m = dl.interpolate(dl.Expression("std::log(2)", degree=1),run.Vm)
            # m.assign(mtrue)

            u,_,_ = run.fwd_solve(m)

            ## =========== run optimization loop ==========
            tol = 1e-11
            c = 1e-4
            maxiter=100
            m, u = run.opt_loop(m, tol, c, maxiter)


            if hasattr(self, 'noise_level'):
                run.save_dict['noise_level'] = self.noise_level


            # save results
            with open(self.save_path / 'results.csv', 'a') as f:
                writer_object = DictWriter(f, run.save_dict.keys())
                if f.tell() == 0:
                    writer_object.writeheader()
                writer_object.writerow(run.save_dict)

class double_data_loop_run():
    def __init__(self,
                 gamma_list, 
                 nx,
                 a,b,
                 omega,
                 oce_val,):
        self.gamma_list = gamma_list
        self.nx = nx
        self.a, self.b = a, b
        self.omega = omega
        self.oce_val = oce_val
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d-%H%M%S")
        self.save_path = increment_path(f'./log/double_data_{dt_string}', mkdir=True)
        if os.path.exists(self.save_path / 'results.csv'):
            pass
        else:
            with open(self.save_path / 'results.csv', 'w') as f:
                pass
    
    def gamma_loop(self):
        for gamma in self.gamma_list:
            # setup misfit terms weights
            beta1 = 1.0
            beta2 = 1.0

            run = dual_data_run(
                self.nx,
                self.a,
                self.b,
                gamma,
                self.omega,
                self.omega,
                self.oce_val,
                self.oce_val,
                normalized_mean1=self.oce_val,
                normalized_mean2=self.oce_val,
                beta1=beta1,
                beta2=beta2
            )

            ## ======== set up the state and adjoint str ========
            a1_state_str = 'ufl.inner(ufl.exp(m) * ufl.grad(self.u_trial), ufl.grad(self.u_test)) * ufl.dx + self.f * self.omega_val1 * self.u_trial * self.u_test * ufl.dx'
            L1_state_str = 'self.f * self.omega_val1 * self.u_oce_val1 * self.u_test * ufl.dx + self.g1 * self.u_test * ufl.dx'
            a1_adj_str = 'ufl.inner(ufl.exp(m) * ufl.grad(self.p_trial), ufl.grad(self.p_test)) * ufl.dx + self.f * self.omega_val1 * self.p_trial * self.p_test * ufl.dx'
            L1_adj_str = '-ufl.inner(u1 - self.ud1, self.p_test) * ufl.dx'

            a2_state_str = 'ufl.inner(ufl.exp(m) * ufl.grad(self.u_trial), ufl.grad(self.u_test)) * ufl.dx + self.f * self.omega_val2 * self.u_trial * self.u_test * ufl.dx'
            L2_state_str = 'self.f * self.omega_val2 * self.u_oce_val2 * self.u_test * ufl.dx + self.g2 * self.u_test * ufl.dx  + self.j1 * self.u_test * self.ds(1) - self.j0 * self.u_test * self.ds(0)'
            a2_adj_str = 'ufl.inner(ufl.exp(m) * ufl.grad(self.p_trial), ufl.grad(self.p_test)) * ufl.dx + self.f * self.omega_val2 * self.p_trial * self.p_test * ufl.dx'
            L2_adj_str = '-ufl.inner(u2 - self.ud2, self.p_test) * ufl.dx'

            run.state_adj_str_setup(a1_state_str, L1_state_str, a1_adj_str, L1_adj_str, a2_state_str, L2_state_str, a2_adj_str, L2_adj_str)

            ## ======== set up forcing term f(x) ========
            from utils.general import normalize_function
            c_1 = 10
            peak_loc = 0.3
            gauss_var = 1/10
            f_str =  f'std::exp(-pow(x[0] - {peak_loc}, 2) / {gauss_var}) * {c_1}'
            f = normalize_function(f_str, run.Vm, ufl.dx)
            run.f_setup(f.compute_vertex_values())

            u1L_D = 0.0#cos(0.0)*3*pi
            u1R_D = 1.0 #cos(3*pi)*3*pi
            u2L_N = 1.0
            u2R_N = 0.1
            run.BC_setup(u1L_D, u1R_D, u2L_N, u2R_N)

            # ====== set up true diffusion
            x = np.linspace(self.a,self.b,self.nx+1)
            mtrue_array = gaussian(x, 0.2, 0.1) + gaussian(x, 0.8, 0.1)
            m_true = dl.Function(run.Vm)
            m_true.vector().set_local(mtrue_array[::-1])
            run.mtrue_setup(m_true)

            # set up data with noise
            noise_level = 0.03
            ud1, ud2, goal_A1, goal_b1, goal_A2, goal_b2 = run.fwd_solve(m)
            utrue1_array = ud1.compute_vertex_values()
            utrue2_array = ud2.compute_vertex_values()
            from utils.general import apply_noise
            np.random.seed(0)
            apply_noise(noise_level, ud1, goal_A1)
            np.random.seed(0)
            apply_noise(noise_level, ud2, goal_A2)
            run.data_setup(ud1.compute_vertex_values(), ud2.compute_vertex_values(),normalize=False)

            #initial guess m
            m = dl.interpolate(dl.Expression("std::log(2)", degree=1),run.Vm)

            u1,u2,_,_,_,_ = run.fwd_solve(m)

            ## ================ run optimization loop ===============
            tol = 1e-11
            c = 1e-4
            maxiter=100
            m, u1, u2 = run.opt_loop(m, tol, c, maxiter)

            if hasattr(self, 'noise_level'):
                run.save_dict['noise_level'] = self.noise_level

            # save results
            with open(self.save_path / 'results.csv', 'a') as f:
                writer_object = DictWriter(f, run.save_dict.keys())
                if f.tell() == 0:
                    writer_object.writeheader()
                writer_object.writerow(run.save_dict)
            
            

if __name__ == '__main__':


    gamma_list = np.logspace(-6, -3, 50)
    nx = 32
    a = 0.0
    b = 1.0
    bc_type = 'DBC'
    omega = 10
    oce_val = 0.0

    exp_run = single_data_loop_run(gamma_list, nx, a, b, bc_type, omega, oce_val)
    exp_run.gamma_loop()

    # exp_run = double_data_loop_run(gamma_list, nx, a, b, omega, oce_val)
    # exp_run.gamma_loop()