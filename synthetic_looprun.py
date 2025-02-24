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



class single_data_loop_run():
    def __init__(self,
                 gamma_list, 
                 nx,
                 a,b,
                 bc_type,
                 omega,
                 oce_val,
                 file_name = 'single_data'):
        self.gamma_list = gamma_list
        self.nx = nx
        self.a, self.b = a, b
        self.BC_type = bc_type
        self.omega = omega
        self.oce_val = oce_val
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d-%H%M%S")
        self.save_path = increment_path(f'./log/{file_name}_{bc_type}_{dt_string}', mkdir=True)
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
            if self.BC_type == 'DBC':
                a_state_str = 'ufl.inner(ufl.exp(m) * ufl.grad(self.u_trial), ufl.grad(self.u_test)) * ufl.dx+self.f * self.omega_val * self.u_trial * self.u_test * ufl.dx'
                L_state_str = 'self.f * self.omega_val * self.u_oce_val * self.u_test * ufl.dx'
                a_adj_str = 'ufl.inner(ufl.exp(m) * ufl.grad(self.p_trial), ufl.grad(self.p_test)) * ufl.dx + self.f * self.omega_val * self.p_trial * self.p_test * ufl.dx'
                L_adj_str = '-ufl.inner(u - self.ud, self.p_test) * ufl.dx'
                run.state_adj_str_setup(a_state_str, L_state_str, a_adj_str, L_adj_str)
            elif self.BC_type == 'NBC':
                a_state_str = 'ufl.inner(ufl.exp(m) * ufl.grad(self.u_trial), ufl.grad(self.u_test)) * ufl.dx + self.f * self.omega_val * self.u_trial * self.u_test * ufl.dx'
                L_state_str = 'self.f * self.omega_val * self.u_oce_val * self.u_test * ufl.dx  + self.j1 * self.u_test * self.ds(1) - self.j0 * self.u_test * self.ds(0)'
                a_adj_str = 'ufl.inner(ufl.exp(m) * ufl.grad(self.p_trial), ufl.grad(self.p_test)) * ufl.dx + self.f * self.omega_val * self.p_trial * self.p_test * ufl.dx'
                L_adj_str = '-ufl.inner(u - self.ud, self.p_test) * ufl.dx'
                run.state_adj_str_setup(a_state_str, L_state_str, a_adj_str, L_adj_str)
            
            # set up forcing term
            from utils.general import normalize_function
            c_1 = 10
            peak_loc = 0.3
            gauss_var = 1/10
            f_str =  f'std::exp(-pow(x[0] - {peak_loc}, 2) / {gauss_var}) * {c_1}'
            f = normalize_function(f_str, run.Vm, ufl.dx)
            run.f_setup(f.compute_vertex_values())

            # set up boundary conditions
            from numpy import cos, pi, sin, exp
            if self.BC_type == 'DBC':
                u0L = 1.0
                u0R = 0.0
            elif self.BC_type == 'NBC':
                u0L = 1.0
                u0R = 0.1
            else:
                raise ValueError(f'Unknown BC type: {self.BC_type}')
            run.BC_setup(u0L, u0R)

            # The true and inverted parameter
            peak_loc = 0.4
            gauss_var = 30
            mtrue_expression_str = f'std::log(1.+ 10.*std::exp(-pow(x[0] - {peak_loc}, 2) * {gauss_var}))'
            # mtrue_expression_str = 'std::x[0]*sin(x[0])'
            mtrue_expression = dl.Expression(mtrue_expression_str, degree=5)
            mtrue = m =  dl.interpolate(mtrue_expression,run.Vm)
            run.mtrue_setup(mtrue)

            noise_level = 0.02
            ud, goal_A, goal_b = run.fwd_solve(m)
            utrue_array = ud.compute_vertex_values()
            from utils.general import apply_noise
            self.utrue_array = ud.compute_vertex_values().copy()
            np.random.seed(0)
            apply_noise(noise_level, ud, goal_A)
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
            if hasattr(self, 'utrue_array'):
                run.save_dict['utrue'] = self.utrue_array.tolist()

            # save results
            with open(self.save_path / 'results.csv', 'a') as f:
                writer_object = DictWriter(f, run.save_dict.keys())
                if f.tell() == 0:
                    writer_object.writeheader()
                writer_object.writerow(run.save_dict)
    
    def nx_loop(self, nx_list, gamma):
        for nx in nx_list:
            run = single_data_run(
                nx,
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
            if self.BC_type == 'DBC':
                a_state_str = 'ufl.inner(ufl.exp(m) * ufl.grad(self.u_trial), ufl.grad(self.u_test)) * ufl.dx+self.f * self.omega_val * self.u_trial * self.u_test * ufl.dx'
                L_state_str = 'self.f * self.omega_val * self.u_oce_val * self.u_test * ufl.dx'
                a_adj_str = 'ufl.inner(ufl.exp(m) * ufl.grad(self.p_trial), ufl.grad(self.p_test)) * ufl.dx + self.f * self.omega_val * self.p_trial * self.p_test * ufl.dx'
                L_adj_str = '-ufl.inner(u - self.ud, self.p_test) * ufl.dx'
                run.state_adj_str_setup(a_state_str, L_state_str, a_adj_str, L_adj_str)
            elif self.BC_type == 'NBC':
                a_state_str = 'ufl.inner(ufl.exp(m) * ufl.grad(self.u_trial), ufl.grad(self.u_test)) * ufl.dx + self.f * self.omega_val * self.u_trial * self.u_test * ufl.dx'
                L_state_str = 'self.f * self.omega_val * self.u_oce_val * self.u_test * ufl.dx  + self.j1 * self.u_test * self.ds(1) - self.j0 * self.u_test * self.ds(0)'
                a_adj_str = 'ufl.inner(ufl.exp(m) * ufl.grad(self.p_trial), ufl.grad(self.p_test)) * ufl.dx + self.f * self.omega_val * self.p_trial * self.p_test * ufl.dx'
                L_adj_str = '-ufl.inner(u - self.ud, self.p_test) * ufl.dx'
                run.state_adj_str_setup(a_state_str, L_state_str, a_adj_str, L_adj_str)
            
            # set up forcing term
            from utils.general import normalize_function
            c_1 = 10
            peak_loc = 0.3
            gauss_var = 1/10
            f_str =  f'std::exp(-pow(x[0] - {peak_loc}, 2) / {gauss_var}) * {c_1}'
            f = normalize_function(f_str, run.Vm, ufl.dx)
            run.f_setup(f.compute_vertex_values())

            # set up boundary conditions
            from numpy import cos, pi, sin, exp
            if self.BC_type == 'DBC':
                u0L = 1.0
                u0R = 0.0
            elif self.BC_type == 'NBC':
                u0L = 1.0
                u0R = 0.1
            else:
                raise ValueError(f'Unknown BC type: {self.BC_type}')
            run.BC_setup(u0L, u0R)

            # The true and inverted parameter
            peak_loc = 0.4
            gauss_var = 30
            mtrue_expression_str = f'std::log(1.+ 10.*std::exp(-pow(x[0] - {peak_loc}, 2) * {gauss_var}))'
            # mtrue_expression_str = 'std::x[0]*sin(x[0])'
            mtrue_expression = dl.Expression(mtrue_expression_str, degree=1)
            mtrue = m =  dl.interpolate(mtrue_expression,run.Vm)
            run.mtrue_setup(mtrue)

            noise_level = 0.00
            ud, goal_A, goal_b = run.fwd_solve(m)
            utrue_array = ud.compute_vertex_values()
            from utils.general import apply_noise
            self.utrue_array = ud.compute_vertex_values().copy()
            np.random.seed(0)
            apply_noise(noise_level, ud, goal_A)
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
            if hasattr(self, 'utrue_array'):
                run.save_dict['utrue'] = self.utrue_array.tolist()
            run.save_dict['nx'] = nx

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

            # The true and inverted parameter
            peak_loc = 0.4
            gauss_var = 30
            mtrue_expression_str = f'std::log(1.+ 10.*std::exp(-pow(x[0] - {peak_loc}, 2) * {gauss_var}))'
            # mtrue_expression_str = 'std::x[0]*sin(x[0])'
            mtrue_expression = dl.Expression(mtrue_expression_str, degree=5)
            mtrue = m =  dl.interpolate(mtrue_expression,run.Vm)
            run.mtrue_setup(mtrue)

            # set up data with noise
            noise_level = 0.01
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

            # save results
            with open(self.save_path / 'results.csv', 'a') as f:
                writer_object = DictWriter(f, run.save_dict.keys())
                if f.tell() == 0:
                    writer_object.writeheader()
                writer_object.writerow(run.save_dict)
            
            

if __name__ == '__main__':


    gamma_list = np.logspace(-7, -4, 50)
    nx = 32
    a = 0.0
    b = 1.0
    bc_type = 'DBC'
    omega = 10
    oce_val = 0.0

    exp_run = single_data_loop_run(gamma_list, nx, a, b, bc_type, omega, oce_val)
    # exp_run.gamma_loop()
    exp_run.nx_loop([8, 16, 32,64], 9.103e-6)

    # exp_run = double_data_loop_run(gamma_list, nx, a, b, omega, oce_val)
    # exp_run.gamma_loop()