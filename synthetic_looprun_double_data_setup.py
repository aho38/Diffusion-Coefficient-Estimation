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


def loop_run(gamma, nx, a, b, omega, oce_val, beta1, beta2, noise_level, save_path,opt_csv_name,seed=0):
    ## ======== initialize runs and setup ========
    run = dual_data_run(nx,a,b,gamma, gamma, omega, omega, oce_val, oce_val, normalized_mean1=0.0,normalized_mean2=0.0,save_path=save_path)
    run.misfit_reg_setup(beta1, beta2, gamma=gamma)# This gamma will override the gamma in the class

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

    ## ======== get dual $u_{1,2}$ solution ========
    from numpy import cos, sin, exp, pi
    x = np.linspace(a,b,nx+1)

    y = sin(2* pi * x)
    y_lin = -2 * pi * x +  pi
    y1 = y.copy()
    y2 = y.copy()

    y1[x>=0.5] = y_lin[x>=0.5]
    y2[x<=0.5] = y_lin[x<=0.5]


    def normalize_u(u_array):
        mean = np.mean(u_array)
        std = np.std(u_array)
        return (u_array - mean) / std, mean, std

    y1, y1_mean, y1_std = normalize_u(y1)
    y2, y2_mean, y2_std = normalize_u(y2)

    # these are for Neumann boundary condition
    dy1 = np.gradient(y1,x,edge_order=2)
    dy2 = np.gradient(y2,x,edge_order=2)

    ## ======== get true diffusion coefficient ========
    def gaussian(x, mean, std):
        return np.exp(-np.power(x - mean, 2.) / (2 * np.power(std, 2.)))

    mtrue_array = gaussian(x, 0.2, 0.1) + gaussian(x, 0.8, 0.1)
    m_true = dl.Function(run.Vm)
    m_true.vector().set_local(mtrue_array[::-1])
    run.mtrue_setup(m_true)

    ## ======== set up distribution function i called f ===========
    from utils.general import normalize_function
    c_1 = 10
    peak_loc = 0.3
    gauss_var = 1/10
    f_str =  f'std::exp(-pow(x[0] - {peak_loc}, 2) / {gauss_var}) * {c_1}'
    f = normalize_function(f_str, run.Vm, ufl.dx)
    forcing_array = f.compute_vertex_values()
    run.f_setup(f.compute_vertex_values())

    ## ========= set up boundary condition ===========
    u1L_D = y1[0]
    u1R_D = y1[-1]
    u2L_N = dy2[0] * np.exp(mtrue_array[0])
    u2R_N = dy2[-1] * np.exp(mtrue_array[-1])
    print(f'Dirichlet_L: {u1L_D}, Dirichlet_R: {u1R_D},')
    print(f'NBC_L: {u2L_N}, NBC_R: {u2R_N}')
    run.BC_setup(u1L_D, u1R_D, u2L_N, u2R_N)

    ## ========= set up solar irradiance forcing ===========
    g1_array = - np.gradient( np.exp(mtrue_array) * np.gradient(y1, x, edge_order=2) , x, edge_order=2) \
            + forcing_array * omega * (y1 - oce_val)

    g1 = dl.Function(run.Vu)
    g1.vector().set_local(g1_array[::-1])

    g2_array = - np.gradient( np.exp(mtrue_array) * np.gradient(y2, x, edge_order=2) , x, edge_order=2) \
                + forcing_array * omega * (y2 - oce_val)
    g2 = dl.Function(run.Vu)
    g2.vector().set_local(g2_array[::-1])
    run.extra_f_setup(g1 = g1, g2 = g2)

    ## ======== set up data ========
    run.data_setup(y1, y2 ,normalize=False)
    from utils.general import apply_noise
    ud1, ud2, A1, b1, A2, b2 = run.fwd_solve(m_true)
    np.random.seed(seed)
    apply_noise(noise_level, ud1, A1)
    np.random.seed(seed)
    apply_noise(noise_level, ud2, A2)
    run.data_setup(ud1.compute_vertex_values(), ud2.compute_vertex_values(), normalize=False, noise_level=noise_level)

    ## ======== Initial Guess ========
    m = dl.interpolate(dl.Expression("std::log(1)", degree=1),run.Vm)
    from utils.general import DboundaryL, DboundaryR
    bc_m = [dl.DirichletBC(run.Vm, m_true, DboundaryL), dl.DirichletBC(run.Vm, m_true, DboundaryR)]
    bc_m[0].apply(m.vector())
    bc_m[1].apply(m.vector())
    _,_,_,_,_,_ = run.fwd_solve(m)

    ## ======== optimization loop ========
    tol = 1e-11
    c = 1e-3
    maxiter=100
    m, u1, u2 = run.opt_loop(m, tol, c, maxiter, save_opt_log=True, plot_opt_step=False, plot_eigval=False, opt_csv_name=opt_csv_name)

    return run.save_dict

class single_data_loop_run():
    def __init__(self,
                 gamma_list, 
                 nx,
                 a,b,
                 bc_type,
                 omega,
                 oce_val,dir_name='TEST_single_data'):
        self.gamma_list = gamma_list
        self.nx = nx
        self.a, self.b = a, b
        self.BC_type = bc_type
        self.omega = omega
        self.oce_val = oce_val
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d-%H%M%S")
        self.save_path = increment_path(f'./log/{dir_name}_{bc_type}_{dt_string}', mkdir=True)
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

            self.noise_level = 0.03
            ud, goal_A, goal_b = run.fwd_solve(run.mtrue)
            utrue_array = ud.compute_vertex_values()
            from utils.general import apply_noise
            self.utrue_array = ud.compute_vertex_values().copy()
            np.random.seed(0)
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

            if hasattr(self, 'utrue_array'):
                run.save_dict['utrue_array'] = self.utrue_array.tolist()


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
            x = np.linspace(self.a,self.b,nx+1)

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

            self.noise_level = 0.03
            ud, goal_A, goal_b = run.fwd_solve(run.mtrue)
            utrue_array = ud.compute_vertex_values()
            from utils.general import apply_noise
            self.utrue_array = ud.compute_vertex_values().copy()
            np.random.seed(0)
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

            if hasattr(self, 'utrue_array'):
                run.save_dict['utrue_array'] = self.utrue_array.tolist()
            
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
                 oce_val,
                 dir_name,
                 ):
        self.gamma_list = gamma_list
        self.nx = nx
        self.a, self.b = a, b
        self.omega = omega
        self.oce_val = oce_val
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d-%H%M%S")
        path_ = f'./log/{dir_name}_{dt_string}'
        self.save_path = increment_path(path_, mkdir=True)
        if os.path.exists(self.save_path / 'results.csv'):
            pass
        else:
            with open(self.save_path / 'results.csv', 'w') as f:
                pass
    
    def gamma_loop(self):
        for gamma in self.gamma_list:
            # setup misfit terms weights
            beta1 = 0.5
            beta2 = 0.5

            run = dual_data_run(
                self.nx,
                self.a,
                self.b,
                gamma,
                gamma,
                self.omega,
                self.omega,
                self.oce_val,
                self.oce_val,
                normalized_mean1=self.oce_val,
                normalized_mean2=self.oce_val,
                save_path=self.save_path
            )

            run.misfit_reg_setup(beta1, beta2)

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

            ## ======== set up u true ========
            from numpy import cos, sin, exp, pi
            x = np.linspace(self.a,self.b,self.nx+1)

            y = sin(2* pi * x)
            y_lin = -2 * pi * x +  pi
            y1 = y.copy()
            y2 = y.copy()

            y1[x>=0.5] = y_lin[x>=0.5]
            y2[x<=0.5] = y_lin[x<=0.5]

            dy1 = np.gradient(y1,x,edge_order=2)
            dy2 = np.gradient(y2,x,edge_order=2)



            ## ======== set up forcing term f(x) ========
            from utils.general import normalize_function
            c_1 = 10
            peak_loc = 0.3
            gauss_var = 1/10
            f_str =  f'std::exp(-pow(x[0] - {peak_loc}, 2) / {gauss_var}) * {c_1}'
            f = normalize_function(f_str, run.Vm, ufl.dx)
            forcing_array = f.compute_vertex_values()
            run.f_setup(f.compute_vertex_values())

            # ====== set up true diffusion
            mtrue_array = gaussian(x, 0.2, 0.1) + gaussian(x, 0.8, 0.1)
            m_true = dl.Function(run.Vm)
            m_true.vector().set_local(mtrue_array[::-1])
            run.mtrue_setup(m_true)

            # ======= set up boundary conditions ========
            u1L_D = y1[0]
            u1R_D = y1[-1]
            u2L_N = dy2[0] * np.exp(mtrue_array[0])
            u2R_N = dy2[-1] * np.exp(mtrue_array[-1])
            run.BC_setup(u1L_D, u1R_D, u2L_N, u2R_N)

            ## ====== solve for final forcing with selected u f and m ==========
            g1_array = - np.gradient( np.exp(mtrue_array) * np.gradient(y1, x, edge_order=2) , x, edge_order=2) \
                        + forcing_array * omega * (y1 - oce_val)

            g1 = dl.Function(run.Vu)
            g1.vector().set_local(g1_array[::-1])

            g2_array = - np.gradient( np.exp(mtrue_array) * np.gradient(y2, x, edge_order=2) , x, edge_order=2) \
                        + forcing_array * omega * (y2 - oce_val)
            g2 = dl.Function(run.Vu)
            g2.vector().set_local(g2_array[::-1])
            run.extra_f_setup(g1 = g1, g2 = g2)

            # set up data with noise
            noise_level = 0.03
            from utils.general import apply_noise
            ud1, ud2, A1, b1, A2, b2 = run.fwd_solve(m_true)
            np.random.seed(0)
            apply_noise(noise_level, ud1, A1)
            np.random.seed(1)
            apply_noise(noise_level, ud2, A2)
            run.data_setup(ud1.compute_vertex_values(), ud2.compute_vertex_values(), normalize=False)

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
            
            if hasattr(self, 'ud1_array'):
                run.save_dict['ud1_array'] = self.ud1_array.tolist()
            if hasattr(self, 'ud2_array'):
                run.save_dict['ud2_array'] = self.ud2_array.tolist()

            # save results
            with open(self.save_path / 'results.csv', 'a') as f:
                writer_object = DictWriter(f, run.save_dict.keys())
                if f.tell() == 0:
                    writer_object.writeheader()
                writer_object.writerow(run.save_dict)
            

def main(seed,noise_level):   

    failed_iter = []
    nx = 64
    # nx_list = [4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,160,192,224,256,288,320,352]
    # nx_list = np.linspace(4,164,41,dtype=int)
    # nx_list = [4,8,16,32,64,128,256,512]
    a = 0.0
    b = 1.0
    omega = 1.0
    oce_val = 0.0
    # noise_level = 0.005
    # gamma = 1e-15
    gamma_list = np.logspace(-12, -2, 101)
    # for j in range(10):
    #     seed = j

    beta1 = 0.5
    beta2 = 0.5

    ## set up save path
    now = datetime.now()
    noise_level_str = str(int(noise_level*100)).zfill(3) if noise_level != 0.005 else '000_5'
     #.zfill fill 0 to 3 digits 
    seed_str = str(seed).zfill(3)
    if beta1 == 0.5 and beta2 == 0.5:
        # dir_name = 'nx_loop_double_data'
        # dir_name = f'gamma_loop_double_data_noise_0dot5_seed_{seed}'
        dir_name = f'double_data_nx_64_noise_{noise_level_str}/gamma_loop_double_data_nx_{nx}_noise_{noise_level_str}_seed_{seed_str}'
    elif beta1 == 1.0 and beta2 == 0.0:
        # dir_name = 'nx_loop_DBC_single_data'
        # dir_name = f'gamma_loop_DBC_single_data_noise_0dot1_seed_{seed}'
        dir_name = f'DBC_nx_64_noise_{noise_level_str}/gamma_loop_DBC_single_data_nx_{nx}_noise_{noise_level_str}_seed_{seed_str}'
    elif beta1 == 0.0 and beta2 == 1.0:
        # dir_name = 'nx_loop_NBC_single_data'
        dir_name = f'NBC_nx_64_noise_{noise_level_str}/gamma_loop_NBC_single_data_nx_{nx}_noise_{noise_level_str}_seed_{seed_str}'
    else:
        raise ValueError('beta1 and beta2 must be 0.5 or 1.0 or 0.0')
    dt_string = now.strftime("%Y%m%d-%H%M%S")
    path_ = f'./log/{dir_name}_{dt_string}'
    save_path = increment_path(path_, mkdir=True)
    if os.path.exists(save_path / f'results.csv'):
        pass
    else:
        with open(save_path / f'results.csv', 'w') as f:
            pass
    ## ===================
    

    # for i, nx in enumerate(nx_list):
    #     print(f'iter: {i}, nx: {nx}')
    for i, gamma in enumerate(gamma_list):
        print(f'iter: {i}, gamma: {gamma}')
        try:
            ## optimize loop
            idx_str = f'0{i}' if i < 10 else f'{i}' 
            opt_csv_name_str = f'opt_log{idx_str}.csv'
            save_dict = loop_run(gamma, nx, a,b, omega, oce_val, beta1, beta2,noise_level, save_path, opt_csv_name_str, seed=seed)
            
            ## save results
            with open(save_path / f'results.csv', 'a') as f:
                writer_object = DictWriter(f, save_dict.keys())
                if f.tell() == 0:
                    writer_object.writeheader()
                writer_object.writerow(save_dict)
        except:
            save_dict = {'nx': nx, 'gamma_val': gamma,'misfit': np.nan,'reg': np.nan,'u_sol1': np.nan,'u_d1': np.nan,'u_sol2': np.nan,'u_d2': np.nan,'m_sol': np.nan,'forcing': np.nan,'g1': np.nan,'g2': np.nan,'gradnorm': np.nan,'gradnorm_list': np.nan,'u_oce1': np.nan,'u_oce2': np.nan,'mtrue': np.nan,'noise_level': np.nan, 'run_time': np.nan,}
            with open(save_path / f'results.csv', 'a') as f:
                writer_object = DictWriter(f, save_dict.keys())
                if f.tell() == 0:
                    writer_object.writeheader()
                writer_object.writerow(save_dict)
            failed_iter.append((i,gamma))
            pass
    print(failed_iter)
    # os.system('say "your algorithm has finished"')



if __name__ == '__main__':
    import concurrent.futures
    seeds = range(100)
    noises = [0.01,0.03, 0.05]
    import itertools
    arguments = list(itertools.product(seeds, noises))

    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        executor.map(main, *zip(*arguments))

