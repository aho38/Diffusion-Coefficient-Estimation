from helm_eq import single_data_run
from helm_eq import dual_data_run
from utils.general import increment_path, data_normalization
import dolfin as dl
import ufl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from csv import DictWriter
from datetime import datetime

def single_loop_run(gamma, a, b, omega, oce_val, save_path, opt_csv_name, bc_type,seed=0):
    ## ======== get dual $u_{1,2}$ solution ========
    import pandas as pd
    temp_df = pd.read_excel("./lake_data/aOTM summary 2013.xls", sheet_name='Temp')
    sal_df = pd.read_excel("./lake_data/aOTM summary 2013.xls", sheet_name='Sal')

    idx_list = []
    for i in range(38):
        idx = i*3 + 1
        # date = temp_df.iloc[0,idx]
        # print(date)
        idx_list.append(idx)

    temp_profiles = temp_df.iloc[44:,idx_list]
    sal_profiles = sal_df.iloc[45:,idx_list]
    # temperature = temp_profiles.mean(axis=1).to_numpy()
    temperature = temp_profiles.iloc[:,4].to_numpy()
    salinity = sal_profiles.iloc[:,4].to_numpy()

    
    temperature = data_normalization(temperature, 29.0, temperature.ptp())
    salinity = data_normalization(salinity, 27.8, salinity.ptp())

    ## ======== initialize runs and setup ========
    nx = len(temperature) - 1
    run = single_data_run(nx, a, b, bc_type, omega, oce_val, normalized_mean=oce_val, save_path=save_path)
    run.misfit_reg_setup(gamma=gamma)# This gamma will override the gamma in the class

    ## ======== set up the state and adjoint str ========
    if bc_type == 'DBC':
        a_state_str = 'ufl.inner(ufl.exp(m) * ufl.grad(self.u_trial), ufl.grad(self.u_test)) * ufl.dx + self.f * self.omega_val * self.u_trial * self.u_test * ufl.dx'
        L_state_str = 'self.f * self.omega_val * self.u_oce_val * self.u_test * ufl.dx + self.g_sun * self.u_test * ufl.dx'
        a_adj_str = 'ufl.inner(ufl.exp(m) * ufl.grad(self.p_trial), ufl.grad(self.p_test)) * ufl.dx + self.f * self.omega_val * self.p_trial * self.p_test * ufl.dx'
        L_adj_str = '-ufl.inner(u - self.ud, self.p_test) * ufl.dx'
    elif bc_type == 'NBC':
        a_state_str = 'ufl.inner(ufl.exp(m) * ufl.grad(self.u_trial), ufl.grad(self.u_test)) * ufl.dx + self.f * self.omega_val * self.u_trial * self.u_test * ufl.dx'
        L_state_str = 'self.f * self.omega_val * self.u_oce_val * self.u_test * ufl.dx + self.g_sun * self.u_test * ufl.dx  + self.j1 * self.u_test * self.ds(1) - self.j0 * self.u_test * self.ds(0)'
        a_adj_str = 'ufl.inner(ufl.exp(m) * ufl.grad(self.p_trial), ufl.grad(self.p_test)) * ufl.dx + self.f * self.omega_val * self.p_trial * self.p_test * ufl.dx'
        L_adj_str = '-ufl.inner(u - self.ud, self.p_test) * ufl.dx'
    run.state_adj_str_setup(a_state_str, L_state_str, a_adj_str, L_adj_str)

    

    ## ======== set up distribution function i called f ===========
    from utils.general import normalize_function
    c_1 = 10
    peak_loc = 0.3
    gauss_var = 1/20
    f_str =  f'std::exp(-pow(x[0] - {peak_loc}, 2) / {gauss_var}) * {c_1}'
    f = normalize_function(f_str, run.Vm, ufl.dx)
    forcing_array = f.compute_vertex_values()
    run.f_setup(f.compute_vertex_values())

    ## ========= set up boundary condition ===========
    uL = temperature[0] if bc_type == 'DBC' else 1.0
    uR = temperature[-1] if bc_type == 'DBC' else 0.0
    print(f'uL: {uL}, uR: {uR}')
    run.BC_setup(uL, uR)

    ## ========= set up solar irradiance forcing ===========
    from utils.general import normalize_function
    c_1 = 10
    peak_loc = 0.3 if bc_type == 'DBC' else 0.9
    gauss_var = 1/30
    # g_str =  f'std::exp(-pow(x[0] - {peak_loc}, 2) / {gauss_var}) * {c_1}'
    g_str = f'std::exp(-10*x[0])'
    g = normalize_function(g_str, run.Vm, ufl.dx)
    g_array = g.compute_vertex_values()

    coef = 1.73961e-6 /5.5e-5 if bc_type == 'DBC' else 0.0
    # print('coef of solar:',coef)
    g.vector().set_local(g_array[::-1]*coef)
    run.extra_f_setup(g_sun = g)
        

    ## ======== set up data ========
    ud_array = temperature if bc_type == 'DBC' else salinity
    run.data_setup(ud_array=ud_array, normalize=False)

    ## ======== Initial Guess ========
    m = dl.interpolate(dl.Expression("std::log(2)", degree=1),run.Vm)
    m0 = m.compute_vertex_values()

    ## ======== optimization loop ========
    tol = 1e-9
    c = 1e-4
    maxiter=100
    m, u = run.opt_loop(m, tol, c, maxiter, real_data = True, opt_csv_name = opt_csv_name, save_opt_log=True)

    return run.save_dict

def double_loop_run(gamma, a, b, omega, oce_val, beta1, beta2, noise_level, save_path,opt_csv_name):
    ## ======== get dual $u_{1,2}$ solution ========
    import pandas as pd
    temp_df = pd.read_excel("./lake_data/aOTM summary 2013.xls", sheet_name='Temp')
    sal_df = pd.read_excel("./lake_data/aOTM summary 2013.xls", sheet_name='Sal')

    idx_list = []
    for i in range(38):
        idx = i*3 + 1
        # date = temp_df.iloc[0,idx]
        # print(date)
        idx_list.append(idx)

    temp_profiles = temp_df.iloc[44:,idx_list]
    sal_profiles = sal_df.iloc[45:,idx_list]
    # temperature = temp_profiles.mean(axis=1).to_numpy()
    temperature = temp_profiles.iloc[:,4].to_numpy()
    salinity = sal_profiles.iloc[:,4].to_numpy()

    
    temperature = data_normalization(temperature, 29.0, temperature.ptp())
    sal_ptp = salinity.ptp()
    salinity = data_normalization(salinity, 27.8, salinity.ptp())
    ## ======== initialize runs and setup ========
    nx = len(temperature) - 1
    run = dual_data_run(nx,a,b,gamma, gamma, omega, omega, oce_val, oce_val, normalized_mean1=oce_val,normalized_mean2=oce_val,save_path=save_path)
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

    ## ======== set up distribution function i called f ===========
    from utils.general import normalize_function
    c_1 = 10
    peak_loc = 0.3
    gauss_var = 1/10
    f_str =  f'std::exp(-pow(x[0] - {peak_loc}, 2) / {gauss_var}) * {c_1}'
    f = normalize_function(f_str, run.Vm, ufl.dx)
    forcing_array = f.compute_vertex_values()
    run.f_setup(f.compute_vertex_values())


    ## ======== set up data ========
    run.data_setup(temperature, salinity, normalize=False, noise_level=noise_level)

    ## ========= set up boundary condition ===========
    u1L_D = temperature[0]
    u1R_D = temperature[-1]
    # rain_flux = 1e-7/5.5e-5/24/sal_ptp
    rhs_value = dl.assemble(run.f*run.omega2*(run.u_oce2 - run.ud2) * ufl.dx)
    u2L_N = 1.0
    u2R_N = u2L_N - rhs_value # 0.0
    print(f'Dirichlet_L: {u1L_D}, Dirichlet_R: {u1R_D},')
    print(f'NBC_L: {u2L_N}, NBC_R: {u2R_N}')
    run.BC_setup(u1L_D, u1R_D, u2L_N, u2R_N)

    ## ========= set up solar irradiance forcing ===========
    from utils.general import normalize_function
    c_1 = 20
    peak_loc = 0.3 
    gauss_var = 1/30
    g_str = f'std::exp(-10*x[0])'
    g = normalize_function(g_str, run.Vm, ufl.dx)
    g_array = g.compute_vertex_values()

    coef_temp = 1.73961e-6 /5.5e-5
    g1 = dl.Function(run.Vu)
    g1.vector().set_local(g_array[::-1]*coef_temp)

    ceof_sal = 0.0
    g2 = dl.Function(run.Vu)
    g2.vector().set_local(g_array[::-1]*ceof_sal)
    run.extra_f_setup(g1 = g1, g2 = g2)


    ## ======== Initial Guess ========
    m = dl.interpolate(dl.Expression("std::log(1)", degree=1),run.Vm)

    ## ======== optimization loop ========
    tol = 1e-9
    c = 1e-3
    maxiter=100
    m, u1, u2 = run.opt_loop(m, tol, c, maxiter, save_opt_log=True, plot_opt_step=False, plot_eigval=False, opt_csv_name=opt_csv_name)

    return run.save_dict

def main():   

    failed_iter = []
    nx = 64
    bc_type = 'NBC'
    a = 0.0
    b = 1.0
    omega = 1.0
    oce_val = 31.0 if bc_type == 'DBC' else 31.8
    gamma_list = np.logspace(-12, -2, 101)

    ## set up save path
    now = datetime.now()

    dir_name = f'{bc_type}_real_data'
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
            save_dict = single_loop_run(gamma, a, b, omega, oce_val, save_path,opt_csv_name_str, bc_type)
            
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

def main_double():
    failed_iter = []
    a = 0.0
    b = 1.0
    omega = 1.0
    oce_val = 0.0
    gamma_list = np.logspace(-12, -2, 101)

    beta1 = 0.5
    beta2 = 0.5

    now = datetime.now()

    if beta1 == 0.5 and beta2 == 0.5:
        # dir_name = 'nx_loop_double_data'
        # dir_name = f'gamma_loop_double_data_noise_0dot5_seed_{seed}'
        dir_name = f'double_data_real_data'
    else:
        raise ValueError('beta1 and beta2 should be 0.5')
    
    dt_string = now.strftime("%Y%m%d-%H%M%S")
    path_ = f'./log/{dir_name}_{dt_string}'
    save_path = increment_path(path_, mkdir=True)
    if os.path.exists(save_path / f'results.csv'):
        pass
    else:
        with open(save_path / f'results.csv', 'w') as f:
            pass
    
    for i, gamma in enumerate(gamma_list):
        print(f'iter: {i}, gamma: {gamma}')
        try:
            ## optimize loop
            idx_str = f'0{i}' if i < 10 else f'{i}' 
            opt_csv_name_str = f'opt_log{idx_str}.csv'
            save_dict = double_loop_run(gamma, a, b, omega, oce_val, beta1, beta2, noise_level=None, save_path=save_path,opt_csv_name=opt_csv_name_str)
            
            ## save results
            with open(save_path / f'results.csv', 'a') as f:
                writer_object = DictWriter(f, save_dict.keys())
                if f.tell() == 0:
                    writer_object.writeheader()
                writer_object.writerow(save_dict)
        except:
            nx = 24
            save_dict = {'nx': nx, 'gamma_val': gamma,'misfit': np.nan,'reg': np.nan,'u_sol1': np.nan,'u_d1': np.nan,'u_sol2': np.nan,'u_d2': np.nan,'m_sol': np.nan,'forcing': np.nan,'g1': np.nan,'g2': np.nan,'gradnorm': np.nan,'gradnorm_list': np.nan,'u_oce1': np.nan,'u_oce2': np.nan,'mtrue': np.nan,'noise_level': np.nan, 'run_time': np.nan,}
            with open(save_path / f'results.csv', 'a') as f:
                writer_object = DictWriter(f, save_dict.keys())
                if f.tell() == 0:
                    writer_object.writeheader()
                writer_object.writerow(save_dict)
            failed_iter.append((i,gamma))


if __name__ == '__main__':
    # main()
    main_double()