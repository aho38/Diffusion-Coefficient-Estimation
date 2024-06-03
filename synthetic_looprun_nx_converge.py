from helm_eq import single_data_run
import os
import dolfin as dl
import ufl
import numpy as np
import matplotlib.pyplot as plt
from utils.general import rmse, nmse, increment_path, DboundaryL, DboundaryR
from datetime import datetime
from csv import DictWriter


def loop_run(gamma, nx, a,b, bc_type, omega, oce_val, noise_level, save_path, opt_csv_name,solar_forcing_mag, u0L = 0.5, u0R = 1):
    ## ======== initialize runs and setup ========
    run = single_data_run(nx,a,b,bc_type, omega, oce_val, normalized_mean=0.0,save_path=save_path)
    run.misfit_reg_setup(gamma)
    x = run.mesh.coordinates()[:,0]

    ## ======== set up the state and adjoint str ========
    a_state_str = 'ufl.inner(ufl.exp(m) * ufl.grad(self.u_trial), ufl.grad(self.u_test)) * ufl.dx + self.f * self.omega_val * self.u_trial * self.u_test * ufl.dx'
    L_state_str = 'self.f * self.omega_val * self.u_oce_val * self.u_test * ufl.dx + self.g_sun * self.u_test * ufl.dx'
    a_adj_str = 'ufl.inner(ufl.exp(m) * ufl.grad(self.p_trial), ufl.grad(self.p_test)) * ufl.dx + self.f * self.omega_val * self.p_trial * self.p_test * ufl.dx'
    L_adj_str = '-ufl.inner(u - self.ud, self.p_test) * ufl.dx'

    run.state_adj_str_setup(a_state_str, L_state_str, a_adj_str, L_adj_str)

    ## ======== set up m true =============
    peak_loc = 0.4
    gauss_var = 30
    mtrue_expression_str = f'std::log(1.+ 10.*std::exp(-pow(x[0] - {peak_loc}, 2) * {gauss_var}))'
    # mtrue_expression_str = 'std::sin(pow(x[0],2)*5*pi)'
    # mtrue_expression_str = 'std::log(1)'
    mtrue_expression = dl.Expression(mtrue_expression_str, degree=5) # why do I need degree 5? 
    mtrue =  dl.interpolate(mtrue_expression,run.Vm)
    mtrue_array = mtrue.compute_vertex_values()
    # mtrue.vector().set_local(mtrue_array)
    run.mtrue_setup(mtrue)

    ## ======== set up distribution =============
    from utils.general import normalize_function
    c_1 = 1
    dist_peak_loc = 0.3
    gauss_var = 1/10
    f_str =  f'std::exp(-pow(x[0] - {dist_peak_loc}, 2) / {gauss_var}) * {c_1}'
    # f_str = '1'
    f = normalize_function(f_str, run.Vm, ufl.dx)
    f_array = f.compute_vertex_values()
    run.f_setup(f.compute_vertex_values())

    ## ======== flatness adjustment ==========
    def sigmoid(x,k=1):
        return 1/(1+np.exp(-k*x))

    def sigmoid_dx2(x,k=1):
        return k**2 * sigmoid(x,k) * (1-sigmoid(x,k)) * (1-2*sigmoid(x,k))
    
    ## ======= set up solar forcing ==========
    # solar_forcing_mag = 5
    peak_loc = 0.2
    gauss_var = 1/50
    # g_str =  f'std::exp(-pow(x[0] - {peak_loc}, 2) / {gauss_var}) * {solar_forcing_mag}'
    g_str = f_str + f' * {solar_forcing_mag}'
    # g_str = 'std::exp(-5 * x[0]-1)'
    # g_str = 'std::sin(x[0]*2*pi)'
    # g_str = f'1 * {solar_forcing_mag}'
    g_sun = dl.Expression(g_str, degree=5)
    g_sun = dl.interpolate(g_sun,run.Vm)

    run.extra_f_setup(g_sun=g_sun)

    ## ======== set up boundary conditions ==========
    print(f'u0L: {u0L}, u0R: {u0R}')
    run.BC_setup(u0L, u0R)

    ## ======== set up data ==========
    u,_,_ = run.fwd_solve(mtrue)
    u_true = u.compute_vertex_values()

    np.random.seed(0)
    max_norm = np.max(abs(u_true))
    noise = noise_level * max_norm * np.random.normal(0,1,u_true.shape)
    noise[0] = noise [-1] = 0
    if noise_level == 0:
        ud_array = u_true
    else:
        ud_array = u_true + noise

    ud = dl.Function(run.Vu)
    ud.vector().set_local(ud_array[::-1])
    run.data_setup(ud_array=ud_array)

    ## ======== numerical g_sun ===========
    du = np.gradient(u_true, x, edge_order=2)
    emdu = np.exp(mtrue.compute_vertex_values())*du
    demdu = np.gradient(emdu, x, edge_order=2)
    g_numerical = -demdu + f_array*omega*(u_true)
    g_num_fem = dl.Function(run.Vm)
    g_num_fem.vector().set_local(g_numerical[::-1])
    # run.extra_f_setup(g_sun=g_num_fem)

    # ============= initial guess m ================
    m = dl.interpolate(dl.Expression("std::log(1)", degree=1),run.Vm)
    bc_m = [dl.DirichletBC(run.Vm, mtrue_array[0],DboundaryL), dl.DirichletBC(run.Vm, mtrue_array[-1],DboundaryR)]
    bc_m[0].apply(m.vector())
    bc_m[1].apply(m.vector())
    # m.assign(mtrue)
    # m = mtrue
    u,_,_ = run.fwd_solve(m)

    ## ========= optimization loop ==========
    tol = 1e-9
    c = 1e-4
    maxiter=50
    m, u = run.opt_loop(m, tol, c, maxiter, plot_opt_step=False, opt_csv_name=opt_csv_name, save_opt_log=True)

    return run.mesh.coordinates(), m.compute_vertex_values(), mtrue_array , u.compute_vertex_values(), run.ud.compute_vertex_values(), run.g_sun.compute_vertex_values(), run.f.compute_vertex_values(), run.save_dict

def main():
    # nx_list = [4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,160,192,224,256,320,352,384,416,448,480,512,1024]
    # nx_list = [8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,68,70,72,74,76,78,80]
    nx = 1024
    # dist_peak_loc_list = [0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # solar_forcing_mag_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50,55,60,65,70]
    solar_forcing_mag_list = np.linspace(0, 6,93)
    solar_forcing_mag_list = np.linspace(0,10,21)
    # solar_forcing_mag = 1.5
    # solar_forcing_mag = 5
    a = 0.0
    b = 1.0
    omega = 1.0
    # omega_list = np.linspace(1.0, 0.0, 11)
    oce_val = 0.0
    gamma = 1e-8
    noise_level = 0.00
    bc_type = 'DBC'
    # Boundary Condition
    u0L = 1.0
    u0R = 0.0

    ## set up save path
    now = datetime.now()
    # dir_name = 'analytical_solution_nx_converge'
    # dir_name = 'DBC_only_one_misfit_nx_converge'
    # dir_name = 'DBC_only_one_misfit_nx_converge_poor_behave'
    # dir_name = 'solar_forcing_mag_change_mtrue_gauss'
    # dir_name = 'solar_forcing_mag_change_mtrue_3oscillation_well_behaved'
    # dir_name = 'attempt_og_run'
    dir_name = 'Paper_story_exploartion_solar_mag_change_const_rhs_larger_REG_m_bc_fixed'
    dt_string = now.strftime("%Y%m%d-%H%M%S")
    path_ = f'./log/{dir_name}_{dt_string}'
    save_path = increment_path(path_, mkdir=True)
    if os.path.exists(save_path / f'results.csv'):
        pass
    else:
        with open(save_path / f'results.csv', 'w') as f:
            pass

    ## ===================
    failed_iter = {}
    # for i , nx in enumerate(nx_list):
    #     print(f'iter: {i}, nx: {nx}')
    # for i, dist_peak_loc in enumerate(dist_peak_loc_list):
    for i , solar_forcing_mag in enumerate(solar_forcing_mag_list):
        print(f'iter: {i}, solar_forcing_mag: {solar_forcing_mag}')
    # for i, omega in enumerate(omega_list):
    #     print(f'iter: {i}, omega: {omega}')
        try:
            ## optimize loop
            idx_str = f'0{i}' if i < 10 else f'{i}' 
            opt_csv_name_str = f'opt_log{idx_str}.csv'
            _, _, _, _, _, _, _, save_dict = loop_run(gamma, nx, a,b, bc_type, omega, oce_val,noise_level, save_path, opt_csv_name_str, solar_forcing_mag, u0L, u0R)

            ## add loop results to save dict
            save_dict = {'solar_mag': solar_forcing_mag, **save_dict}
            # save_dict["solar_mag"] = solar_forcing_mag
            
            ## save results
            with open(save_path / f'results.csv', 'a') as f:
                writer_object = DictWriter(f, save_dict.keys())
                if f.tell() == 0:
                    writer_object.writeheader()
                writer_object.writerow(save_dict)
        except:
            failed_iter[i] = nx
            pass
    print(failed_iter)





if __name__ == '__main__':
    main()