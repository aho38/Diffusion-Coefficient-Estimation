from helm_eq import single_data_run
import dolfin as dl
import ufl
import numpy as np
import math
import hippylib as hp
from datetime import datetime
import os
from csv import DictWriter
from utils.general import increment_path, DboundaryL, DboundaryR, Dboundary, NboundaryL, NboundaryR, data_normalization

class synthetic_loop():
    def __init__(self, 
                nx, #"node number"
                a, #'left boundary',
                b, #'right boundary',
                bc_type, #'NBC or DBC',
                omega, # exchange rate
                oce_val, # ocean value
                normalized_mean, #'mean for normalization
                a_state_str, #'state equation'
                L_state_str, #'state equation RHS'
                a_adj_str, #'adjoint equation'
                L_adj_str, #'adjoint equation RHS'
                func='Lagrange',
                ) -> None:
        # set up some PDE domains
        self.a, self.b, self.nx = a, b, nx
        self.mesh = dl.IntervalMesh(nx, a, b)
        self.Vm = self.Vu = dl.FunctionSpace(self.mesh, func, 1)
        
        # set up trial and test functions
        self.u_trial, self.p_trial, self.m_trial = dl.TrialFunction(self.Vu), dl.TrialFunction(self.Vu), dl.TrialFunction(self.Vm)
        self.u_test, self.p_test, self.m_test = dl.TestFunction(self.Vu), dl.TestFunction(self.Vu), dl.TestFunction(self.Vm)
        self.bc_type = bc_type

        # set up some PDE parameters
        self.omega_val = dl.Constant(omega)
        self.omega = omega
        self.u_oce_val = dl.Constant(f'{oce_val - normalized_mean}')
        self.u_oce = oce_val
        self.normalized_mean = normalized_mean

        # set up state and adjoint for fwd solve
        self.a_state_str, self.L_state_str = a_state_str, L_state_str
        self.a_adj_str, self.L_adj_str = a_adj_str, L_adj_str


    def forcing_funcs_setup(self, **kwargs):
        '''
        make sure these are all function of dolfin show up in fwd and adj equation
        '''
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

    def data_setup(self, PATH = None, ud_array=None, u_true_array = None, ud = None, data_index=2, normalize=False, noise_level = None):
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

        if noise_level is not None:
            self.noise_level = noise_level
        
        if u_true_array is not None:
            self.u_true_array = u_true_array
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
    
    def misfit_reg_setup(self,gamma):
        self.gamma = gamma
        # weak for for setting up the misfit and regularization compoment of the cost
        W_equ   = ufl.inner(self.u_trial, self.u_test) * ufl.dx
        R_equ   = gamma * ufl.inner(ufl.grad(self.m_trial), ufl.grad(self.m_test)) * ufl.dx

        self.W = dl.assemble(W_equ)
        self.R = dl.assemble(R_equ)
    

    
    def cost(self, u, m):
        diff = u.vector() - self.ud.vector()
        reg = m.vector().inner(self.R*m.vector() ) 
        misfit = diff.inner(self.W * diff)
        return [0.5 * reg + 0.5 * misfit, misfit, reg]
    
    def misfit_reg_setup(self,gamma):
        self.gamma = gamma
        # weak for for setting up the misfit and regularization compoment of the cost
        W_equ   = ufl.inner(self.u_trial, self.u_test) * ufl.dx
        R_equ   = gamma * ufl.inner(ufl.grad(self.m_trial), ufl.grad(self.m_test)) * ufl.dx

        self.W = dl.assemble(W_equ)
        self.R = dl.assemble(R_equ)
    
    def wf_matrices_setup(self,m, u, p):
        # weak form for setting up matrices
        Wum_equ = ufl.inner(ufl.exp(m) * self.m_trial * ufl.grad(self.p_test), ufl.grad(p)) * ufl.dx
        C_equ   = ufl.inner(ufl.exp(m) * self.m_trial * ufl.grad(u), ufl.grad(self.u_test)) * ufl.dx
        Wmm_equ = ufl.inner(ufl.exp(m) * self.m_trial * self.m_test *  ufl.grad(u),  ufl.grad(p)) * ufl.dx
        M_equ   = ufl.inner(self.m_trial, self.m_test) * ufl.dx
        return Wum_equ, C_equ, Wmm_equ, M_equ

    def opt_loop(self, 
                m, #'diffusion coef',
                tol, #'tolerance of grad',
                c, #'constant of armijo linesearch',
                maxiter, #'maximum newton"s step',
                plot_opt_step = False,
                ):
        
        u,_,_ = run.fwd_solve(m) # initial values of diffusion


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
            self.p = p

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
            self.CT_P = CT_p
            MG = CT_p + self.R * m.vector()
            dl.solve(self.M, g, MG)
            self.g = g
            self.MG = MG

            # calculate the norm of the gradient
            grad2 = g.inner(MG)
            gradnorm = math.sqrt(grad2)

            # raise Exception('Stop')
            # set the CG tolerance (use Eisenstat–Walker termination criterion)
            if iter == 1:
                gradnorm_ini = gradnorm
            tolcg = min(0.5, (gradnorm/(gradnorm_ini+1e-15)))

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
            Psolver = dl.PETScKrylovSolver("cg", hp.amg_method())
            Psolver.set_operator(P)

            solver = hp.CGSolverSteihaug()
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
        if hasattr(self,'u_true_array'):
            self.save_dict['u_true'] = self.u_true_array.tolist()
        if hasattr(self,'noise_level'):
            self.save_dict['noise_level'] = self.noise_level
        # except:
        #     print('Something Went Wrong')
        
        return m, u

class logger():
    def __init__(self, filename):
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d-%H%M%S")
        self.save_path = increment_path(f'./log/{filename}_{dt_string}', mkdir=True)
        if os.path.exists(self.save_path / 'results.csv'):
            pass
        else:
            with open(self.save_path / 'results.csv', 'w') as f:
                pass

    def save(self, save_dict):
        with open(self.save_path / 'results.csv', 'a') as f:
            writer_object = DictWriter(f, run.save_dict.keys())
            if f.tell() == 0:
                writer_object.writeheader()
            writer_object.writerow(run.save_dict)




if __name__ == '__main__':
    nx = 32
    a,b = 0,1
    bc_type = 'DBC'
    omega = 10
    oce_val = 0.0
    

    ## ============== Set up state and adjoint for fwd solve ==============
    a_state_str = 'ufl.inner(ufl.exp(m) * ufl.grad(self.u_trial), ufl.grad(self.u_test)) * ufl.dx + self.f * self.omega_val * self.u_trial * self.u_test * ufl.dx'
    L_state_str = 'self.f * self.omega_val * self.u_oce_val * self.u_test * ufl.dx + self.g_sun * self.u_test * ufl.dx'
    a_adj_str = 'ufl.inner(ufl.exp(m) * ufl.grad(self.p_trial), ufl.grad(self.p_test)) * ufl.dx + self.f * self.omega_val * self.p_trial * self.p_test * ufl.dx'
    L_adj_str = '-ufl.inner(u - self.ud, self.p_test) * ufl.dx'

    run = synthetic_loop(nx, a, b, bc_type, omega, oce_val,oce_val, a_state_str, L_state_str, a_adj_str, L_adj_str)

    ### ================= Forcing Term ==================
    from utils.general import normalize_function
    c_1 = 10
    peak_loc = 0.3
    gauss_var = 1/10
    f_str =  f'std::exp(-pow(x[0] - {peak_loc}, 2) / {gauss_var}) * {c_1}'
    f = normalize_function(f_str, run.Vm, ufl.dx)

    ### ================= solar forcing =====================
    c_1 = 5
    peak_loc = 0.2
    gauss_var = 1/50
    g_str =  f'std::exp(-pow(x[0] - {peak_loc}, 2) / {gauss_var}) * {c_1}'
    g = dl.Expression(g_str, degree=5)
    g = dl.interpolate(g,run.Vm)

    run.forcing_funcs_setup(f=f, g_sun=g)

    ### ============== true diffusion coef =================
    # The true and inverted parameter
    peak_loc = 0.4
    gauss_var = 30
    mtrue_expression_str = f'std::log(1.+ 10.*std::exp(-pow(x[0] - {peak_loc}, 2) * {gauss_var}))'
    mtrue_expression = dl.Expression(mtrue_expression_str, degree=5)
    mtrue =  dl.interpolate(mtrue_expression,run.Vm)
    mtrue_array = mtrue.compute_vertex_values()
    run.mtrue_setup(mtrue)

    ### ============== BC setup ==============
    u0L = 1.0#cos(0.0)*3*pi
    u0R = 0.0 #cos(3*pi)*3*pi
    print(f'u0L: {u0L}, u0R: {u0R}')
    run.BC_setup(u0L, u0R)

    ### ============== data setup =============
    from utils.general import sigmoid_dx2
    x_sigmoid = np.linspace(-1,1,nx+1)
    u,_,_ = run.fwd_solve(mtrue)
    u_true = u.compute_vertex_values()

    beta = -0.03
    adj_func = beta*sigmoid_dx2(x_sigmoid,k=5)
    u_true += adj_func

    np.random.seed(0)
    noise_level = 0.03
    max_norm = np.max(abs(u_true))
    noise = noise_level * max_norm * np.random.normal(0,1,u_true.shape)
    noise[0] = noise [-1] = 0
    ud_array = u_true + noise
    ud = dl.Function(run.Vu)
    ud.vector().set_local(ud_array[::-1])
    run.data_setup(ud_array=ud_array, noise_level=noise_level, u_true_array=u_true)

    ### ========== initial guess m ==========
    m = dl.interpolate(dl.Expression("std::log(2)", degree=1),run.Vm)
    # m.assign(mtrue)
    u,_,_ = run.fwd_solve(m)

    ### ========== gamma loop ===========
    file_name = f'Helm_beta_{int(beta * 1000)}div1000_zeroderiv_test'
    logger = logger(file_name)
    gamma_list = np.logspace(-7, -4, 50)
    for gamma in gamma_list:
        ### ==================== run optimization loop ==========
        tol = 1e-9
        c = 1e-4
        maxiter=100
        # gamma =4e-5
        run.misfit_reg_setup(gamma)

        # minitial guess m
        m = dl.interpolate(dl.Expression("std::log(2)", degree=1),run.Vm)
        # m.assign(mtrue)

        m, u = run.opt_loop(m, tol, c, maxiter, plot_opt_step=False)

        logger.save(run.save_dict)

    import matplotlib.pyplot as plt
    x = np.linspace(0,1,nx+1)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 3.6)) 
    ax[0].plot(x,ud_array, 'o-', label=r'$u_d$')
    ax[0].plot(x,u.compute_vertex_values(), 'o-', label=r'$u_{sol}$')
    ax[0].set_xlabel(r'$x$')
    ax[0].set_ylabel(r'$u$')
    ax[0].set_title(f'Model Misfit, $\gamma = {run.gamma:1.1e}$', fontsize=15)
    ax[0].grid()
    ax[0].legend(fontsize=12)

    ax[1].plot(x,np.exp(mtrue_array), '-', label=r'$m_{true}$', color='tab:blue', linewidth=2.5)
    ax[1].plot(x,np.exp(m.compute_vertex_values()), '-', label=r'$m_{sol}$', color='tab:red', linewidth=2.5)
    # ax[1].set_title(f'gamma = {gamma[kappa_min_ind]:.3e}')
    ax[1].set_xlabel(r'$x$')
    ax[1].set_ylabel(r'$m$')
    ax[1].set_title(r'Recovered Param vs. True Param', fontsize=15)
    ax[1].grid()
    ax[1].legend(fontsize=12)
    plt.show()
