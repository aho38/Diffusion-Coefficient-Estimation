from pathlib import Path
import dolfin as dl
import glob
import re
import json
import numpy as np


def DboundaryL(x, on_boundary):
    return on_boundary and dl.near(x[0], 0.0)

def DboundaryR(x, on_boundary):
    return on_boundary and dl.near(x[0], 1.0)

def Dboundary(x,on_boundary):
    return on_boundary

class NboundaryL(dl.SubDomain):
    def inside(self, x, on_boundary, tol = 1e-10):
        return on_boundary and dl.near(x[0], 0.0, tol)
    
class NboundaryR(dl.SubDomain):
    def inside(self, x, on_boundary, tol = 1e-10):
        return on_boundary and dl.near(x[0], 1.0, tol)
    
def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    print(path.exists())
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{str(n).zfill(2)}{suffix}")  # update path
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path

def data_normalization(ud, mean, range):
    return (ud - mean)/range

def normalize_function(func_str, fuc_space, dx):
    f = dl.Expression(func_str, degree=5)
    f = dl.interpolate(f,fuc_space)
    f_int = dl.assemble(f*dx)
    f_val = f.vector().get_local()
    f_val /= f_int
    f.vector().set_local(f_val)
    return f

def result_save(save_dir, save_dict):
    json_file = json.dumps(save_dict)
    omega = save_dict['omega']
    oce_val = save_dict['u_oce']
    gamma = save_dict['gamma_val']
    gamma_str = f"{gamma:1.2e}"[:6] + f"{gamma:1.2e}"[6:].zfill(2)
    with open(str(save_dir / f"omega={omega}-u_oce={oce_val}-g={gamma_str}.json"), 'w') as file:
        file.write(json_file)

def apply_noise(noise_level, u, A):
    MAX = u.vector().norm("linf")
    noise = dl.Vector()
    A.init_vector(noise,1)
    noise_array = noise_level * MAX * np.random.normal(0,1,u.compute_vertex_values().shape)
    noise_array[0] = noise_array[-1] = 0
    noise.set_local(noise_array)
    u.vector().axpy(1., noise)

def sensitivity_array(mtrue, utrue, distribution_array, omega_val = 1):
    import ufl
    import dolfin as dl
    nx = len(mtrue) - 1 # number of elements
    mesh = dl.UnitIntervalMesh(nx) # create mesh
    V = dl.FunctionSpace(mesh, "Lagrange", 1) # define function space

    # assign each array to function space (note that this is possible since our poly order is 1. Meaning our nodal values are the same as the coefficients in the basis functions)
    mtrue_el = dl.Function(V)
    mtrue_el.vector().set_local(mtrue)
    utrue_el = dl.Function(V)
    utrue_el.vector().set_local(utrue)
    distribution = dl.Function(V)
    distribution.vector().set_local(distribution_array)
    omega = dl.Constant(omega_val)

    # define trial and test functions
    u_trial, m_trial = dl.TrialFunction(V), dl.TrialFunction(V)
    u_test = dl.TestFunction(V)

    # define the bilinear form
    drdm_exp  = dl.inner(dl.exp(mtrue_el)*m_trial*dl.grad(utrue_el), dl.grad(u_test)) * dl.dx # get the matrix A of (u^T A m) = drdm, since m is the trial function and u is the test function. the ith column of drdm will be dr/dm_i.
    drdm = dl.assemble(drdm_exp) # this should creat an matrix with (u_i, m_j)

    def boundary(x,on_boundary):
        return on_boundary

    bc = dl.DirichletBC(V, dl.Constant(0.), boundary) # set the boundary condition to be zero for the adjoint problem we're solving
    a_goal = ufl.inner(ufl.exp(mtrue_el) * ufl.grad(u_trial), ufl.grad(u_test)) * ufl.dx + omega * distribution * u_trial * u_test * ufl.dx # construct the LHS of the adjoint problem
    A = dl.assemble(a_goal)
    bc.apply(A)
    sensitivity_array = np.zeros((drdm.array().shape))
    for i in range(drdm.array().shape[0]):
        '''solve A x = -b where x is dudm_i'''
        sensitivity = dl.Function(V)
        b_sens = dl.Vector() # define rhs which is b = dr/dm_i
        A.init_vector(b_sens, 0) # init vector
        b_sens.set_local(drdm.array()[:,i]) # set b to be dr/dm_i
        bc.apply(b_sens)
        dl.solve(A, sensitivity.vector(), -b_sens) # negative b.
        sensitivity_array[:,i] = sensitivity.vector().get_local() # this will save the sensitivity as (m_i, u_j)
    
    return sensitivity_array, mtrue_el, utrue_el


def density_calculation(t, s):
    """
    Calculate the density of a substance given the temperature and salinity.
    
    Parameters:
        t (float): The temperature in degrees Celsius.
        s (float): The salinity in parts per thousand.
    
    Returns:
        float: The density of the substance.
    """
    A = 8.24493 * 1e-1 - 4.0899 * 1e-3 * t + 7.6438 * 1e-5 * t**2 - 8.2467 * 1e-7 * t**3 + 5.3875 * 1e-9 * t**4
    B =-5.72466 * 1e-3 + 1.0227 * 1e-4 * t - 1.6546 * 1e-6 * t**2
    C = 4.83140 * 1e-4
    
    rho_0 = 999.842594 + 6.793952 * 1e-2 * t - 9.095290 * 1e-3 * t**2 + 1.001685 * 1e-4 * t**3 - 1.120083 * 1e-6 * t**4 + 6.536332 * 1e-9 * t**5
    
    rho = rho_0 + A * s + B * s**(3/2) + C * s**2
    
    return rho


def sigmoid(x,k=1):
    return 1/(1+np.exp(-k*x))

def sigmoid_dx2(x,k=1):
    return k**2 * sigmoid(x,k) * (1-sigmoid(x,k)) * (1-2*sigmoid(x,k))

def rmse(u1, u2):
    return np.sqrt(np.mean((u1-u2)**2))

def nmse(u, utrue, a=0, b=1):
    # Initialize mesh and function space
    len = u.shape[0]-1
    mesh = dl.IntervalMesh(len , a, b)
    Vm = dl.FunctionSpace(mesh, 'Lagrange', 1)
    # define fem functions
    Dsol_fem = dl.Function(Vm)
    Dtrue_fem = dl.Function(Vm)
    # Set values
    Dsol_fem.vector().set_local(u[::-1])
    Dtrue_fem.vector().set_local(utrue[::-1])
    # compute error
    error = dl.assemble((Dsol_fem - Dtrue_fem) * (Dsol_fem - Dtrue_fem) / (Dtrue_fem*Dtrue_fem) * dl.dx)
    return error / (b-a)

def curvature(reg_list, gamma_list, misfit_list):
    eta = np.array(reg_list)/np.array(gamma_list)
    rho = np.array(misfit_list)

    gamma_index = (np.array(gamma_list)).nonzero()[0][:-1]
    gamma = np.array(gamma_list)[gamma_index]

    dg = gamma_list[1:] - gamma_list[:-1]
    dg = np.array(dg[gamma_index])
    deta = (eta[1:] - eta[:-1]) / dg
    drho = (rho[1:] - rho[:-1]) / dg

    eta = eta[:-1]
    rho = rho[:-1]

    kappa = 2 * (eta * rho / deta)
    kappa = kappa * (gamma**2 * deta * rho + 2 * gamma * eta * rho + gamma**4 * eta * deta)
    kappa = kappa / (gamma**2 * eta**2 + rho**2)**(3/2)
    kappa = kappa / gamma

    kappa_min_ind = kappa.argmin()
    return kappa, kappa_min_ind


def eig_residual_test(x,u,m,f,omg, vec_return=False):
    '''compute the residual of the eigenvalue problem
        ||-d/dx ( exp(m)*u*grad(u) ) + f*omg*(u)||

    if small or close to 0, then the problem is ill-conditioned
    
    '''
    du = np.gradient(u, x, edge_order=2)
    emdu = np.exp(m)*du
    demdu = np.gradient(emdu,x, edge_order=2)
    res = -demdu + f*omg*(u)
    if vec_return:
        return res
    return np.linalg.norm(res)


def FD_discretize_equation(D, d, f, x, lbc=0, rbc=0):
    """ 
    Discretizes the equation -d/dx (D(x) d/dx u) + d(x) u = f(x) using finite differences. -D(x) * d²u/dx² - D'(x) * du/dx + d(x) u
    """

    n = len(x)  # Number of points
    dx = x[1] - x[0] 

    # Initialize the A and b for the linear stuff (Au = b)
    A = np.zeros((n, n))
    b = np.zeros(n)

    dD = np.gradient(D, x, edge_order=2)

    # Interior points
    for i in range(1, n - 1):
        A[i, i - 1] = -D[i] / dx**2 + dD[i] / (2*dx)
        A[i, i] = D[i] * 2  / dx**2 + d[i]
        A[i, i + 1] = -D[i] / dx**2 - dD[i] / (2*dx)
        b[i] = f[i]

    # Boundary conditions. adjust based on your specific problem
    A[0, 0] = 1
    b[0] = lbc
    A[n - 1, n - 1] = 1
    b[n - 1] = rbc

    return A, b


def projection(v, u):
    '''Projection of v onto u.'''
    return u * (v @ u) / (u @ u)