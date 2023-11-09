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

def sensitivity_array(mtrue, utrue):
    import ufl
    nx = len(mtrue) - 1
    mesh = dl.UnitIntervalMesh(nx)
    V = dl.FunctionSpace(mesh, "Lagrange", 1)

    mtrue_el = dl.Function(V)
    mtrue_el.vector().set_local(mtrue)
    utrue_el = dl.Function(V)
    utrue_el.vector().set_local(utrue)

    u_trial, p_trial, m_trial = dl.TrialFunction(V), dl.TrialFunction(V), dl.TrialFunction(V)
    u_test, p_test, m_test = dl.TestFunction(V), dl.TestFunction(V), dl.TestFunction(V)

    drdm_exp  = dl.inner(dl.exp(mtrue_el)*m_trial*dl.grad(utrue_el), dl.grad(u_test)) * dl.dx
    drdm = dl.assemble(drdm_exp)

    def boundary(x,on_boundary):
        return on_boundary

    bc = dl.DirichletBC(V, dl.Constant(0.), boundary)
    a_goal = ufl.inner(ufl.exp(mtrue_el) * ufl.grad(u_trial), ufl.grad(u_test)) * ufl.dx
    A = dl.assemble(a_goal)
    bc.apply(A)
    sensitivity_array = np.zeros((drdm.array().shape))
    for i in range(drdm.array().shape[0]):
        sensitivity = dl.Function(V)
        b_sens = dl.Vector()
        A.init_vector(b_sens, 1)
        b_sens.set_local(drdm.array()[:,i])
        bc.apply(b_sens)
        dl.solve(A, sensitivity.vector(), -b_sens)
        sensitivity_array[i,:] = sensitivity.vector()
    
    return sensitivity_array