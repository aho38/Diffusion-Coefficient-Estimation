import numpy as np
import dolfin

def generate_random_vector(v: dolfin.function.function.Function, mean: float = 0., std: float = 1., seed: int = 0, ):
    np.random.seed(seed)
    if hasattr(v, 'vector'):
        shape = v.vector().get_local().shape
    else:
        shape = v.get_local().shape
    rand_vector = np.random.normal(mean, std, shape)
    if hasattr(v, 'vector'):
        v.vector().set_local(rand_vector)
    else:
        v.set_local(rand_vector)
    return v