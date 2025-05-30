
import numpy as np

def _integrate(values: np.ndarray, dx: float, dy: float = None, dz: float = None) -> float:
    if dy is None:
        dy = dx
    if dz is None:
        dz = dx
    dV = dx * dy * dz
    # potential snag: floating point precision depends on order of sum and multiplication,
    # but so does performance, slightly
    return np.sum(values) * dV

def _grad_squared(function: np.ndarray, dx: float, dy: float = None, dz: float = None) -> np.ndarray:
    '''returns (∇f)^2, using a symmetric difference quotient
    dx should be the spacing between adjacent grid points'''
    if dy is None:
        dy = dx
    if dz is None:
        dz = dx
    
    grad_squared = np.zeros_like(function)
    it = np.nditer(function, flags=['multi_index'])
    # a symmetric differnence doesn't use the value at the point, so we don't actually need to keep it
    for _ in it:
        ix, iy, iz = it.multi_index
        gx = (function[ix + 1, iy, iz] - function[ix - 1, iy, iz]) / (2*dx)
        gy = (function[ix, iy + 1, iz] - function[ix, iy - 1, iz]) / (2*dy)
        gz = (function[ix, iy, iz + 1] - function[ix, iy, iz - 1]) / (2*dz)
        grad_squared[ix, iy, iz] = gx**2 + gy**2 + gz**2
    return grad_squared	

def kinetic_energy(density: np.ndarray) -> float:
    return 0

def ke_gradient(density: np.ndarray) -> np.ndarray:
    return np.zeros(density.shape)

def hartree_energy(density: np.ndarray) -> float:
    return 0

def hartree_gradient(density: np.ndarray) -> np.ndarray:
    return np.zeros(density.shape)

def xc_energy(density: np.ndarray) -> float:
    return 0

def xc_gradient(density: np.ndarray) -> np.ndarray:
    return np.zeros(density.shape)

energy = np.infty
energy_tolerance = 1e-6 # or whatever
gradient_scale = .1

if __name__ == "__main__":
    box_dims = np.array([8.0, 4.0, 1.0]) # angstroms
    resolution = 20 # points per anstrom
    density = np.ones(resolution*box_dims)
    # fill in density with appropriate guess (uniform?)

    while True:
        previous_energy = energy
        # calculate energy of configuration
        energy = kinetic_energy(density) + hartree_energy(density) + xc_energy(density)

        # check convergence
        if abs(previous_energy - energy) < energy_tolerance:
            break # converged

        # calculate gradient
        energy_gradient = ke_gradient(density) + hartree_gradient(density) + xc_gradient(density)

        # nudge in that direction
        density -= energy_gradient * gradient_scale
        # repeat
    
    print(energy)