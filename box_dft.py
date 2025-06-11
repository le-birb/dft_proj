
from functools import lru_cache
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
        lx, ly, lz = function.shape
        # i + 1 needs to be wrapped around explicitly, i- 1 does not
        gx = (function[(ix + 1) % lx, iy, iz] - function[ix - 1, iy, iz]) / (2*dx)
        gy = (function[ix, (iy + 1) % ly, iz] - function[ix, iy - 1, iz]) / (2*dy)
        gz = (function[ix, iy, (iz + 1) % lz] - function[ix, iy, iz - 1]) / (2*dz)
        grad_squared[ix, iy, iz] = gx**2 + gy**2 + gz**2
    return grad_squared	

def _laplacian(function: np.ndarray, dx: float, dy: float = None, dz: float = None) -> np.ndarray:
    # if dy is None:
    #     dy = dx
    # if dz is None:
    #     dz = dx

    laplacian = np.zeros_like(function)
    it = np.nditer(function, flags=['multi_index'])
    for value in it:
        ix, iy, iz = it.multi_index
        lx, ly, lz = function.shape
        laplacian[ix, iy, iz] = \
            (
                function[(ix + 1) % lx, iy, iz] + function[ix - 1, iy, iz] + \
                function[ix, (iy + 1) % ly, iz] + function[ix, iy - 1, iz] + \
                function[ix, iy, (iz + 1) % lz] + function[ix, iy, iz - 1] - \
                6*value
            ) / dx**2 # TODO: think about adding support for different dx, dy, dz; think about it real hard
    return laplacian

_tf_factor = 3/10 * (3*np.pi**2)**(2/3)
_vw_factor = 1/8

def kinetic_energy(density: np.ndarray, dx: float) -> float:
    TF = np.power(density, 5/3)
    VW = _grad_squared(density, dx)/density
    return _tf_factor * _integrate(TF, dx) # + _vw_factor * _integrate(VW, dx)

def ke_gradient(density: np.ndarray, dx: float) -> np.ndarray:
    tf_term = 5/3 * density**(2/3)
    vw_term = _grad_squared(density, dx) / (8 * density**2) - _laplacian(density, dx) / (4 * density)
    return tf_term # + vw_term

def ke_lagrange_gradient(density: np.ndarray, dx: float) -> np.ndarray:
    klg_dens = density**(-1/3)
    return _tf_factor * (10/9) * klg_dens

# this should always be the same for a given calculation
@lru_cache
def _delta_r_base(shape: tuple[int, ...],  dx: float) -> np.ndarray:
    grid = np.zeros(shape)
    it = np.nditer(grid, flags=['multi_index'])
    for _ in it:
        xp, yp, zp = it.multi_index
        grid[xp, yp, zp] = 1/np.abs(xp**2 + yp**2 + zp**2)
    grid[0,0,0] = 0
    return grid / dx

def _inv_delta_r(shape: tuple[int, ...], ri: tuple[int, int, int], dx: float) -> np.ndarray:
    base = _delta_r_base(shape, dx)
    xi, yi, zi = ri
    lx, ly, lz = shape
    translated_grid = np.zeros_like(base)
    # the precalcualted distances may simply be copied into place
    translated_grid[xi:, yi:, zi:] = base[:lx-xi,  :ly-yi,  :lz-zi ]
    translated_grid[:xi, yi:, zi:] = base[xi:0:-1, :ly-yi,  :lz-zi ]
    translated_grid[xi:, :yi, zi:] = base[:lx-xi,  yi:0:-1, :lz-zi ]
    translated_grid[:xi, :yi, zi:] = base[xi:0:-1, yi:0:-1, :lz-zi ]
    translated_grid[xi:, yi:, :zi] = base[:lx-xi,  :ly-yi,  zi:0:-1]
    translated_grid[:xi, yi:, :zi] = base[xi:0:-1, :ly-yi,  zi:0:-1]
    translated_grid[xi:, :yi, :zi] = base[:lx-xi,  yi:0:-1, zi:0:-1]
    translated_grid[:xi, :yi, :zi] = base[xi:0:-1, yi:0:-1, zi:0:-1]

    return translated_grid

def hartree_energy(density: np.ndarray, dx: float) -> float:
    integrand = np.zeros_like(density)
    it = np.nditer(density, flags=['multi_index'])
    for dens_at_r in it:
        ri = it.multi_index
        integrand[ri] = dens_at_r * _integrate(density * _inv_delta_r(density.shape, ri, dx), dx)
    return .5 * _integrate(integrand, dx)

def hartree_gradient(density: np.ndarray, dx: float) -> np.ndarray:
    gradient = np.zeros_like(density)
    it = np.nditer(density, flags=['multi_index'])
    for _ in it:
        ri = it.multi_index
        gradient[ri] = _integrate(density * _inv_delta_r(density.shape, ri, dx), dx)
    return .5 * gradient
#Should be zero for this system
def hartree_lagrange_gradient(density: np.ndarray, dx: float) -> float:
    return 0

_lda_factor = -3/4*(3/np.pi)**(1/3)
_a0 = .529 # angstroms
_a = (np.log(2) - 1)/(2*np.pi**2)
_b = 20.4562557
_inv_rs_factor = (4*np.pi/3)**(1/3) / _a0

def xc_energy(density: np.ndarray, dx: float) -> float:
    inv_rs = density**(1/3) * _inv_rs_factor
    integrand = _lda_factor * density**(4/3) + density * _a * np.log(1 + _b * (inv_rs + inv_rs**2))
    return _integrate(integrand, dx)

def xc_gradient(density: np.ndarray, dx: float) -> np.ndarray:
    cbrt_dens = density**(1/3)
    x_gradient = _lda_factor * 4/3 * cbrt_dens
    inv_rs = _inv_rs_factor * cbrt_dens
    eps_c = _a * np.log(1 + _b * (inv_rs + inv_rs**2))
    d_eps_c_drho = _a*_b*_inv_rs_factor**3/3 * inv_rs**-2 * (1 + 2*inv_rs)/(1 + _b*inv_rs*(1 + inv_rs))
    return x_gradient + eps_c + density * d_eps_c_drho

def xc_lagrange_gradient(density: np.ndarray, dx: float) -> float:
    xclg_dens = density**(-1/3)
    return _lda_factor * xclg_dens * (4/9)

def _initial_density(shape: tuple[int, int, int], dx: float, electron_count: int) -> np.ndarray:
    # noninteracting pib states are sqrt(2/Lx) sin(n pi x / Lx) in each direction
    # shift everything over by dx/2 to be in the centers of the grid cells instead of the corners
    x_centers = np.arange(shape[0]) * dx + dx/2
    y_centers = np.arange(shape[1]) * dx + dx/2
    z_centers = np.arange(shape[2]) * dx + dx/2
    Lx = shape[0] * dx
    Ly = shape[1] * dx
    Lz = shape[2] * dx

    def pib_energy(nx, ny, nz):
        return (nx/Lx)**2 + (ny/Ly)**2 + (nz/Lz)**2

    state_candidates = [(1, 1, 1)]
    old_states = set()
    density = np.zeros(shape)
    electrons_left = electron_count
    # the wavefunction norm factor is the sqrt of this, but we want a density
    normalization_factor = 8/(Lx*Ly*Lz)

    while electrons_left > 0:
        electrons_in_next_state = min(electrons_left, 2)
        electrons_left -= electrons_in_next_state

        nx, ny, nz = state_candidates.pop(0)
        old_states.add((nx, ny, nz))
        x_axis = np.sin(nx*np.pi*x_centers/Lx)**2
        y_axis = np.sin(ny*np.pi*y_centers/Ly)**2
        z_axis = np.sin(nz*np.pi*z_centers/Lz)**2
        
        density += electrons_in_next_state * normalization_factor * np.einsum('i,j,k', x_axis, y_axis, z_axis) # this einsum builds a 3d grid out of the axes in the way we want
        
        # the next highest energy state is either already present in the list, or is one of these
        if (nx + 1, ny, nz) not in old_states and (nx + 1, ny, nz) not in state_candidates:
            state_candidates.append((nx + 1, ny, nz))
        if (nx, ny + 1, nz) not in old_states and (nx, ny + 1, nz) not in state_candidates:
            state_candidates.append((nx, ny + 1, nz))
        if (nx, ny, nz + 1) not in old_states and (nx, ny, nz + 1) not in state_candidates:
            state_candidates.append((nx, ny, nz + 1))
        # make sure that the lowest energy state is at position 0
        state_candidates.sort(key = lambda n: pib_energy(*n))

    return density
    

energy = np.inf
energy_tolerance = 1e-3 # or whatever
gradient_scale = 1

new_density_frac = .1

if __name__ == "__main__":
    box_dims = np.array([16.0, 8.0, 2.0]) # bohr
    points_per_angstrom = 5
    electron_count = 14
    dx = 1/points_per_angstrom
    box_shape = (box_dims*points_per_angstrom).astype(np.int_)

    density = _initial_density(box_shape, dx, electron_count)

    for i in range(20):
        print(f"Beginning iteration {i}")
        previous_energy = energy
        # calculate energy of configuration
        energy = kinetic_energy(density, dx) + hartree_energy(density, dx) + xc_energy(density, dx)
        print(energy)

        # check convergence
        if abs(previous_energy - energy) < energy_tolerance:
            break # converged

        # calculate gradient
        energy_gradient = ke_gradient(density, dx) + hartree_gradient(density, dx) + xc_gradient(density, dx)

        # nudge in that direction
        new_density = density - energy_gradient * gradient_scale
        density = density*(1-new_density_frac) + new_density*new_density_frac
    
    print(f"final result:{energy}")