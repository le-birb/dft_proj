
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

_tf_factor = 3/10 * (3*np.pi**2)**(2/3)

def kinetic_energy(density: np.ndarray, dx: float) -> float:
    TF = np.power(density, 5/3)
    return _tf_factor * _integrate(TF, dx) 

def ke_gradient(density: np.ndarray, dx: float) -> np.ndarray:
    tf_term = 5/3 * density**(2/3) * _tf_factor
    return tf_term

# this should always be the same for a given calculation
@lru_cache
def _delta_r_base(shape: tuple[int, ...],  dx: float) -> np.ndarray:
    "Returns an array of 1/r', with 0 at (0, 0, 0)" 
    grid = np.zeros(shape)
    it = np.nditer(grid, flags=['multi_index'])
    for _ in it:
        xp, yp, zp = it.multi_index
        grid[xp, yp, zp] = 1/np.abs(xp**2 + yp**2 + zp**2)
    grid[0,0,0] = 0
    return grid / dx

def _inv_delta_r(shape: tuple[int, ...], r_index: tuple[int, int, int], dx: float) -> np.ndarray:
    """
    Returns an array of values of 1/abs(r - r'), as a function of r', given a particular r
    At r'=r, it puts 0
    """
    # we take an array with r at (0, 0, 0), and construct a new array by copying relevant pieces into their proper places
    base = _delta_r_base(shape, dx)
    xi, yi, zi = r_index
    x_end, y_end, z_end = shape
    translated_grid = np.zeros_like(base)
    # the precalcualted distances may simply be copied into place
    translated_grid[xi:, yi:, zi:] = base[:x_end-xi, :y_end-yi, :z_end-zi]
    translated_grid[:xi, yi:, zi:] = base[xi:0:-1,   :y_end-yi, :z_end-zi]
    translated_grid[xi:, :yi, zi:] = base[:x_end-xi, yi:0:-1,   :z_end-zi]
    translated_grid[:xi, :yi, zi:] = base[xi:0:-1,   yi:0:-1,   :z_end-zi]
    translated_grid[xi:, yi:, :zi] = base[:x_end-xi, :y_end-yi, zi:0:-1  ]
    translated_grid[:xi, yi:, :zi] = base[xi:0:-1,   :y_end-yi, zi:0:-1  ]
    translated_grid[xi:, :yi, :zi] = base[:x_end-xi, yi:0:-1,   zi:0:-1  ]
    translated_grid[:xi, :yi, :zi] = base[xi:0:-1,   yi:0:-1,   zi:0:-1  ]

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
        r_index = it.multi_index
        gradient[r_index] = _integrate(density * _inv_delta_r(density.shape, r_index, dx), dx)
    return .5 * gradient

_lda_factor = -3/4*(3/np.pi)**(1/3)
_a0 = 1 # angstroms
_a = (np.log(2) - 1)/(2*np.pi**2)
_b = 20.4562557
_inv_rs_factor = (4*np.pi/3)**(1/3) / _a0

def correlation_energy(density: np.ndarray, dx: float) -> float:
    inv_rs = density**(1/3) * _inv_rs_factor
    integrand = density * _a * np.log(1 + _b * (inv_rs + inv_rs**2))
    return _integrate(integrand, dx)

def exchange_energy(density: np.ndarray, dx: float) -> float:
    return _integrate(_lda_factor * density**(4/3), dx)


def xc_gradient(density: np.ndarray, dx: float) -> np.ndarray:
    cbrt_dens = density**(1/3)
    x_gradient = _lda_factor * 4/3 * cbrt_dens
    inv_rs = _inv_rs_factor * cbrt_dens
    eps_c = _a * np.log(1 + _b * (inv_rs + inv_rs**2))
    d_eps_c_drho = _a*_b*_inv_rs_factor**3/3 * inv_rs**-2 * (1 + 2*inv_rs)/(1 + _b*inv_rs*(1 + inv_rs))
    return x_gradient + eps_c + density * d_eps_c_drho

energy = np.inf
energy_tolerance = 1e-4 # or whatever
gradient_scale = .01

new_density_frac = .3

if __name__ == "__main__":
    box_dims = np.array([16.0, 8.0, 3.0]) # bohr
    box_volume = np.prod(box_dims)
    points_per_bohr = 5
    electron_count = 14
    dx = 1/points_per_bohr
    box_shape = (box_dims*points_per_bohr).astype(np.int_)

    # density = _initial_density(box_shape, dx, electron_count)
    density = np.ones(box_shape) * electron_count / box_volume

    print(f"Min density: {np.min(density)}")
    print(f"Electron count: {_integrate(density, dx)}")

    def delta_n(density: np.ndarray) -> float:
        # maybe dx should be a parameter but whatever
        return _integrate(density, dx) - electron_count

    for i in range(200):
        print(f"Beginning iteration {i}")
        previous_energy = energy
        # calculate energy of configuration
        kinetic  = kinetic_energy(density, dx)
        hartree = hartree_energy(density, dx)
        exchange = exchange_energy(density, dx)
        correlation = correlation_energy(density, dx)
        energy = kinetic + hartree + exchange + correlation
        print(f"Current energy: {energy}")
        print(f"Hartree energy: {hartree}")
        print(f"Kinetic energy: {kinetic}")
        print(f"Exchange energy: {exchange}")
        print(f"Correlation energy: {correlation}")
        print(f"Energy change: {energy - previous_energy}")
        print("")

        if abs(previous_energy - energy) < energy_tolerance:
            break # converged

        kegrad = ke_gradient(density, dx)
        hgrad = hartree_gradient(density, dx)
        xcgrad = xc_gradient(density, dx)
        energy_gradient = kegrad + hgrad + xcgrad

        # this term is to maintain the constraint of constant electron number
        # essentially, we compensate for the change in electron number due to the gradient step
        # by adding in a constant density to bring the totoal cound back in line
        lagrange_multiplier = _integrate(energy_gradient, dx) * gradient_scale / box_volume
        new_density = density - energy_gradient * gradient_scale + lagrange_multiplier
        # density mixing helps solutions be more stable
        density = density*(1-new_density_frac) + new_density*new_density_frac

        print(f"Min density: {np.min(density)}")
        print(f"Electron count: {_integrate(density, dx)}")
        print(f"Multiplier: {lagrange_multiplier}")
        print("")
    
    print(f"final result: {energy}")
