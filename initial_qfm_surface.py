import numpy as np

from simsopt.configs import get_w7x_data
from simsopt.field import BiotSavart, coils_via_symmetries
from simsopt.geo import QfmSurface, SurfaceRZFourier, ToroidalFlux

nfp = 5  # Number of field periods.
current_scale = -12985/15000 # Scaling currents. This convention ensures toroidal flux >0

def build_field():
    """Makes W7-X field. Returns (field, magnetic axis curve)."""

    curves, currents, axis = get_w7x_data()
    coils = coils_via_symmetries(curves, np.array(currents) * current_scale, nfp, True)
    return BiotSavart(coils), axis



def initial_surface(axis, minor_radius, mpol=3, nphi=40, ntheta=40):
    """Creates an initial surface defined by a constant minor radius from the magnetic axis, ie a 'twisted donut'.

    mpol is poloidal mode number. We set ntor, the toroidal mode number, to match mpol in this function.

    Sets our grid to be one full poloidal turn and one toroidal field period (theta in [0,1], phi in [0,1/nfp]).

    flip_theta= True sets the poloidal direction so that the flux is always positive.
    """
    surface = SurfaceRZFourier(mpol=mpol, ntor=mpol, stellsym=True, nfp=nfp,
                               quadpoints_phi = np.linspace(0, 1/nfp, nphi, endpoint= False),
                               quadpoints_theta = np.linspace(0,1, ntheta, endpoint=False))
    surface.fit_to_curve(axis, minor_radius, flip_theta=True)

    return surface


def toroidal_flux(field, surface):
    """Gets the toroidal flux through a given surface in Wb."""
    return ToroidalFlux(surface, BiotSavart(field.coils)).J()



def optimise_qfm(field, surface, target_flux, tol=1e-9):

    """"Deforms the initial surface to minimise quadratic flux
    subject to the constraint on toroidal flux.

    Modifies the surface and returns the residual (the normalised quadratic flux).

    Two stage solve: LBFGS then SLSQP.
    """

    flux = ToroidalFlux(surface, BiotSavart(field.coils))
    qfm = QfmSurface(BiotSavart(field.coils), surface, flux, target_flux)

    qfm.minimize_qfm_penalty_constraints_LBFGS(tol=tol, maxiter = 1000, constraint_weight=1.0)

    qfm.minimize_qfm_exact_constraints_SLSQP(tol=tol, maxiter=1000)

    return float(np.linalg.norm(qfm.qfm.J()))


def make_anchor(minor_radius=0.2, mpol=3):
    """Builds the anchor surface. Returns (surface, flux, residual)."""
    field, axis = build_field()
    surface = initial_surface(axis, minor_radius, mpol)

    target = toroidal_flux(field, surface)
    residual = optimise_qfm(field, surface, target)

    return surface, toroidal_flux(field, surface), residual

if __name__ == "__main__":
    surface, flux, residual = make_anchor(minor_radius=0.2)
    print(f"anchor: flux = {flux:.4f} Wb, residual = {residual:.3e}")
    surface.save(f"anchor_flux_{flux:.4f}.json")