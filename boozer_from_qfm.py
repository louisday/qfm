"""Convert saved QFM surfaces into Boozer surfaces, plot the iota and iota' profiles,
interpolate to estimate iota' at the resonance.

A Boozer surface is parametrised by Boozer coordinates which is the coordinate
system used by Y. Feng in the derivation of the island width formula. Futher,
in these coordinates field lines are straight and iota is found by simple
differentiation: iota = dtheta/dphi.

We load each QFM surface, convert it to a SurfaceXYZTensorFourier, and use the
built-in Simsopt solver BoozerSurface to generate a Boozer surface whilst keeping
the toroidal flux. We need to make a guess at iota for the Boozer solver.

We extract the iota profile from the Boozer surfaces and take a derivative to
find the iota' profile (against flux). We interpolate to find an approximation
for the magnetic shear at the resonance relative to flux.

The formula for island width requires d|iota|/dr, but we have d|iota|/dPsi,
so we can write:

    iota' = d|iota|/dr = (d|iota|/dPsi) * (dPsi/dr)

The second factor is estimated by finding the effective minor radius - the radius
of a circle with cross-sectional area equal to our surface, given by:

    r(Psi) = sqrt( A(Psi) / pi )

where A is the area enclosed by the surface's cross-section at phi = 0. We calculate
both factors from finite differences over the same two surfaces either side of the
island band.

Key parameters to set:
- the QFM surfaces to load: mpol (we suggest mpol=6), grid, fluxes. These must
be saved in "qfm full set [config]".
- the headroom: this is the additional Fourier modes to allow the Boozer solver
to use. HEADROOM = 2 seems to work.
- the iota guess: this must be negative for our sign convention. For the standard
config, since you will start on a surface just below the resonance, you want a
guess around -0.97. The iota from the solved surface will serve as the guess for
the next surface.
- the resonant flux: this only selects the pair of surfaces to estimate iota' from.
This should be the flux of the high Fourier resolution QFM surface that passes through
the x and o points.
- which plots to show.

Things to highlight:
1. BoozerSurface will not accept a SurfaceRZFourier object. This motivates our conversion
to SurfaceXYZTensorFourier.
2. The Boozer solver needs its own grid of at least 2*mpol+1 points in each angle
since we have increased the Fourier modes.
3. G is just a helpful check that the solver hasn't failed, it does not tell you
anything about whether the surface is nested or self-intersecting. Each surface
has to be checked manually on the Poincare plot.
4. If you try and make a Boozer surface inside the island band you will get self-intersecting
surfaces but the iota profile may still look sensible and monotonic. So check the surfaces.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from simsopt._core import load
from simsopt.field import BiotSavart
from simsopt.geo import (BoozerSurface, SurfaceRZFourier,
                         SurfaceXYZTensorFourier, ToroidalFlux)

from w7x_config import NFP, build_field
from initial_qfm_surface import (CONFIG, PLOT_NTHETA, cross_section_rz,
                                 plot_all, surface_path, toroidal_flux)

SAVE = True             # True means save the surfaces and the profile data. False means solve and plot only.

SURFACE_DIR = "qfm full set"   # which QFM library to load surfaces from

""" Which saved QFM surfaces to convert. These must already exist in
 "[SURFACE_DIR] [config]" at the given mpol and grid. We have been using
 mpol=6 surfaces for the Boozer solves."""
SOURCE_MPOL = 6
SOURCE_NPHI = SOURCE_NTHETA = 48
FLUXES = (0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9)

# The Boozer solve is more reliable if given an extra 2 Fourier modes.
HEADROOM = 2


"""Starting guess for iota to give to the Boozer solve. Iota is negative in our convention
due to the negative currents (this leads to positive toroidal flux)."""
IOTA_GUESS = -0.97

"""Sometimes the iota guess doesn't converge so we guess around it to see if these guesses converge."""
IOTA_OFFSETS = (0.0, -0.02, 0.02, -0.05, 0.05, -0.08, 0.08)

# This tolerance appears to lead to convergence.
NEWTON_TOL = 1e-11

"""We calculate the Boozer solve's internal parameter G (mu0*sum|I|) ourselves and compare to
the output to serve as a check that the surface isn't degenerate."""
G_TOLERANCE = 0.02
MU0 = 4 * np.pi * 1e-7

# The folder name to save surfaces into. They will be saved as "boozer surfaces [config]".
BOOZER_DIR = "boozer surfaces"

"""This is the flux of the resonant surface so the magnetic shear estimate
knows where to interpolate to. You must determine this from finding the QFM surface
that passes through the x/o points."""
RESONANT_FLUX = 2.23
"""How many theta samples to take for the cross-sectional area calculation.
2000 seems to be okay: 400 and 8000 samples agree to five s.f."""
AREA_NTHETA = 2000

"""Which cross-section to find the effective minor radius from. We set this 
to phi=0. Note that the cross-section location changes r_eff AND dPsi/dr 
simultaneously, which seem to mostly cancel."""
AREA_PHI_OVER_PI = 0.0
MAJOR_RADIUS = 5.7

PLOT_SURFACES = True    # plot the Boozer surfaces over the Poincare data
PLOT_IOTA = True        # plot |iota| against toroidal flux
PLOT_SHEAR = True       # plot d|iota|/dPsi against toroidal flux

# Which cross-sections to plot. Units are phi/pi so (0.0, 0.1, 0.2, 0.3) is a four panel plot of the first field period.
PHIS_OVER_PI = (0.0, 0.1, 0.2, 0.3)

# Select the Poincare data to load from poincare_fieldlines.py
POINCARE_N = 200           # no. of field lines in the dataset to load
POINCARE_TMAX = 4000      # tmax of the dataset to load
POINCARE_START_PHI_OVER_PI = 0.2

ISLAND_BAND = (1.86, 2.35)   # shaded on the plots. Should be determined from the qfm full set.


def g_guess(field):
    """Calculate the Boozer G, mu0 x the total unsigned coil current.

    It is important to take the abs() since coil currents are often negative in
    our convention. We use this G to check the output of the Boozer solve; if
    the Gs don't agree then the solver has failed.
    """
    return MU0 * sum(abs(coil.current.get_value()) for coil in field.coils)


def convert_surface(qfm, mpol):
    """Take the qfm surface (a SurfaceRZFourier object) and convert it to a SurfaceXYZTensorFourier object
    so that the Boozer solve can read it.

    Perform a least squares fit to match the geometry.
    """
    phis = np.linspace(0, 1/NFP, 2*mpol + 1, endpoint=False)
    thetas = np.linspace(0, 1, 2*mpol + 1, endpoint=False)

    source = SurfaceRZFourier(
        mpol=qfm.mpol, ntor=qfm.ntor, stellsym=qfm.stellsym, nfp=NFP,
        quadpoints_phi=phis, quadpoints_theta=thetas)
    source.x = qfm.x

    surface = SurfaceXYZTensorFourier(
        mpol=mpol, ntor=mpol, stellsym=qfm.stellsym, nfp=NFP,
        quadpoints_phi=phis, quadpoints_theta=thetas)
    surface.least_squares_fit(source.gamma())
    return surface


def convert_and_solve(field, qfm, iota_guess=IOTA_GUESS, headroom=HEADROOM):
    """Convert one QFM surface to a Boozer surface and extract its surface parameters.

    Returns (surface, flux, iota, G, residual). If no surface converged then set all to None, other than flux.
    Try to solve at various iota guesses if the first doesn't converge.
    """
    flux = toroidal_flux(field, qfm)
    reference = g_guess(field)

    for offset in IOTA_OFFSETS:
        surface = convert_surface(qfm, qfm.mpol + headroom)
        boozer = BoozerSurface(
            BiotSavart(field.coils), surface,
            ToroidalFlux(surface, BiotSavart(field.coils)), flux,
            options={"verbose": False, "newton_tol": NEWTON_TOL})

        try:
            result = boozer.run_code(iota_guess + offset)
        except Exception:
            continue

        if abs(result["G"] - reference) < G_TOLERANCE * reference:
            return (surface, flux, float(result["iota"]), float(result["G"]),
                    float(np.linalg.norm(result["residual"], np.inf)))

    return None, flux, None, None, None


def boozer_section_rz(surface, phi_over_pi=0.0, ntheta=PLOT_NTHETA, nsample=400):
    """Determine the (r,z) coordinates of a Boozer surface at a given toroidal
    cross-section.

    This is entirely analogous to in previous scripts cross_section_rz however
    the complication is that the Boozer surfaces are parametrised by Boozer
    and so picking out a particular geometric toroidal cross-section is not
    trivial.

    To solve this, for each poloidal angle theta, we take a sample of phi_boozer
    points on the surface (around the phi_geometric angle). We calculate each point's
    geometric coordinates xyz, and we estimate (interpolate) the Boozer coordinates that correspond
    to a given toroidal angle.

    A telltale sign of this function failing is up-down asymmetry when you plot a
    Boozer surface's geometric cross-section.
    """


    half = 0.5 / NFP                  # half a field period either side
    band = SurfaceXYZTensorFourier(
        mpol=surface.mpol, ntor=surface.ntor, stellsym=surface.stellsym,
        nfp=NFP,
        quadpoints_phi=np.linspace(phi_over_pi/2 - half, phi_over_pi/2 + half,
                                   nsample, endpoint=False),
        quadpoints_theta=np.linspace(0, 1, ntheta, endpoint=False))
    band.x = surface.x

    xyz = band.gamma()
    r = np.sqrt(xyz[..., 0]**2 + xyz[..., 1]**2)
    z = xyz[..., 2]

    # unwrap removes angle jumps
    geometric = np.unwrap(np.arctan2(xyz[..., 1], xyz[..., 0]), axis=0)

    target = phi_over_pi * np.pi
    r_at = np.array([np.interp(target, geometric[:, j], r[:, j])
                     for j in range(ntheta)])
    z_at = np.array([np.interp(target, geometric[:, j], z[:, j])
                     for j in range(ntheta)])

    return np.append(r_at, r_at[0]), np.append(z_at, z_at[0])


def area_equivalent_radius(qfm, phi_over_pi=AREA_PHI_OVER_PI,
                           ntheta=AREA_NTHETA):
    """Finds the effective minor radius of a QFM surface. Also sets the
    average major radius.

    We load the surface's cross-sectional (r,z) coordinates. Then we apply
    the shoelace formula to find the cross-sectional area.

    A = 1/2 | sum_i ( r_i z_{i+1} - r_{i+1} z_i ) |

    This is exact for the polygon with corners at the sampled points.

    Then we estimate the minor radius using:

    r = sqrt(A/pi)

    the radius of the circle with the same area as the cross-section.

    We set the major radius R and return (r,R).
    """
    r, z = cross_section_rz(qfm, phi_over_pi, ntheta)
    r, z = r[:-1], z[:-1]

    area = 0.5 * np.abs(np.sum(r*np.roll(z, -1) - np.roll(r, -1)*z))
    return np.sqrt(area/np.pi), MAJOR_RADIUS


def bracket_for(flux, resonant=RESONANT_FLUX):
    """Chooses the nearest Boozer solved fluxes above and below the resonance.
    If results do not surround resonance, choose the first and last surfaces.

    This doesn't automatically exclude the island so be careful that you don't
    input "Boozer solved surfaces" over the island band that are actually cusped.
    """
    below = np.nonzero(flux < resonant)[0]
    above = np.nonzero(flux > resonant)[0]
    if len(below) and len(above):
        return int(below[-1]), int(above[0]), True
    return 0, len(flux) - 1, False


def shear_at_resonance(flux, iota, radius):
    """Using the two surfaces at the edges of the island band, estimate the
     magnetic shear at the resonance.

     Calculates:
    d|iota|/dPsi,
    dPsi/dr,
    and d|iota|/dr = d|iota|/dPsi x dPsi/dr

    Returns a dict with the low and high fluxes, straddles tells us if the two surfaces are
    surrounding the resonance or not, along with the above values.
    """
    low, high, straddles = bracket_for(flux)

    d_psi = flux[high] - flux[low]
    diota_dpsi = (iota[high] - iota[low]) / d_psi
    dpsi_dr = d_psi / (radius[high] - radius[low])

    return {"low": flux[low], "high": flux[high], "straddles": straddles,
            "diota_dpsi": diota_dpsi, "dpsi_dr": dpsi_dr,
            "iota_prime": diota_dpsi * dpsi_dr}


def boozer_path(flux, config=CONFIG, nphi=SOURCE_NPHI, ntheta=SOURCE_NTHETA,
                mpol=SOURCE_MPOL, headroom=HEADROOM):
    """Determine the folder and the filename for each Boozer surface.

    The name details the QFM surface it came from plus the headroom (should be set to 2).
    """
    return (Path(f"{BOOZER_DIR} {config}") /
            f"boozer_surface_flux_{flux:.4f}"
            f"_nphi{nphi}_ntheta{ntheta}_mpol{mpol}_headroom{headroom}.json")


def profile_path(config=CONFIG, nphi=SOURCE_NPHI, ntheta=SOURCE_NTHETA,
                 mpol=SOURCE_MPOL, headroom=HEADROOM):
    """Determine where to save the iota profile and G."""
    return (Path(f"{BOOZER_DIR} {config}") /
            f"boozer_profile_nphi{nphi}_ntheta{ntheta}"
            f"_mpol{mpol}_headroom{headroom}.npz")


def save_boozer_surface(surface, flux, config=CONFIG):
    """Save the Boozer surface to its corresponding folder with its name."""
    path = boozer_path(flux, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    surface.save(str(path))
    print(f"saved surface to {path}")
    return path


def plot_profile(flux, values, ylabel, title, resonance=False, filename=None):
    """Plot a given quantity against toroidal flux, with the island band shaded.

    resonance=True adds the |iota| = 1 line, which is only meaningful for iota.
    """
    figure, ax = plt.subplots(figsize=(9, 6))

    ax.plot(flux, values, "o-", lw=2, ms=5)
    if resonance:
        ax.axhline(1.0, color="0.5", ls="--", lw=1.4, label=r"$|\iota| = 1$")
    ax.axvspan(*ISLAND_BAND, color="#c0392b", alpha=0.12,
               label="5/5 island band")

    ax.set_xlabel(r"toroidal flux  $\Psi_T$  [Wb]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, lw=0.5, alpha=0.6)
    ax.legend()

    figure.tight_layout()
    if filename:
        figure.savefig(filename, dpi=150)
        print(f"saved plot to {filename}")
    plt.show()


def solve_family(field):
    """Make Boozer surfaces from a set of QFM surfaces while constraining the toroidal flux.
    Also make the iota profile data."""
    print(f"converting {len(FLUXES)} surfaces at mpol={SOURCE_MPOL} "
          f"({SOURCE_NPHI}x{SOURCE_NTHETA}) to Boozer at "
          f"mpol={SOURCE_MPOL + HEADROOM}\n")
    print(f"{'flux':>6} {'iota':>11} {'G':>9} {'residual':>11} "
          f"{'r [m]':>8} {'time':>7}")

    rows = []
    surfaces = []
    guess = IOTA_GUESS

    for target in FLUXES:
        source = surface_path(target, CONFIG, SOURCE_NPHI, SOURCE_NTHETA,
                              SOURCE_MPOL, SURFACE_DIR)
        if not source.exists():
            print(f"{target:6.2f}   no QFM surface at {source}")
            continue

        start = time.perf_counter()
        qfm = load(str(source))
        surface, flux, iota, G, residual = convert_and_solve(field, qfm, guess)

        if surface is None:
            print(f"{flux:6.2f}   no convergence from any guess, skipped")
            continue


        minor, major = area_equivalent_radius(qfm)

        print(f"{flux:6.2f} {iota:11.6f} {G:9.3f} {residual:11.3e} "
              f"{minor:8.4f} "
              f"{time.perf_counter() - start:6.1f}s")

        guess = iota
        rows.append((flux, iota, G, residual, minor, major))
        surfaces.append((flux, surface))
        if SAVE:
            save_boozer_surface(surface, flux)

    if not rows:
        raise SystemExit("nothing converged")

    flux, iota, G, residual, minor, major = np.array(sorted(rows)).T
    profile = dict(flux=flux, iota=iota, G=G, residual=residual,
                   minor_radius=minor, major_radius=major)
    print(f"\n{len(rows)} of {len(FLUXES)} surfaces converted")
    return profile, surfaces


def main():

    """Chain everything together.

    So:
    - Convert each QFM surface to Boozer coordinates
    - Report iota, iota' and G
    - Save it """
    from poincare_fieldlines import load_poincare_data

    field, axis = build_field(CONFIG)

    profile, surfaces = solve_family(field)
    flux = profile["flux"]
    iota = np.abs(profile["iota"])
    radius = profile["minor_radius"]


    shear = None
    if len(flux) > 1:
        shear = shear_at_resonance(flux, iota, radius)

        print(f"\nmagnetic shear at Psi_T = {RESONANT_FLUX:.2f} Wb, "
              f"calculated over {shear['low']:.2f}-{shear['high']:.2f} Wb:")
        if not shear["straddles"]:
            print(f"  WARNING: those surfaces do NOT straddle the resonance, "
                  f"so this is\n           not the background shear across "
                  f"the island")
        print(f"  d|iota|/dPsi     = {shear['diota_dpsi']:.4f} /Wb")
        print(f"  dPsi/dr          = {shear['dpsi_dr']:.4f} Wb/m"
              f"   (area-equivalent r at phi = "
              f"{AREA_PHI_OVER_PI:.2f}pi)")
        print(f"  iota' = d|iota|/dr = {shear['iota_prime']:.4f} /m")
        print(f"  R                = {MAJOR_RADIUS:.4f} m")
        print(f"  -> iota' and R are the two profile inputs Feng's width needs")

    if SAVE:
        path = profile_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        profile["resonant_flux"] = RESONANT_FLUX
        if shear is not None:
            profile.update(shear)
        np.savez(path, **profile)
        print(f"saved profile to {path}")

    if PLOT_SURFACES:
        r, z, line, panel = load_poincare_data(
            CONFIG, POINCARE_N, POINCARE_TMAX,
            POINCARE_START_PHI_OVER_PI
        )
        plot_all(surfaces, r, z, panel, axis, phis_over_pi=PHIS_OVER_PI,
                 section=boozer_section_rz,
                 filename=f"{CONFIG}_boozer_surfaces.png" if SAVE else None,
                 title=f"Boozer surfaces on Poincare sections, {CONFIG}")

    if PLOT_IOTA:
        plot_profile(flux, iota, r"$|\iota|$",
                     f"Rotational transform profile, {CONFIG}", resonance=True,
                     filename=f"{CONFIG}_iota_profile.png" if SAVE else None)

    if PLOT_SHEAR and len(flux) > 1:
        plot_profile(flux, np.gradient(iota, flux),
                     r"$d|\iota| / d\Psi_T$  [1/Wb]", f"Magnetic shear, {CONFIG}",
                     filename=f"{CONFIG}_magnetic_shear.png" if SAVE else None)


if __name__ == "__main__":
    main()
