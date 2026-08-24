"""
Poincare plot for the W7-X standard configuration.
Code borrowed from Chris Smiet.
"""

import time

import numpy as np

from simsopt.configs import get_w7x_data
from simsopt.field import (BiotSavart, InterpolatedField,
                           LevelsetStoppingCriterion, SurfaceClassifier,
                           coils_via_symmetries, compute_fieldlines,
                           plot_poincare_data)
from simsopt.geo import SurfaceRZFourier
from simsopt.util import comm_world, proc0_print


def make_vacuum_standard_plot():
    """Poincare plot for the standard config.
    Code borrowed from Chris Smiet."""

    nfieldlines = 30
    tmax_fl = 4000
    degree = 4

    nfp = 5
    curves, currents, ma = get_w7x_data()
    currents_mescan = np.array(currents)*-12985/15000
    coils = coils_via_symmetries(curves, currents_mescan, nfp, True)
    curves = [c.curve for c in coils]
    bs = BiotSavart(coils)

    mpol = 5
    ntor = 5
    stellsym = True

    s = SurfaceRZFourier.from_nphi_ntheta(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp,
                                          range="full torus", nphi=64, ntheta=24)
    s.fit_to_curve(ma, 1.60, flip_theta=False)

    sc_fieldline = SurfaceClassifier(s, h=0.03, p=2)

    def trace_fieldlines(bfield, label):
        t1 = time.time()
        R0 = np.linspace(ma.gamma()[0, 0], ma.gamma()[0, 0] + 0.44, nfieldlines)
        Z0 = [ma.gamma()[0, 2] for i in range(nfieldlines)]
        phis = [(i/4)*(2*np.pi/nfp) for i in range(4)]
        fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
            bfield, R0, Z0, tmax=tmax_fl, tol=1e-9, comm=comm_world,
            phis=phis, stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)])
        t2 = time.time()
        proc0_print(f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}", flush=True)
        return fieldlines_phi_hits, phis

    # Bounds for the interpolated magnetic field chosen so that the surface is
    # entirely contained in it
    n = 25
    rs = np.linalg.norm(s.gamma()[:, :, 0:2], axis=2)
    zs = s.gamma()[:, :, 2]
    rrange = (np.min(rs), np.max(rs), n)
    phirange = (0, 2*np.pi/nfp, n*2)
    # exploit stellarator symmetry and only consider positive z values:
    zrange = (0, np.max(zs), n//2)

    def skip(rs, phis, zs):
        # The RegularGridInterpolant3D class allows us to specify a function that
        # is used in order to figure out which cells to be skipped.  Internally,
        # the class will evaluate this function on the nodes of the regular mesh,
        # and if all of the eight corners are outside the domain, then the cell
        # is skipped.  Since the surface may be curved in a way that for some
        # cells, all mesh nodes are outside the surface, but the surface still
        # intersects with a cell, we need to have a bit of buffer in the signed
        # distance (essentially blowing up the surface a bit), to avoid ignoring
        # cells that shouldn't be ignored
        rphiz = np.asarray([rs, phis, zs]).T.copy()
        dists = sc_fieldline.evaluate_rphiz(rphiz)
        skip = list((dists < -0.05).flatten())
        proc0_print("Skip", sum(skip), "cells out of", len(skip), flush=True)
        return skip

    proc0_print('Initializing InterpolatedField')
    bsh = InterpolatedField(
        bs, degree, rrange, phirange, zrange, nfp=nfp, stellsym=True, skip=skip, extrapolate=False
    )
    proc0_print('Done initializing InterpolatedField')

    proc0_print('Beginning field line tracing')
    fieldlines_phi_hits, phis = trace_fieldlines(bsh, 'bsh')

    plot_poincare_data(fieldlines_phi_hits, phis, 'poincare_fieldline.png', dpi=150)

    return fieldlines_phi_hits, phis


if __name__ == "__main__":
    make_vacuum_standard_plot()
