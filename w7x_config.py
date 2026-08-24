"""W7-X configurations that subsequent scripts will load.
First 5 entries are non-planar coil currents.
Last 2 entries are planar coil currents.

Currents are given as ratios and are all multiplied by the current scale to match W7-X.

Note that the negative current scale ensures that B_phi, toroidal flux and the G value in the Boozer solves are all positive.
"""

import numpy as np

from simsopt.configs import get_w7x_data
from simsopt.field import BiotSavart, Current, coils_via_symmetries

NFP = 5
CURRENT_SCALE = -12985 / 15000

# five non-planar coils, then two planar coils.
CONFIGURATIONS = {
    "standard": np.asarray([1, 1, 1, 1, 1, 0, 0]),
    "high_iota": np.asarray([1, 1, 1, 1, 1, -0.23, -0.23]),
    "bigger_islands": np.asarray([1, 1, 1, 1, 0.4, 0, 0]),
    "CYM": np.asarray([1.01, 1.05, 1.05, 1.01, 1.16, -0.33, -0.33]),
}


def build_field(config="standard"):
    """Build our field for a given configuration
    """
    ratios = CONFIGURATIONS[config]

    curves, default_currents, axis = get_w7x_data()
    nominal = default_currents[0].get_value()
    currents = [Current(float(value))
                for value in nominal * CURRENT_SCALE * ratios]
    coils = coils_via_symmetries(curves, currents, NFP, True)

    return BiotSavart(coils), axis
