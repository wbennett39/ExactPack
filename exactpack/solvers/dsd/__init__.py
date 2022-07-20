r"""Python based solvers for HE burn time for DSD verification problems.

This suite of solvers calculates HE burn times for DSD verification problems.

The DSD High Explosive Problem Set is a series of three problems
designed to test the burn table solution (HE light times) generated by DSD
level-set solvers for burn simulations. The Cylindrical Expansion test
problem has an exact burn time solution in 2D [Bdzil]_. The 2D Rate Stick
and Explosive Arc test problems can each be reduced to a quasi-linear PDE,
which can be solved using highly-accurate numerical methods. That solution
must then be inverted to obtain the burn time solution on the specified grid.

DSD theory uses boundary angles to describe how the HE combustion wave (burn
front) interacts with surrounding materials. The boundary angle,
:math:`\omega_c`, is defined as the angle between the normal to the HE boundary
and the normal to the burn front shock wave, as shown in the figure.

.. figure:: dsd_edgeangle.png
   :align: center
   :scale: 50%

   The boundary angle :math:`\omega_c` as defined by the normals to the
   burn front, :math:`\vec{n_s}`, and to the HE boundary, :math:`\vec{n_B}`.

Three angles are considered:

* The maximum possible angle is found in the case of a reflective boundary
  or an abutting rigid inert body. Here, the burn front is expected to be
  perpendicular to the boundary, resulting in a maximum edge angle of
  :math:`\omega = \frac{\pi}{2}`. This holds for any HE.
* The minimum angle, :math:`\omega_s`, is found in the case of HE product
  expansion into a vacuum. The value of this angle will depend on which HE
  material is being used. For all HEs, :math:`0 < \omega_s < \frac{\pi}{2}`.
* The edge angle, :math:`\omega_c`, between the HE and a deformable inert
  material depends on which HE material and which inert material are in use.
  For all combinations, :math:`\omega_s < \omega_c < \frac{\pi}{2}`.

Which, if any, of these angles need to be defined depends on the test problem
being evaluated.

.. [Bdzil] Bdzil, J. B., R. J. Henninger, and J. W. Walter, Test Problems
   for DSD2D, LA-14277, 2006.

"""

from .ratestick import RateStick
from .cylexpansion import CylindricalExpansion
from .explosivearc import ExplosiveArc
