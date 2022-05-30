import numpy as np
import sys

sys.path.append("../..")
import pyfem
import utils

sys.path.append("../../../a2d/examples/elasticity")
import example as a2d


# Switch on logger
utils.timer_on()

# Set up mesh
creator = pyfem.ProblemCreator(64, 32, 32, element_type="block")
conn, X, dof_fixed, nodal_force = creator.create_linear_elasticity_problem()

"""
Linear/Nonlinear elasticity problem
"""

# Compute element-wise Jacobian using a2d
problem_info = {"type": "elasticity", "E": 10, "nu": 0.3}
model_a2d = pyfem.A2DWrapper(X, conn, dof_fixed, None, a2d, problem_info)
K_a2d = model_a2d.compute_jacobian()

# Compute element-wise Jacobian using pyfem
quadrature = pyfem.QuadratureBlock3D()
basis = pyfem.BasisBlock3D(quadrature)
model = pyfem.LinearElasticity(X, conn, dof_fixed, None, nodal_force, quadrature, basis)
K = model.compute_jacobian()

# Check difference
diff = np.max(K - K_a2d) / np.max(K)
print(f"(K - K_a2d) / K = {diff:.10e}")

"""
Helmholtz problem
"""

# Compute element-wise Jacobian using a2d
r0 = 0.05
problem_info = {"type": "helmholtz", "r0": r0}
model_a2d = pyfem.A2DWrapper(X, conn, dof_fixed, None, a2d, problem_info)
K_a2d = model_a2d.compute_jacobian()

# Compute element-wise Jacobian using pyfem
quadrature = pyfem.QuadratureBlock3D()
basis = pyfem.BasisBlock3D(quadrature)
model = pyfem.Helmholtz(r0, X, conn, quadrature, basis)
K = model.compute_jacobian()

# Check difference
diff = np.max(K - K_a2d) / np.max(K)
print(f"(K - K_a2d) / K = {diff:.10e}")
