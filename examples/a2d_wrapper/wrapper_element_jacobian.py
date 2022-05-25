import sys
import numpy as np

sys.path.append("../..")
import pyfem

sys.path.append("../../../a2d/examples/elasticity")
import model as a2dmodel

# Set up mesh
creator = pyfem.ProblemCreator(8, 4, 4, element_type="block")
nodes, conn, X, dof_fixed, nodal_force = creator.create_linear_elasticity_problem()

# Compute element-wise Jacobian with a2d
nelems = conn.shape[0]
nnodes_per_elem = conn.shape[1]
ndof_per_node = 3
nquads = 8
nnodes = X.shape[0]
jac = np.zeros((nelems, nnodes_per_elem, nnodes_per_elem, ndof_per_node, ndof_per_node))

E = 10.0
nu = 0.3
mu = E / (2 * (1 + nu))
lam = E * nu / (1 + nu) / (1 - 2 * nu)
print(mu)
print(lam)
data = np.zeros((nelems, nquads, 2))  # Constitutive
data[:, :, 0] = mu  # Lame parameter: mu
data[:, :, 1] = lam  # Lame parameter: lambda

U = np.zeros((nnodes, ndof_per_node))
conn = conn.astype(np.intc)
a2dmodel.compute_jac(conn, X, data, U, jac)

# Compute element-wise Jacobian using pyfem
quadrature = pyfem.QuadratureBlock3D()
basis = pyfem.BasisBlock3D(quadrature)
model = pyfem.LinearElasticity(
    nodes, X, conn, dof_fixed, None, nodal_force, quadrature, basis, E=E, nu=nu
)
K = model.compute_jacobian()
model._jacobian_mat_to_tensor(model.Ke_mat, model.Ke_tensor)

# Cross-check
print(jac / model.Ke_tensor)
print(jac.shape)
print(model.Ke_tensor.shape)
