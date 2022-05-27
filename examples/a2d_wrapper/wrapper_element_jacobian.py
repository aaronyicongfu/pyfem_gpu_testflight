import sys
import numpy as np

sys.path.append("../..")
import pyfem
import utils

sys.path.append("../../../a2d/examples/elasticity")
import model as a2dmodel


@utils.time_this
def compute_jac_a2d():
    a2dmodel.compute_jac(conn, X, data, U, jac)
    return


@utils.time_this
def compute_helmholtz_jac_a2d():
    a2dmodel.compute_helmholtz_jac(conn, X, data, x, hz_jac)


# Switch on logger
# utils.timer_on()

# Set up mesh
creator = pyfem.ProblemCreator(16, 8, 8, element_type="block")
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
data = np.zeros((nelems, nquads, 2))  # Constitutive
data[:, :, 0] = mu  # Lame parameter: mu
data[:, :, 1] = lam  # Lame parameter: lambda

U = np.random.rand(nnodes, ndof_per_node)
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

# Check jacobian
diff = jac / model.Ke_tensor
print("Check Jacobian:")
print(diff.min())
print(diff.max())

# Compute filtered variable with a2d
ndof_per_node = 1
nodes, conn, X, x = creator.create_helmhotz_problem()
conn = conn.astype(np.intc)
hz_jac = np.zeros(
    (nelems, nnodes_per_elem, nnodes_per_elem, ndof_per_node, ndof_per_node)
)
r0 = 0.1
x = x[:, np.newaxis]
data[:, :, :] = r0
a2dmodel.compute_helmholtz_jac(conn, X, data, x, hz_jac)

# Compute filtered variable using pyfem
model = pyfem.Helmholtz(r0, nodes, X, conn, quadrature, basis)
K = model.compute_jacobian()
model._jacobian_mat_to_tensor(model.Ke_mat, model.Ke_tensor)

# Check Helmholtz filter
print("Check Helmholtz filter:")
diff = hz_jac / model.Ke_tensor
print(hz_jac.flatten()[0:50])
print(model.Ke_tensor.flatten()[0:50])
print(diff.min())
print(diff.max())
