import sys
from scipy.sparse.linalg import spsolve
import numpy as np
import matplotlib.pyplot as plt
import argparse


def print_array(array, name):
    for i, v in enumerate(array):
        print(f"{name}[{i}] = {v:10.5f}")


def print_mat(mat):
    nrow, ncols = mat.shape
    for i in range(nrow):
        for j in range(ncols):
            print(f"{mat[i, j]:6.3f}", end="")
        print()


sys.path.append("..")
import pyfem, utils

p = argparse.ArgumentParser()
p.add_argument("--nx", type=int, default=1)
p.add_argument("--ny", type=int, default=1)
p.add_argument("--nz", type=int, default=1)
p.add_argument("--lx", type=float, default=1.0)
p.add_argument("--ly", type=float, default=1.0)
p.add_argument("--lz", type=float, default=1.0)
p.add_argument("--bcl_temp", type=float, default=1.0)
# p.add_argument("--bcr_temp", type=float, default=0.0)
p.add_argument("--source_temp", type=float, default=0.0)
p.add_argument("--print_mat", action="store_true")

args = p.parse_args()

nelems_x = args.nx
nelems_y = args.ny
nelems_z = args.nz
Lx = args.lx
Ly = args.ly
Lz = args.lz

ramp = 5.0

creator = pyfem.ProblemCreator(
    nnodes_x=nelems_x + 1,
    nnodes_y=nelems_y + 1,
    nnodes_z=nelems_z + 1,
    Lx=Lx,
    Ly=Ly,
    Lz=Lz,
    element_type="block",
)

# Set fixed dof
dof_fixed_left = []
dof_fixed_right = []
for k in range(nelems_z + 1):
    for j in range(nelems_y + 1):
        dof_fixed_left.append(creator.nodes3d[k, j, 0])  # fixed left edge
        dof_fixed_right.append(creator.nodes3d[k, j, nelems_x])  # fixed right edge

# Set mesh
conn, X = creator.conn, creator.X
nnodes = X.shape[0]
nelems = conn.shape[0]

# Set model
quadrature = pyfem.QuadratureBlock3D()
basis = pyfem.BasisBlock3D(quadrature)
model = pyfem.LinearPoisson(
    X=X,
    conn=conn,
    dof_fixed=dof_fixed_left,
    dof_fixed_vals=[args.bcl_temp] * len(dof_fixed_left),
    quadrature=quadrature,
    basis=basis,
    gfunc=lambda x: args.source_temp,
    p=ramp,
)

# Construct linear system, apply bc and solve
K = model.compute_jacobian()
if args.print_mat:
    print("no bc:")
    print_mat(K)
rhs = model.compute_rhs()
K, rhs = model.apply_dirichlet_bcs(K, rhs, enforce_symmetric_K=False)
if args.print_mat:
    print("with bc:")
    print_mat(K)
u = spsolve(K, rhs)

# Save solution to vtk
sol = {"u": u}
utils.to_vtk(conn, X, sol)

# Set design variable and evaluate gradient
x = np.ones(nnodes)
comp, _ = model.compliance(x)
grad = model.compliance_grad(x, u, weighted=False)

# Print solutions
print("Solution")
print_array(u, "sol")

# Print gradient
print_array(grad, "g")
