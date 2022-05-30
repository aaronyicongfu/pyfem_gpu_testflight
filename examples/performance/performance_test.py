import sys
import argparse

sys.path.append("../..")
import pyfem
import utils

p = argparse.ArgumentParser()
p.add_argument("-nx", type=int, default=64)
p.add_argument("-ny", type=int, default=64)
p.add_argument("--problem", default="poisson", choices=["poisson", "elasticity"])
p.add_argument("--method", default="gmres", choices=["gmres", "direct"])
args = p.parse_args()

# Switch on timer
utils.timer_on()

creator = pyfem.ProblemCreator(nnodes_x=args.nx, nnodes_y=args.ny)
quadrature = pyfem.QuadratureBilinear2D()
basis = pyfem.BasisBilinear2D(quadrature)

if args.problem == "poisson":
    conn, X, dof_fixed = creator.create_poisson_problem()
    model = pyfem.LinearPoisson2D(X, conn, dof_fixed, None, quadrature, basis)
else:
    conn, X, dof_fixed, nodal_force = creator.create_linear_elasticity_problem()
    model = pyfem.LinearElasticity(
        X, conn, dof_fixed, None, nodal_force, quadrature, basis
    )

assembler = pyfem.Assembler(model)
u = assembler.solve(method="gmres")

print("Summary")
print(f"ndof: {len(u)}")
