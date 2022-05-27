"""
This script shows how to solve the Helmholtz problem on predefined solution field
x and visualize the filtered field
"""
import sys

sys.path.append("../..")
import pyfem
import matplotlib.pyplot as plt


def solve_helmholtz_problem(ax, element_type="quad"):
    creator = pyfem.ProblemCreator(nnodes_x=96, nnodes_y=96, element_type=element_type)
    nodes, conn, X, x = creator.create_helmhotz_problem()

    if element_type == "quad":
        quadrature = pyfem.QuadratureBilinear2D()
        basis = pyfem.BasisBilinear2D(quadrature)
    elif element_type == "tri":
        quadrature = pyfem.QuadratureTriangle2D()
        basis = pyfem.BasisTriangle2D(quadrature)

    model = pyfem.Helmholtz(0.05, nodes, X, conn, quadrature, basis)
    assembler = pyfem.Assembler(model)
    u = model.apply(x)
    assembler.plot(u, ax)
    ax.set_title(f"element: {element_type}")

    print("Check if the integral of x is preserved:")
    print("sum x:", x.sum())
    print("sum u:", u.sum())
    return assembler, x


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(12, 4), nrows=1, ncols=3, constrained_layout=True)
    solve_helmholtz_problem(ax[1], "quad")
    assembler, x = solve_helmholtz_problem(ax[2], "tri")
    assembler.plot(x, ax[0])
    ax[0].set_title("raw field")

    plt.show()
