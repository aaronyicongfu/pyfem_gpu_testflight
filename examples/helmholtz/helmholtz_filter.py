import sys

sys.path.append("../..")
import pyfem
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import spsolve

creator = pyfem.ProblemCreator(nnodes_x=96, nnodes_y=96, element_type="quad")
nodes, conn, X, x = creator.create_helmhotz_problem()

quadrature = pyfem.QuadratureBilinear2D()
basis = pyfem.BasisBilinear2D(quadrature)

model = pyfem.Helmholtz(0.05, nodes, X, conn, quadrature, basis)
assembler = pyfem.Assembler(model)

u = model.apply(x)

fig, ax = plt.subplots()
assembler.plot(u, ax)
plt.show()
