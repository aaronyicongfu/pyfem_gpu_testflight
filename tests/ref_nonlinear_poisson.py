import numpy as np
from scipy import sparse
from scipy import special
from scipy.sparse.linalg import spsolve
import matplotlib.pylab as plt
import matplotlib.tri as tri
from icecream import ic

class PoissonProblem:
    def __init__(self, N):
        self.N = min(10, max(N, 2))

    def gfunc(self, xvals, yvals):
        """
        Given the x and y locations return the right-hand-side
        """
        g = 1e4 * xvals*(1.0 - xvals)*(1.0 - 2.0*xvals)*yvals*(1.0 - yvals)*(1.0 - 2.0*yvals)

        return g

    def hfunc(self, xdv, xvals, yvals):
        """
        Given the design variables and the x and y locations return h
        """
        h = np.ones(xvals.shape, dtype=xdv.dtype)
        for k in range(self.N):
            coef = special.binom(self.N - 1, k)
            xarg = coef * (1.0 - xvals)**(self.N - 1 - k) * xvals**k
            yarg = 4.0 * yvals * (1.0 - yvals)
            h += xdv[k] * xarg * yarg

        return h

class NonlinearPoisson:
    def __init__(self, conn, x, bcs, problem):
        """
        Initialize the nonlinear Poisson problem

        - grad . (h(x)(1.0 + u^2) grad(u))) = g

        Args:
            conn: The connectivity
            x: The node locations
            bcs: The boundary conditions
            problem: The Poisson problem instance
        """

        self.conn = np.array(conn)
        self.x = np.array(x)
        self.problem = problem
        self.u_save = None

        self.nelems = self.conn.shape[0]
        self.nnodes = int(np.max(self.conn)) + 1
        self.nvars = self.nnodes

        self.reduced = self._compute_reduced_variables(self.nvars, bcs)

        # Set up arrays for assembling the matrix
        i = []
        j = []
        for index in range(self.nelems):
            for ii in self.conn[index, :]:
                for jj in self.conn[index, :]:
                    i.append(ii)
                    j.append(jj)

        # Convert the lists into numpy arrays
        self.i = np.array(i, dtype=int)
        self.j = np.array(j, dtype=int)

    def _compute_reduced_variables(self, nvars, bcs):
        """
        Compute the reduced set of variables
        """
        reduced = list(range(nvars))
        for node in bcs:
            reduced.remove(node)

        return reduced

    def _eval_basis_and_jacobian(self, xi, eta, xe, ye, J, detJ, invJ=None):
        """
        Evaluate the basis functions and Jacobian of the transformation
        """

        N = 0.25*np.array([(1.0 - xi)*(1.0 - eta),
                            (1.0 + xi)*(1.0 - eta),
                            (1.0 + xi)*(1.0 + eta),
                            (1.0 - xi)*(1.0 + eta)])
        Nxi = 0.25*np.array([-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)])
        Neta = 0.25*np.array([-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)])

        # Compute the Jacobian transformation at each quadrature points
        J[:, 0, 0] = np.dot(xe, Nxi)
        J[:, 1, 0] = np.dot(ye, Nxi)
        J[:, 0, 1] = np.dot(xe, Neta)
        J[:, 1, 1] = np.dot(ye, Neta)

        # Compute the inverse of the Jacobian
        detJ[:] = J[:, 0, 0]*J[:, 1, 1] - J[:, 0, 1]*J[:, 1, 0]

        if invJ is not None:
            invJ[:, 0, 0] = J[:, 1, 1]/detJ
            invJ[:, 0, 1] = -J[:, 0, 1]/detJ
            invJ[:, 1, 0] = -J[:, 1, 0]/detJ
            invJ[:, 1, 1] = J[:, 0, 0]/detJ

        return N, Nxi, Neta

    def assemble_residual(self, xdv, u):
        """
        Assemble the residuals

        Args:
            xdv: The design variable values
            u: The state variable values

        Returns:
            res: The residuals of the governing equations
        """

        # Check what type to use
        dtype = float
        if np.iscomplexobj(xdv) or np.iscomplexobj(u):
            dtype = complex

        # Gauss points
        gauss_pts = [-1.0/np.sqrt(3.0), 1.0/np.sqrt(3.0)]

        # The residuals for each finite-element
        Re = np.zeros((self.nelems, 4), dtype=dtype)

        # Data for each finite-element
        Be = np.zeros((self.nelems, 2, 4))

        detJ = np.zeros(self.nelems)
        J = np.zeros((self.nelems, 2, 2))
        invJ = np.zeros(J.shape)

        # Compute the x and y coordinates of each element
        xe = self.x[self.conn, 0]
        ye = self.x[self.conn, 1]

        # Set the state variable for all the elements
        ue = u[self.conn]

        for j in range(2):
            for i in range(2):
                xi = gauss_pts[i]
                eta = gauss_pts[j]
                N, Nxi, Neta = self._eval_basis_and_jacobian(xi, eta, xe, ye, J, detJ, invJ)

                # Compute the derivative of the shape functions w.r.t. xi and eta
                # [Nx, Ny] = [Nxi, Neta]*invJ
                Nx = np.outer(invJ[:, 0, 0], Nxi) + np.outer(invJ[:, 1, 0], Neta)
                Ny = np.outer(invJ[:, 0, 1], Nxi) + np.outer(invJ[:, 1, 1], Neta)

                # Set the B matrix for each element
                Be[:, 0, :] = Nx
                Be[:, 1, :] = Ny

                # Compute the u values at all the element locations
                uvals = np.dot(ue, N)

                # Evaluate the function
                xvals = np.dot(xe, N)
                yvals = np.dot(ye, N)
                g = self.problem.gfunc(xvals, yvals)
                h = self.problem.hfunc(xdv, xvals, yvals)

                # Add the contribution to the element residuals
                Re += np.einsum('n,nij,nil,nl -> nj', detJ * h * (1.0 + uvals**2), Be, Be, ue)
                Re -= np.outer(detJ * g, N)

        # Assemble the residuals
        res = np.zeros(self.nvars, dtype=u.dtype)
        for i in range(4):
            np.add.at(res, self.conn[:, i], Re[:, i])

        return res

    def assemble_jacobian(self, xdv, u):
        """
        Assemble the residual and Jacobian matrix

        Args:
            xdv: The design variable values
            u: The state variable values

        Returns:
            The Jacobian matrix
        """

        # Check what type to use
        dtype = float
        if np.iscomplexobj(xdv) or np.iscomplexobj(u):
            dtype = complex

        # Gauss points
        gauss_pts = [-1.0/np.sqrt(3.0), 1.0/np.sqrt(3.0)]

        # The residuals for each finite-element
        Re = np.zeros((self.nelems, 4), dtype=dtype)
        Ke = np.zeros((self.nelems, 4, 4), dtype=dtype)

        # Data for each finite-element
        Be = np.zeros((self.nelems, 2, 4))

        detJ = np.zeros(self.nelems)
        J = np.zeros((self.nelems, 2, 2))
        invJ = np.zeros(J.shape)

        # Compute the x and y coordinates of each element
        xe = self.x[self.conn, 0]
        ye = self.x[self.conn, 1]

        # Set the state variable for all the elements
        ue = u[self.conn]
        for j in range(2):
            for i in range(2):
                xi = gauss_pts[i]
                eta = gauss_pts[j]
                N, Nxi, Neta = self._eval_basis_and_jacobian(xi, eta, xe, ye, J, detJ, invJ)

                # Compute the derivative of the shape functions w.r.t. xi and eta
                # [Nx, Ny] = [Nxi, Neta]*invJ
                Nx = np.outer(invJ[:, 0, 0], Nxi) + np.outer(invJ[:, 1, 0], Neta)
                Ny = np.outer(invJ[:, 0, 1], Nxi) + np.outer(invJ[:, 1, 1], Neta)

                # Set the B matrix for each element
                Be[:, 0, :] = Nx
                Be[:, 1, :] = Ny

                # Compute the u values at all the element locations
                uvals = np.dot(ue, N)

                # Evaluate the function
                xvals = np.dot(xe, N)
                yvals = np.dot(ye, N)
                g = self.problem.gfunc(xvals, yvals)
                h = self.problem.hfunc(xdv, xvals, yvals)

                Re += np.einsum('n,nij,nil,nl -> nj', detJ * h * (1.0 + uvals**2), Be, Be, ue)
                Re -= np.outer(detJ * g, N)

                Ke += np.einsum('n,nij,nil -> njl', detJ * h * (1.0 + uvals**2), Be, Be)
                Ke += np.einsum('n,nij,nil,nl,k -> njk', 2.0 * detJ * h * uvals, Be, Be, ue, N)

        # Assemble the residuals
        res = np.zeros(self.nvars, dtype=dtype)
        for i in range(4):
            np.add.at(res, self.conn[:, i], Re[:, i])

        # Assemble the Jacobian matrix
        K = sparse.coo_matrix((Ke.flatten(), (self.i, self.j)))
        K = K.tocsr()

        return res, K

    def eval_ks(self, pval, u):

        # Compute the offset
        offset = np.max(u)

        # Compute the element stiffness matrix
        gauss_pts = [-1.0/np.sqrt(3.0), 1.0/np.sqrt(3.0)]

        # Assemble all of the the 4 x 4 element stiffness matrix
        Ke = np.zeros((self.nelems, 4, 4))
        Be = np.zeros((self.nelems, 2, 4))

        J = np.zeros((self.nelems, 2, 2))
        detJ = np.zeros(self.nelems)

        # Compute the x and y coordinates of each element
        xe = self.x[self.conn, 0]
        ye = self.x[self.conn, 1]

        # Compute the values of u for each element
        ue = u[self.conn]

        expsum = 0.0
        for j in range(2):
            for i in range(2):
                xi = gauss_pts[i]
                eta = gauss_pts[j]
                N, Nxi, Neta = self._eval_basis_and_jacobian(xi, eta, xe, ye, J, detJ)

                # Compute the values at the nodes
                uvals = np.dot(ue, N)

                expsum += np.sum(detJ * np.exp(pval*(uvals - offset)))

        return offset + np.log(expsum)/pval

    def eval_ks_adjoint_rhs(self, pval, u):

        # Compute the offset
        offset = np.max(u)

        # Compute the element stiffness matrix
        gauss_pts = [-1.0/np.sqrt(3.0), 1.0/np.sqrt(3.0)]

        # Information about the element transformation
        detJ = np.zeros(self.nelems)
        J = np.zeros((self.nelems, 2, 2))

        # Compute the x and y coordinates of each element
        xe = self.x[self.conn, 0]
        ye = self.x[self.conn, 1]

        # Compute the values of u for each element
        ue = u[self.conn]

        expsum = 0.0
        for j in range(2):
            for i in range(2):
                xi = gauss_pts[i]
                eta = gauss_pts[j]
                N, Nxi, Neta = self._eval_basis_and_jacobian(xi, eta, xe, ye, J, detJ)

                # Compute the values at the nodes
                uvals = np.dot(ue, N)

                expsum += np.sum(detJ * np.exp(pval*(uvals - offset)))

        # Store the element-wise right-hand-side
        erhs = np.zeros(self.conn.shape)

        for j in range(2):
            for i in range(2):
                xi = gauss_pts[i]
                eta = gauss_pts[j]
                N, Nxi, Neta = self._eval_basis_and_jacobian(xi, eta, xe, ye, J, detJ)

                # Compute the values at the nodes
                uvals = np.dot(ue, N)

                erhs += np.outer(detJ * np.exp(pval*(uvals - offset))/expsum, N)

        # Convert to the right-hand-side
        rhs = np.zeros(self.nnodes)
        for i in range(4):
            np.add.at(rhs, self.conn[:, i], erhs[:, i])

        return rhs

    def reduce_vector(self, forces):
        """
        Eliminate essential boundary conditions from the vector
        """
        return forces[self.reduced]

    def reduce_matrix(self, matrix):
        """
        Eliminate essential boundary conditions from the matrix
        """
        temp = matrix[self.reduced, :]
        return temp[:, self.reduced]

    def solve(self, xdv, u0=None, tol=1e-10, atol=1e-12, max_iter=10):
        """
        Perform a linear static analysis
        """

        dtype = float
        if np.iscomplexobj(xdv):
            dtype = complex
        elif u0 is not None and np.iscomplexobj(u0):
            dtype = complex

        if u0 is None:
            # Set the initial guess as u = 0
            u = np.zeros(self.nvars, dtype=dtype)
        else:
            u = u0

        res_norm_init = 0.0
        print("ref", '{0:5s} {1:25s}'.format('Iter', 'Norm'))

        for k in range(max_iter):
            res, K = self.assemble_jacobian(xdv, u)
            resr = self.reduce_vector(res)

            res_norm = np.sqrt(np.dot(resr, resr))
            print('{0:5d} {1:25.15e}'.format(k, res_norm))

            if k == 0:
                res_norm_init = res_norm
            elif res_norm < tol * res_norm_init or res_norm < atol:
                break

            Kr = self.reduce_matrix(K)
            updater = sparse.linalg.spsolve(Kr, resr)

            update = np.zeros(self.nvars, dtype=dtype)
            update[self.reduced] = updater
            u -= update

        return u

    def solve_adjoint(self, xdv, u, pval=10.0):
        """
        Compute the adjoint
        """

        res, K = self.assemble_jacobian(xdv, u)

        Kr = self.reduce_matrix(K)

        dfdu = self.eval_ks_adjoint_rhs(pval, u)
        dfdur = self.reduce_vector(dfdu)

        KrT = Kr.T
        psir = sparse.linalg.spsolve(KrT, -dfdur)

        psi = np.zeros(self.nvars)
        psi[self.reduced] = psir

        return psi

    def plot(self, u=None, ax=None, **kwargs):
        """
        Create a plot
        """

        # Plot the saved copy of u
        if u is None and self.u_save is not None:
            u = self.u_save

        # Create the triangles
        triangles = np.zeros((2*self.nelems, 3), dtype=int)
        triangles[:self.nelems, 0] = self.conn[:, 0]
        triangles[:self.nelems, 1] = self.conn[:, 1]
        triangles[:self.nelems, 2] = self.conn[:, 2]

        triangles[self.nelems:, 0] = self.conn[:, 0]
        triangles[self.nelems:, 1] = self.conn[:, 2]
        triangles[self.nelems:, 2] = self.conn[:, 3]

        # Create the triangulation object
        tri_obj = tri.Triangulation(self.x[:,0], self.x[:,1], triangles)

        if ax is None:
            fig, ax = plt.subplots()

        # Set the aspect ratio equal
        ax.set_aspect('equal')

        # Create the contour plot
        ax.tricontourf(tri_obj, u, **kwargs)

        return

# n = 75
# nelems = n**2
# nnodes = (n + 1)*(n + 1)
# y = np.linspace(0, 1, n + 1)
# x = np.linspace(0, 1, n + 1)
# nodes = np.arange(0, (n + 1)*(n + 1)).reshape((n + 1, n + 1))

# # Set the node locations
# X = np.zeros((nnodes, 2))
# for j in range(n + 1):
#     for i in range(n + 1):
#         X[i + j*(n + 1), 0] = x[i]
#         X[i + j*(n + 1), 1] = y[j]

# # Set the connectivity
# conn = np.zeros((nelems, 4), dtype=int)
# for j in range(n):
#     for i in range(n):
#         conn[i + j*n, 0] = nodes[j, i]
#         conn[i + j*n, 1] = nodes[j, i + 1]
#         conn[i + j*n, 2] = nodes[j + 1, i + 1]
#         conn[i + j*n, 3] = nodes[j + 1, i]

# # Set the constrained degrees of freedom at each node
# bcs = []
# for j in range(n+1):
#     bcs.append(nodes[j, 0])
#     bcs.append(nodes[j, -1])
#     bcs.append(nodes[0, j])
#     bcs.append(nodes[-1, j])

# # Make the boundary conditions unique
# bcs = np.unique(bcs)

# problem = PoissonProblem(10)

# # Create the Poisson problem
# poisson = NonlinearPoisson(conn, X, bcs, problem)

# x = np.ones(problem.N)/problem.N
# u = np.ones(poisson.nvars)

# # Set the perturbation direction
# p = np.zeros(poisson.nvars)
# p[nodes] = X[nodes, 0] + X[nodes, 1] + X[nodes, 0]*X[nodes, 1]

# res, J = poisson.assemble_jacobian(x, u)
# Jp = J.dot(p)

# eps = 1e-30
# fd = poisson.assemble_residual(x, u + 1j * eps * p).imag/eps

# for i in range(10):
#     print('{0:25.15e} {1:25.15e} {2:25.15e}'.format(fd[i], Jp[i], (fd[i] - Jp[i])/fd[i]))

# u = poisson.solve(x)
# poisson.plot(u)

# for pval in [1.0, 10, 100, 500]:
#     print('KS value: {0:20.10f} max: {1:20.10f}'.format(poisson.eval_ks(pval, u), np.max(u)))
#     psi = poisson.solve_adjoint(x, u, pval=pval)
#     poisson.plot(psi)