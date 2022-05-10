import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pylab as plt
import matplotlib.tri as tri

class Poisson:
    def __init__(self, conn, x, bcs, gfunc):
        self.conn = np.array(conn)
        self.x = np.array(x)

        self.nelems = self.conn.shape[0]
        self.nnodes = int(np.max(self.conn)) + 1
        self.nvars = self.nnodes

        self.reduced = self._compute_reduced_variables(self.nvars, bcs)
        self.g = self._compute_rhs(gfunc)

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

    def _compute_rhs(self, gfunc):
        """
        Compute the right-hand-side using the function callback
        """

        forces = np.zeros(self.nnodes)

        # Compute the element stiffness matrix
        gauss_pts = [-1.0/np.sqrt(3.0), 1.0/np.sqrt(3.0)]

        J = np.zeros((self.nelems, 2, 2))
        detJ = np.zeros(self.nelems)

        # Compute the x and y coordinates of each element
        xe = self.x[self.conn, 0]
        ye = self.x[self.conn, 1]

        fe = np.zeros(self.conn.shape)

        for j in range(2):
            for i in range(2):
                xi = gauss_pts[i]
                eta = gauss_pts[j]
                N, Nxi, Neta = self._eval_basis_and_jacobian(xi, eta, xe, ye, J, detJ)

                # Evaluate the function
                xvals = np.dot(xe, N)
                yvals = np.dot(ye, N)
                gvals = gfunc(xvals, yvals)

                fe += np.outer(detJ * gvals, N)

        for i in range(4):
            np.add.at(forces, self.conn[:, i], fe[:, i])

        return forces

    def assemble_jacobian(self):
        """
        Assemble the Jacobian matrix
        """

        # Compute the element stiffness matrix
        gauss_pts = [-1.0/np.sqrt(3.0), 1.0/np.sqrt(3.0)]

        # Assemble all of the the 4 x 4 element stiffness matrix
        Ke = np.zeros((self.nelems, 4, 4))
        Be = np.zeros((self.nelems, 2, 4))

        J = np.zeros((self.nelems, 2, 2))
        invJ = np.zeros(J.shape)
        detJ = np.zeros(self.nelems)

        # Compute the x and y coordinates of each element
        xe = self.x[self.conn, 0]
        ye = self.x[self.conn, 1]

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

                # This is a fancy (and fast) way to compute the element matrices
                Ke += np.einsum('n,nij,nil -> njl', detJ, Be, Be)

        K = sparse.coo_matrix((Ke.flatten(), (self.i, self.j)))
        K = K.tocsr()

        return K

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

    def solve(self):
        """
        Perform a linear static analysis
        """

        K = self.assemble_jacobian()
        Kr = self.reduce_matrix(K)
        fr = self.reduce_vector(self.g)

        ur = sparse.linalg.spsolve(Kr, fr)

        u = np.zeros(self.nvars)
        u[self.reduced] = ur

        return u

    def solve_adjoint(self, pval):

        K = self.assemble_jacobian()
        Kr = self.reduce_matrix(K)
        fr = self.reduce_vector(self.g)

        ur = sparse.linalg.spsolve(Kr, fr)

        u = np.zeros(self.nvars)
        u[self.reduced] = ur

        max_val = self.eval_ks(pval, u)
        rhs = self.eval_ks_adjoint_rhs(pval, u)
        rhsr = self.reduce_vector(rhs)

        # Solve K * psi = - df/du^{T}
        psir = sparse.linalg.spsolve(Kr, -rhsr)

        psi = np.zeros(self.nvars)
        psi[self.reduced] = psir

        return psi

    def plot(self, u, ax=None, **kwargs):
        """
        Create a plot
        """

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

m = 200
n = 50
nelems = m*n
nnodes = (m + 1)*(n + 1)
y = np.linspace(0, 4, n + 1)
x = np.linspace(0, 10, m + 1)
nodes = np.arange(0, (n + 1)*(m + 1)).reshape((n + 1, m + 1))

# Set the node locations
X = np.zeros((nnodes, 2))
for j in range(n + 1):
    for i in range(m + 1):
        X[i + j*(m + 1), 0] = x[i]
        X[i + j*(m + 1), 1] = y[j]

# Set the connectivity
conn = np.zeros((nelems, 4), dtype=int)
for j in range(n):
    for i in range(m):
        conn[i + j*m, 0] = nodes[j, i]
        conn[i + j*m, 1] = nodes[j, i + 1]
        conn[i + j*m, 2] = nodes[j + 1, i + 1]
        conn[i + j*m, 3] = nodes[j + 1, i]

# Set the constrained degrees of freedom at each node
bcs = []
for j in range(n):
    bcs.append(nodes[j, 0])
    bcs.append(nodes[j, -1])

def gfunc(xvals, yvals):
    return xvals * (xvals - 5.0) * (xvals - 10.0) * yvals * (yvals - 4.0)

# Create the Poisson problem
poisson = Poisson(conn, X, bcs, gfunc)

# Solve for the displacements
u = poisson.solve()

# Plot the u and the v displacements
fig, ax = plt.subplots(figsize=(8, 4))
poisson.plot(u, ax=ax, levels=20)
ax.set_title('u')

fig, ax = plt.subplots(4, 1, figsize=(8, 15))
for index, pval in enumerate([1, 10, 100, 500]):
    psi = poisson.solve_adjoint(pval)
    poisson.plot(psi, ax=ax[index], levels=20)
    ax[index].set_title(r'$\psi$ with p = %g'%(pval))

plt.show()