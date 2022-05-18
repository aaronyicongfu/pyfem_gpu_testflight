import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pylab as plt
import matplotlib.tri as tri


class PlaneStress:
    def __init__(self, conn, x, bcs, forces={}, E=10.0, nu=0.3, rho=1.0):
        self.conn = np.array(conn)
        self.x = np.array(x)

        self.nelems = self.conn.shape[0]
        self.nnodes = int(np.max(self.conn)) + 1
        self.nvars = 2 * self.nnodes

        # Compute the constitutivve matrix
        self.C = E * np.array(
            [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - nu)]]
        )
        self.C *= 1.0 / (1.0 - nu**2)
        self.rho = rho

        self.reduced = self._compute_reduced_variables(self.nvars, bcs)
        self.f = self._compute_forces(self.nvars, forces)

        # Set up the i-j indices for the matrix - these are the row
        # and column indices in the stiffness matrix
        self.var = np.zeros((self.conn.shape[0], 8), dtype=int)
        self.var[:, ::2] = 2 * self.conn
        self.var[:, 1::2] = 2 * self.conn + 1

        i = []
        j = []
        for index in range(self.nelems):
            for ii in self.var[index, :]:
                for jj in self.var[index, :]:
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

        # For each node that is in the boundary condition dictionary
        for node in bcs:
            uv_list = bcs[node]

            # For each index in the boundary conditions (corresponding to
            # either a constraint on u and/or constraint on v
            for index in uv_list:
                var = 2 * node + index
                reduced.remove(var)

        return reduced

    def _compute_forces(self, nvars, forces):
        """
        Unpack the dictionary containing the forces
        """
        f = np.zeros(nvars)

        for node in forces:
            f[2 * node] += forces[node][0]
            f[2 * node + 1] += forces[node][1]

        return f

    def assemble_stiffness_matrix(self):
        """
        Assemble the stiffness matrix
        """

        # Compute the element stiffness matrix
        gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

        # Assemble all of the the 8 x 8 element stiffness matrix
        Ke = np.zeros((self.nelems, 8, 8))
        Be = np.zeros((self.nelems, 3, 8))

        J = np.zeros((self.nelems, 2, 2))
        invJ = np.zeros(J.shape)

        # Compute the x and y coordinates of each element
        xe = self.x[self.conn, 0]
        ye = self.x[self.conn, 1]

        for j in range(2):
            for i in range(2):
                xi = gauss_pts[i]
                eta = gauss_pts[j]
                Nxi = 0.25 * np.array(
                    [-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)]
                )
                Neta = 0.25 * np.array(
                    [-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)]
                )

                # Compute the Jacobian transformation at each quadrature points
                J[:, 0, 0] = np.dot(xe, Nxi)
                J[:, 1, 0] = np.dot(ye, Nxi)
                J[:, 0, 1] = np.dot(xe, Neta)
                J[:, 1, 1] = np.dot(ye, Neta)

                # Compute the inverse of the Jacobian
                detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
                invJ[:, 0, 0] = J[:, 1, 1] / detJ
                invJ[:, 0, 1] = -J[:, 0, 1] / detJ
                invJ[:, 1, 0] = -J[:, 1, 0] / detJ
                invJ[:, 1, 1] = J[:, 0, 0] / detJ

                # Compute the derivative of the shape functions w.r.t. xi and eta
                # [Nx, Ny] = [Nxi, Neta]*invJ
                Nx = np.outer(invJ[:, 0, 0], Nxi) + np.outer(invJ[:, 1, 0], Neta)
                Ny = np.outer(invJ[:, 0, 1], Nxi) + np.outer(invJ[:, 1, 1], Neta)

                # Set the B matrix for each element
                Be[:, 0, ::2] = Nx
                Be[:, 1, 1::2] = Ny
                Be[:, 2, ::2] = Ny
                Be[:, 2, 1::2] = Nx

                # This is a fancy (and fast) way to compute the element matrices
                Ke += np.einsum("n,nij,ik,nkl -> njl", detJ, Be, self.C, Be)

                # This is a slower way to compute the element matrices
                # for k in range(self.nelems):
                #     Ke[k, :, :] += detJ[k]*np.dot(Be[k, :, :].T, np.dot(self.C, Be[k, :, :]))

        K = sparse.coo_matrix((Ke.flatten(), (self.i, self.j)))
        K = K.tocsr()

        return K

    def compute_strains(self, u):
        """
        Compute the strains at each quadrature point
        """

        # The strain at each quadrature point
        strain = np.zeros((self.nelems, 4, 3))

        # Compute the element stiffness matrix
        gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

        Be = np.zeros((self.nelems, 3, 8))
        J = np.zeros((self.nelems, 2, 2))
        invJ = np.zeros(J.shape)

        # Compute the x and y coordinates of each element
        xe = self.x[self.conn, 0]
        ye = self.x[self.conn, 1]

        for j in range(2):
            for i in range(2):
                xi = gauss_pts[i]
                eta = gauss_pts[j]
                Nxi = 0.25 * np.array(
                    [-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)]
                )
                Neta = 0.25 * np.array(
                    [-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)]
                )

                # Compute the Jacobian transformation at each quadrature points
                J[:, 0, 0] = np.dot(xe, Nxi)
                J[:, 1, 0] = np.dot(ye, Nxi)
                J[:, 0, 1] = np.dot(xe, Neta)
                J[:, 1, 1] = np.dot(ye, Neta)

                # Compute the inverse of the Jacobian
                detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
                invJ[:, 0, 0] = J[:, 1, 1] / detJ
                invJ[:, 0, 1] = -J[:, 0, 1] / detJ
                invJ[:, 1, 0] = -J[:, 1, 0] / detJ
                invJ[:, 1, 1] = J[:, 0, 0] / detJ

                # Compute the derivative of the shape functions w.r.t. xi and eta
                # [Nx, Ny] = [Nxi, Neta]*invJ
                Nx = np.outer(invJ[:, 0, 0], Nxi) + np.outer(invJ[:, 1, 0], Neta)
                Ny = np.outer(invJ[:, 0, 1], Nxi) + np.outer(invJ[:, 1, 1], Neta)

                # Set the B matrix for each element
                Be[:, 0, ::2] = Nx
                Be[:, 1, 1::2] = Ny
                Be[:, 2, ::2] = Ny
                Be[:, 2, 1::2] = Nx

                index = i + 2 * j
                for k in range(self.nelems):
                    strain[k, index, :] = np.dot(Be[k], u[self.var[k, :]])

        return strain

    def compute_nodal_strains(self, u):
        """
        Compute the strain values at the nodes
        """

        # Compute the strains at each quadrature point
        e = self.compute_strains(u)

        # Compute the weights - the number of times a quadrature
        # point will be summed to a node
        weights = np.zeros(self.nnodes)
        weights[self.conn] += 1.0

        # Allocate the space for the nodal strains
        strains = np.zeros((self.nnodes, 3))

        # Account for the difference in indexing between quadrature
        # points and nodes
        index = [0, 1, 3, 2]

        # Perform the averaging by summing and then weighting the strains
        strains[self.conn, :] += e[:, index, :]
        for k in range(3):
            strains[:, k] *= 1.0 / weights

        return strains

    def assemble_mass_matrix(self):
        """
        Assemble the stiffness matrix
        """

        # Compute the element stiffness matrix
        gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

        # Assemble all of the the 8 x 8 element mass matrices
        Me = np.zeros((self.nelems, 8, 8))
        He = np.zeros((self.nelems, 2, 8))

        J = np.zeros((self.nelems, 2, 2))
        invJ = np.zeros(J.shape)

        # Compute the x and y coordinates of each element
        xe = self.x[self.conn, 0]
        ye = self.x[self.conn, 1]

        for j in range(2):
            for i in range(2):
                xi = gauss_pts[i]
                eta = gauss_pts[j]
                N = 0.25 * np.array(
                    [
                        (1.0 - xi) * (1.0 - eta),
                        (1.0 + xi) * (1.0 - eta),
                        (1.0 + xi) * (1.0 + eta),
                        (1.0 - xi) * (1.0 + eta),
                    ]
                )
                Nxi = 0.25 * np.array(
                    [-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)]
                )
                Neta = 0.25 * np.array(
                    [-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)]
                )

                # Compute the Jacobian transformation at each quadrature points
                J[:, 0, 0] = np.dot(xe, Nxi)
                J[:, 1, 0] = np.dot(ye, Nxi)
                J[:, 0, 1] = np.dot(xe, Neta)
                J[:, 1, 1] = np.dot(ye, Neta)

                # Compute the inverse of the Jacobian
                detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]

                # Set the B matrix for each element
                He[:, 0, ::2] = N
                He[:, 1, 1::2] = N

                # This is a fancy (and fast) way to compute the element matrices
                Me += self.rho * np.einsum("n,nij,nil -> njl", detJ, He, He)

                # This is a slower way to compute the element matrices
                # for k in range(self.nelems):
                #     Me[k, :, :] += self.rho*detJ[k]*np.dot(He[k, :, :].T, He[k, :, :])

        M = sparse.coo_matrix((Me.flatten(), (self.i, self.j)))
        M = M.tocsr()

        return M

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

        K = self.assemble_stiffness_matrix()
        Kr = self.reduce_matrix(K)
        fr = self.reduce_vector(self.f)

        ur = sparse.linalg.spsolve(Kr, fr)

        u = np.zeros(self.nvars)
        u[self.reduced] = ur

        return u

    def frequencies(self, k=5, sigma=0.0):
        """
        Compute the k-th smallest natural frequencies
        """

        K = self.assemble_stiffness_matrix()
        Kr = self.reduce_matrix(K)

        M = self.assemble_mass_matrix()
        Mr = self.reduce_matrix(M)

        # Find the eigenvalues closest to zero. This uses a shift and
        # invert strategy around sigma = 0, which means that the largest
        # magnitude values are closest to zero.
        eigs, ur = sparse.linalg.eigsh(Kr, M=Mr, k=5, sigma=sigma, which="LM", tol=1e-6)

        u = np.zeros((self.nvars, k))
        for i in range(k):
            u[self.reduced, i] = ur[:, i]

        return np.sqrt(eigs), u

    def plot(self, u, ax=None, **kwargs):
        """
        Create a plot
        """

        # Create the triangles
        triangles = np.zeros((2 * self.nelems, 3), dtype=int)
        triangles[: self.nelems, 0] = self.conn[:, 0]
        triangles[: self.nelems, 1] = self.conn[:, 1]
        triangles[: self.nelems, 2] = self.conn[:, 2]

        triangles[self.nelems :, 0] = self.conn[:, 0]
        triangles[self.nelems :, 1] = self.conn[:, 2]
        triangles[self.nelems :, 2] = self.conn[:, 3]

        # Create the triangulation object
        tri_obj = tri.Triangulation(self.x[:, 0], self.x[:, 1], triangles)

        if ax is None:
            fig, ax = plt.subplots()

        # Set the aspect ratio equal
        ax.set_aspect("equal")

        # Create the contour plot
        ax.tricontourf(tri_obj, u, **kwargs)

        return


if __name__ == "__main__":
    m = 200
    n = 50
    nelems = m * n
    nnodes = (m + 1) * (n + 1)
    y = np.linspace(0, 2, n + 1)
    x = np.linspace(0, 10, m + 1)
    nodes = np.arange(0, (n + 1) * (m + 1)).reshape((n + 1, m + 1))

    # Set the node locations
    X = np.zeros((nnodes, 2))
    for j in range(n + 1):
        for i in range(m + 1):
            X[i + j * (m + 1), 0] = x[i]
            X[i + j * (m + 1), 1] = y[j]

    # Set the connectivity
    conn = np.zeros((nelems, 4), dtype=int)
    for j in range(n):
        for i in range(m):
            conn[i + j * m, 0] = nodes[j, i]
            conn[i + j * m, 1] = nodes[j, i + 1]
            conn[i + j * m, 2] = nodes[j + 1, i + 1]
            conn[i + j * m, 3] = nodes[j + 1, i]

    # Set the constrained degrees of freedom at each node
    bcs = {}
    for j in range(n):
        bcs[nodes[j, 0]] = [0, 1]

    P = 1.0
    forces = {}
    for j in range(n):
        forces[nodes[j, -1]] = [0, -P]

    # Create the plane stress object
    ps = PlaneStress(conn, X, bcs, forces)

    # Solve for the displacements
    u = ps.solve()
    e = ps.compute_nodal_strains(u)

    # Plot the u and the v displacements
    fig, ax = plt.subplots(2, 1, figsize=(8, 4))
    ps.plot(u[::2], ax=ax[0], levels=20)
    ps.plot(u[1::2], ax=ax[1], levels=20)
    ax[0].set_title("u")
    ax[1].set_title("v")

    # Plot the strains
    fig, ax = plt.subplots(3, 1, figsize=(8, 6))
    ps.plot(e[:, 0], ax=ax[0], levels=20)
    ps.plot(e[:, 1], ax=ax[1], levels=20)
    ps.plot(e[:, 2], ax=ax[2], levels=20)
    ax[0].set_title(r"$\epsilon_x$")
    ax[1].set_title(r"$\epsilon_y$")
    ax[2].set_title(r"$\gamma_{xy}$")
    plt.show()
