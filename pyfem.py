import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, gmres
from abc import ABC, abstractmethod
import matplotlib.tri as tri
import pyamg
import utils
from utils import time_this


class QuadratureBase(ABC):
    """
    Abstract base class for quadrature object
    """

    @abstractmethod
    def __init__(self, pts, weights):
        """
        Args:
            pts: list-like, list of quadrature points
            weights: list-like, list of quadrature weights
        """
        assert len(pts) == len(weights)  # sanity check
        self.pts = pts
        self.weights = weights
        self.nquads = pts.shape[0]
        return

    @time_this
    def get_nquads(self):
        """
        Get number of quadrature points per element
        """
        return self.nquads

    @time_this
    def get_pt(self, idx=None):
        """
        Query the <idx>-th quadrature point (xi, eta) or (xi, eta, zeta) based
        on quadrature type, if idx is None, return all quadrature points as a
        list
        """
        if idx:
            return self.pts[idx]
        else:
            return self.pts

    @time_this
    def get_weight(self, idx=None):
        """
        Query the weight of <idx>-th quadrature point, if idx is None, return
        all quadrature points as a list
        """
        if idx:
            return self.weights[idx]
        else:
            return self.weights


class QuadratureBilinear2D(QuadratureBase):
    @time_this
    def __init__(self):
        # fmt: off
        pts = np.array([[-1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0)],
                        [ 1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0)],
                        [ 1.0 / np.sqrt(3.0),  1.0 / np.sqrt(3.0)],
                        [-1.0 / np.sqrt(3.0),  1.0 / np.sqrt(3.0)]])
        # fmt: on
        weights = np.array([1.0, 1.0, 1.0, 1.0])
        super().__init__(pts, weights)
        return


class BasisBase(ABC):
    """
    Abstract class for the element basis function
    """

    @abstractmethod
    def __init__(self, ndims, nnodes_per_elem, quadrature: QuadratureBase):
        """
        Inputs:
            ndims: int, number of physical dimensions (2 or 3)
            nnodes_per_elem: int, number of nodes per element
            quadrature: object of type QuadratureBase
        """
        self.ndims = ndims
        self.nnodes_per_elem = nnodes_per_elem
        self.quadrature = quadrature
        self.nquads = self.quadrature.get_nquads()
        self.N = None
        self.Nderiv = None
        return

    @abstractmethod
    def _eval_shape_fun_on_quad_pt(self, qpt):
        """
        Args:
            qpt: a single quadrature point in local coordinate (xi, eta, ...)

        Return:
            shape_vals: list-like, shape function value for each node
        """
        shape_vals = []
        return shape_vals

    @abstractmethod
    def _eval_shape_deriv_on_quad_pt(self, qpt):
        """
        Args:
            qpt: a single quadrature point in local coordinate (xi, eta, ...)

        Return:
            shape_derivs: 1-dim list, [*dN1, *dN2, ...], where dNi = [dNi/dxi,
                          dNi/deta, ..]
        """
        shape_derivs = []
        return shape_derivs

    @time_this
    def eval_shape_fun(self):
        """
        Evaluate the shape function values at quadrature points

        Return:
            shape function values at each quadrature point
        """
        if self.N is None:
            self.N = np.zeros((self.nquads, self.nnodes_per_elem))
            self.N[:, :] = list(
                map(self._eval_shape_fun_on_quad_pt, self.quadrature.get_pt())
            )
        return self.N

    @time_this
    def eval_shape_fun_deriv(self):
        """
        Evaluate the shape function derivatives at quadrature points

        Return:
            shape function derivatives at each quadrature point w.r.t. each
            local coordinate xi, eta, ...
        """
        if self.Nderiv is None:
            self.Nderiv = np.zeros((self.nquads, self.nnodes_per_elem, self.ndims))
            self.Nderiv[:, :, :] = np.array(
                list(map(self._eval_shape_deriv_on_quad_pt, self.quadrature.get_pt()))
            ).reshape((self.nquads, self.nnodes_per_elem, self.ndims))
        return self.Nderiv


class BasisBilinear2D(BasisBase):
    @time_this
    def __init__(self, quadrature):
        ndims = 2
        nnodes_per_elem = 4
        super().__init__(ndims, nnodes_per_elem, quadrature)
        return

    @time_this
    def _eval_shape_fun_on_quad_pt(self, qpt):
        shape_vals = [
            0.25 * (1.0 - qpt[0]) * (1.0 - qpt[1]),
            0.25 * (1.0 + qpt[0]) * (1.0 - qpt[1]),
            0.25 * (1.0 + qpt[0]) * (1.0 + qpt[1]),
            0.25 * (1.0 - qpt[0]) * (1.0 + qpt[1]),
        ]
        return shape_vals

    @time_this
    def _eval_shape_deriv_on_quad_pt(self, qpt):
        shape_derivs = [
            # fmt: off
            -0.25 * (1.0 - qpt[1]),
            -0.25 * (1.0 - qpt[0]),
             0.25 * (1.0 - qpt[1]),
            -0.25 * (1.0 + qpt[0]),
             0.25 * (1.0 + qpt[1]),
             0.25 * (1.0 + qpt[0]),
            -0.25 * (1.0 + qpt[1]),
             0.25 * (1.0 - qpt[0]),
        ]
        return shape_derivs


class ModelBase(ABC):
    """
    Abstract base class for the problem model and physics
    """

    @abstractmethod
    def __init__(
        self,
        ndof_per_node,
        nodes,
        X,
        conn,
        dof_fixed,
        dof_fixed_vals,
        quadrature: QuadratureBase,
        basis: BasisBase,
    ):
        """
        ndof_per_node: int, number of component of state variable
        nodes: 0-based nodal indices, (nnodes, )
        X: nodal location matrix, (nnodes, ndims)
        conn: connectivity matrix, (nelems, nnodes_per_elem)
        dof_fixed: list of dof indices to enforce boundary condition, (Ndof_bc, )
        dof_fixed_vals: list of bc values, (Ndof_bc, ), None means all fixed
                        values are 0
        """
        self.ndof_per_node = ndof_per_node
        self.nodes = np.array(nodes, dtype=int)
        self.X = np.array(X, dtype=float)
        self.conn = np.array(conn, dtype=int)
        self.dof_fixed = np.array(dof_fixed, dtype=int)
        if dof_fixed_vals is None:
            self.dof_fixed_vals = None
        else:
            self.dof_fixed_vals = np.array(dof_fixed_vals, dtype=float)
        self.quadrature = quadrature
        self.basis = basis

        # Get dimension information
        self.nelems = conn.shape[0]
        self.nnodes_per_elem = conn.shape[1]
        self.nnodes = X.shape[0]
        self.ndims = X.shape[1]
        self.nquads = quadrature.get_nquads()

        # Sanity check: nodes
        assert len(self.nodes.shape) == 1  # shape check
        assert self.nodes.min() == 0  # 0-based index check
        assert self.nodes.max() == self.nodes.shape[0] - 1  # no-skip check
        assert len(self.nodes) == len(set(self.nodes))  # no-duplicate check

        # Sanity check: conn
        assert self.conn.flatten().min() == 0
        assert self.conn.flatten().max() == self.nodes.shape[0] - 1

        """
        Compute dof information
        """
        # Create dof arrays
        self.dof, self.dof_each_node, self.conn_dof = utils.create_dof(
            self.nnodes,
            self.nelems,
            self.nnodes_per_elem,
            self.ndof_per_node,
            self.nodes,
            self.conn,
        )

        # Compute free dof indices
        self.dof_free = np.setdiff1d(self.dof, self.dof_fixed)

        """
        Allocate memory for the element-wise data
        """
        # Nodal coordinates
        self.Xe = np.zeros((self.nelems, self.nnodes_per_elem, self.ndims))
        utils.scatter_node_to_elem(self.conn, self.X, self.Xe)

        # Element-wise rhs
        self.rhs_e = np.zeros((self.nelems, self.nnodes_per_elem * self.ndof_per_node))

        # Element-wise Jacobian matrix
        self.Ke = np.zeros(
            (
                self.nelems,
                self.nnodes_per_elem * self.ndof_per_node,
                self.nnodes_per_elem * self.ndof_per_node,
            )
        )

        """
        Allocate memory for quadrature data
        """
        # Nodal coordinates
        self.Xq = np.zeros((self.nelems, self.nquads, self.ndims))

        # Jacobian transformation
        self.Jq = np.zeros((self.nelems, self.nquads, self.ndims, self.ndims))

        # Jacobian transformation inverse
        self.invJq = np.zeros((self.nelems, self.nquads, self.ndims, self.ndims))

        # Jacobian determinant
        self.detJq = np.zeros((self.nelems, self.nquads))

        # gradient of basis w.r.t. global coordinates
        self.Ngrad = np.zeros(
            (self.nelems, self.nquads, self.nnodes_per_elem, self.ndims)
        )

        """
        Global linear system
        """
        # Compute indices for non-zero entries
        self.nz_i, self.nz_j = self._compute_nz_pattern()

        # rhs and K
        self.rhs = np.zeros(self.nnodes * self.ndof_per_node)
        return

    @abstractmethod
    def compute_rhs(self):
        """
        Compute the global rhs vector (without considering boundary conditions)

        Return:
            self.rhs
        """
        return self.rhs

    @abstractmethod
    def compute_jacobian(self):
        """
        Compute the global Jacobian matrix (without considering boundary conditions)

        Return:
            K
        """
        K = None
        return K

    @time_this
    def apply_dirichlet_bcs(self, K, rhs, enforce_symmetric_K=True):
        """
        Apply Dirichlet boundary conditions to global Jacobian matrix and
        right-hand-side vector. In general, the linear system with boundary
        conditions becomes:

            [Krr Krb  [ur    [fr
             0   I  ]  u0] =  u0]

        where u0 is the fixed values for Dirichlet bc, ur is the unknown states,
        optionally, to make the system symmetric, we could move Krb to rhs:

         => [Krr 0  [ur    [fr - Krb u0
             0   I]  u0] =  u0         ]

        Inputs:
            K: stiffness matrix
            rhs: right-hand-side vector
            enforce_symmetric_K: zero out rows and columns to maintain symmetry

        Return:
            K: stiffness matrix with bcs
            rhs: rhs with bcs
        """
        # Save Krb and diagonals
        temp = K[self.dof_free, :]
        Krb = temp[:, self.dof_fixed]
        diag = K.diagonal()

        # Zero-out rows
        for i in self.dof_fixed:
            K.data[K.indptr[i] : K.indptr[i + 1]] = 0

        # Zero-out columns
        if enforce_symmetric_K:
            K = K.tocsc()
            for i in self.dof_fixed:
                K.data[K.indptr[i] : K.indptr[i + 1]] = 0
            K = K.tocsr()

        # Set diagonals to 1
        diag[self.dof_fixed] = 1.0
        K.setdiag(diag)

        # Remove 0
        K.eliminate_zeros()

        # Set rhs
        if self.dof_fixed_vals is None:
            rhs[self.dof_fixed] = 0.0
        else:
            rhs[self.dof_fixed] = self.dof_fixed_vals[:]
            if enforce_symmetric_K:
                rhs[self.dof_free] -= Krb.dot(self.dof_fixed_vals)
        return K, rhs

    @time_this
    def _compute_nz_pattern(self):
        """
        Compute indices for non-zero entries in the global stiffness matrix

        Return:
            nz_i: i-coordinates
            nz_j: j-coordinates
        """
        elem_to_mat_i = [
            i
            for i in range(self.nnodes_per_elem * self.ndof_per_node)
            for j in range(self.nnodes_per_elem * self.ndof_per_node)
        ]
        elem_to_mat_j = [
            j
            for i in range(self.nnodes_per_elem * self.ndof_per_node)
            for j in range(self.nnodes_per_elem * self.ndof_per_node)
        ]
        nz_i = self.conn_dof[:, elem_to_mat_i].flatten()
        nz_j = self.conn_dof[:, elem_to_mat_j].flatten()
        return nz_i, nz_j

    @time_this
    def _assemble_rhs(self, rhs_e, rhs):
        """
        Assemble the global rhs given element-wise rhs: rhs_e -> rhs

        Inputs:
            quadrature: the quadrature object
            rhs_e: element-wise rhs, (nelems, nnodes_per_elem * ndof_per_node)

        Outputs:
            rhs: global rhs, (nnodes * ndof_per_node, )
        """
        for n in range(self.quadrature.get_nquads()):
            np.add.at(rhs[:], self.conn_dof[:, n], rhs_e[:, n])
        return

    @time_this
    def _assemble_jacobian(self, Ke):
        """
        Assemble global K matrix
        """
        K = sparse.coo_matrix((Ke.flatten(), (self.nz_i, self.nz_j)))
        return K.tocsr()


class LinearPoisson2D(ModelBase):
    """
    The 2-dimensional Poisson equation

    Equation:
    -∆u = g in Ω

    Boundary condition:
    u = u0 on ∂Ω

    Weak form:
    ∫ ∇u ∇v dΩ = ∫gv dΩ for test function v
    """

    @time_this
    def __init__(
        self,
        nodes,
        X,
        conn,
        dof_fixed,
        dof_fixed_vals,
        quadrature: QuadratureBase,
        basis: BasisBase,
    ):
        ndof_per_node = 1
        super().__init__(
            ndof_per_node, nodes, X, conn, dof_fixed, dof_fixed_vals, quadrature, basis
        )
        self.fun_vals = None
        return

    @time_this
    def compute_rhs(self):
        # Compute element rhs vector -> rhs_e
        self._compute_element_rhs(self.rhs_e)

        # Assemble the global rhs vector -> rhs
        self._assemble_rhs(self.rhs_e, self.rhs)
        return self.rhs

    @time_this
    def compute_jacobian(self):
        """
        Compute the element Jacobian (stiffness) matrices Ke and assemble the
        global K

        Return:
            K: (sparse) global Jacobian matrix
        """
        # Compute element Jacobian -> Ke
        self._compute_element_jacobian(self.Ke)

        # Assemble global Jacobian -> K
        K = self._assemble_jacobian(self.Ke)
        return K

    @time_this
    def _compute_gfun(self, Xq, fun_vals):
        xvals = Xq[..., 0]
        yvals = Xq[..., 1]
        fun_vals[...] = xvals * (xvals - 5.0) * (xvals - 10.0) * yvals * (yvals - 4.0)
        return

    @time_this
    def _compute_element_rhs(self, rhs_e):
        """
        Evaluate element-wise rhs vectors:

            rhs_e = ∑ detJq wq (gN)_q
                    q

        Outputs:
            rhs_e: element-wise rhs, (nelems, nnodes_per_elem * ndof_per_node)
        """
        # Compute shape function and derivatives
        N = self.basis.eval_shape_fun()
        Nderiv = self.basis.eval_shape_fun_deriv()

        # Compute Jacobian transformation -> Jq
        utils.compute_jtrans(self.Xe, Nderiv, self.Jq)

        # Compute Jacobian determinant -> detJq
        utils.compute_jdet(self.Jq, self.detJq)

        # Compute element mapping -> Xq
        utils.compute_elem_interp(N, self.Xe, self.Xq)

        # Allocate memory for quadrature function values
        if self.fun_vals is None:
            self.fun_vals = np.zeros(self.Xq.shape[0:-1])

        # Evaluate function g
        self._compute_gfun(self.Xq, self.fun_vals)

        # Compute element rhs
        wq = self.quadrature.get_weight()
        rhs_e[...] = np.einsum("ik,k,jk,ik -> ij", self.detJq, wq, N, self.fun_vals)
        return

    @time_this
    def _compute_element_jacobian(self, Ke):
        """
        Evaluate element-wise Jacobian matrices

            Ke = ∑ detJq wq ( NxNxT + NyNyT)_q
                 q

        Outputs:
            Ke: element-wise Jacobian matrix, (nelems, nnodes_per_elem *
                ndof_per_node, nnodes_per_elem * ndof_per_node)
        """
        # Compute Jacobian derivatives
        Nderiv = self.basis.eval_shape_fun_deriv()

        # Compute Jacobian transformation -> Jq
        utils.compute_jtrans(self.Xe, Nderiv, self.Jq)

        # Compute Jacobian determinant -> detJq
        utils.compute_jdet(self.Jq, self.detJq)

        # Compute shape function gradient -> (invJq), Ngrad
        utils.compute_basis_grad(self.Jq, self.detJq, Nderiv, self.invJq, self.Ngrad)

        wq = self.quadrature.get_weight()
        Ke[...] = np.einsum(
            "iq,q,iqjl,iqkl -> ijk", self.detJq, wq, self.Ngrad, self.Ngrad
        )
        return


class PlaneStress2D(ModelBase):
    """
    The 2-dimensional linear elasticity equation

    Integral form using the principle of virtual displacement:
    ∫ eTs dΩ = ∫uTf dΩ

    where:
        e: strain vector
        s: stress vector
        u: nodal displacement
        f: nodal external force
    """

    @time_this
    def __init__(
        self,
        nodes,
        X,
        conn,
        dof_fixed,
        dof_fixed_vals,
        nodal_force,
        quadrature: QuadratureBase,
        basis: BasisBase,
        E=10.0,
        nu=0.3,
    ):
        # Call base class's constructor
        ndof_per_node = 2
        super().__init__(
            ndof_per_node, nodes, X, conn, dof_fixed, dof_fixed_vals, quadrature, basis
        )

        # Plane stress-specific variables
        self.nodal_force = nodal_force
        self.C = E * np.array(
            [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - nu)]]
        )
        self.C *= 1.0 / (1.0 - nu**2)
        n_stress_tensor = int(ndof_per_node * (ndof_per_node + 1) / 2)
        self.Be = np.zeros(
            (
                self.nelems,
                self.nquads,
                n_stress_tensor,
                self.nnodes_per_elem * ndof_per_node,
            )
        )
        return

    @time_this
    def compute_rhs(self):
        """
        It turns out for this problem, the rhs (before applying bcs) is just nodal force
        """
        self.rhs[
            self.dof_each_node[list(self.nodal_force.keys())].flatten()
        ] = np.array(list(self.nodal_force.values())).flatten()
        return self.rhs

    @time_this
    def compute_jacobian(self):
        """
        Compute the element Jacobian (stiffness) matrices Ke and assemble the
        global K

        Return:
            K: (sparse) global Jacobian matrix
        """
        # Compute element Jacobian -> Ke
        self._compute_element_jacobian(self.Ke)

        # Assemble global Jacobian -> K
        K = self._assemble_jacobian(self.Ke)
        return K

    @time_this
    def _compute_element_B(self, Ngrad, Be):
        """
        Compute element B matrix

        Input:
            Ngrad: shape function gradient, (nelems, nquads, nnodes_per_elem,
                   ndims)

        Output:
            Be: function that maps u to strain, (nelems, ndof_per_node,
            nnodes_per_elem * ndof_per_node)
        """
        Nx = Ngrad[..., 0]
        Ny = Ngrad[..., 1]
        Be[:, :, 0, ::2] = Nx
        Be[:, :, 1, 1::2] = Ny
        Be[:, :, 2, ::2] = Ny
        Be[:, :, 2, 1::2] = Nx
        return

    @time_this
    def _compute_element_jacobian(self, Ke):
        """
        Evaluate element-wise stiffness matrix Ke

        ∫ eTs dΩ can be integrated numerically using quadrature:

            ∫eTs dΩ ~= ∑ detJq wq eTs
                       q

        where e = Bu, s = Ce = CBu.  Then, Ke can be computed by taking the 2nd
        order derivatives w.r.t. u:

            Ke = ∑ detJq wq (BTCB)q
                  q

        Outputs:
            Ke: element-wise Jacobian matrix, (nelems, nnodes_per_elem *
                ndof_per_node, nnodes_per_elem * ndof_per_node)
        """
        # Compute Jacobian derivatives
        Nderiv = self.basis.eval_shape_fun_deriv()

        # Compute Jacobian transformation -> Jq
        utils.compute_jtrans(self.Xe, Nderiv, self.Jq)

        # Compute Jacobian determinant -> detJq
        utils.compute_jdet(self.Jq, self.detJq)

        # Compute shape function gradient -> (invJq), Ngrad
        utils.compute_basis_grad(self.Jq, self.detJq, Nderiv, self.invJq, self.Ngrad)

        # Compute Be matrix -> Be
        self._compute_element_B(self.Ngrad, self.Be)

        wq = self.quadrature.get_weight()
        Ke[...] = np.einsum(
            "iq,q,iqnj,nm,iqmk->ijk", self.detJq, wq, self.Be, self.C, self.Be
        )
        return


class Assembler:
    @time_this
    def __init__(self, model: ModelBase):
        """
        The finite element problem assembler.

        Args:
            model: the problem model instance
        """
        self.model = model
        return

    @time_this
    def solve(self, method="gmres"):
        """
        Perform the static analysis
        """
        # Construct the linear system
        K = self.model.compute_jacobian()
        rhs = self.model.compute_rhs()

        # Apply Dirichlet boundary conditions
        K, rhs = self.model.apply_dirichlet_bcs(K, rhs, enforce_symmetric_K=True)

        # Solve the linear system
        u = self._solve_linear_system(K, rhs, method)
        return u

    @time_this
    def plot(self, u, ax, **kwargs):
        """
        Create a 2-dimensional contour plot for a scalar variable u
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
        tri_obj = tri.Triangulation(self.X[:, 0], self.X[:, 1], triangles)

        # Set the aspect ratio equal
        ax.set_aspect("equal")

        # Create the contour plot
        ax.tricontourf(tri_obj, u, **kwargs)
        return

    @time_this
    def _solve_linear_system(self, K, rhs, method):
        """
        Solve the linear system

        Args:
            method: direct or gmres

        Return:
            u: solution
        """
        if method == "direct":
            u = spsolve(K, rhs)
        else:
            ml = pyamg.smoothed_aggregation_solver(K)
            M = ml.aspreconditioner()
            u, fail = gmres(K, rhs, M=M)
            if fail:
                raise RuntimeError(f"GMRES failed with code {fail}")
        return u


class ProblemCreator:
    """
    Utility to create problem mesh, boundary condition and load
    """

    @time_this
    def __init__(self, nelems_x, nelems_y):
        self.nelems_per_node = 4
        self.nelems_x = nelems_x
        self.nelems_y = nelems_y

        self.nelems = self.nelems_x * self.nelems_y
        self.nnodes = (self.nelems_x + 1) * (self.nelems_y + 1)

        x = np.linspace(0, self.nelems_x / self.nelems_y, self.nelems_x + 1)
        y = np.linspace(0, 1, self.nelems_y + 1)
        nodes2d = np.arange(0, (self.nelems_y + 1) * (self.nelems_x + 1)).reshape(
            (self.nelems_y + 1, self.nelems_x + 1)
        )

        # Set the node locations
        X = np.zeros((self.nnodes, 2))
        for j in range(self.nelems_y + 1):
            for i in range(self.nelems_x + 1):
                X[i + j * (self.nelems_x + 1), 0] = x[i]
                X[i + j * (self.nelems_x + 1), 1] = y[j]

        # Set the connectivity
        conn = np.zeros((self.nelems, 4), dtype=int)
        for j in range(self.nelems_y):
            for i in range(self.nelems_x):
                conn[i + j * self.nelems_x, 0] = nodes2d[j, i]
                conn[i + j * self.nelems_x, 1] = nodes2d[j, i + 1]
                conn[i + j * self.nelems_x, 2] = nodes2d[j + 1, i + 1]
                conn[i + j * self.nelems_x, 3] = nodes2d[j + 1, i]

        self.nodes2d = nodes2d
        self.nodes = nodes2d.flatten()
        self.conn = conn
        self.X = X

        return

    @time_this
    def create_poisson_problem(self):
        # Set fixed dof
        dof_fixed = []
        for j in range(self.nelems_y):
            dof_fixed.append(self.nodes2d[j, 0])
            dof_fixed.append(self.nodes2d[j, -1])
        return self.nodes, self.conn, self.X, dof_fixed

    @time_this
    def create_linear_elasticity_problem(self):
        # Set fixed dof
        dof_fixed = []
        for j in range(self.nelems_y):
            dof_fixed.append(2 * self.nodes2d[j, 0])
            dof_fixed.append(2 * self.nodes2d[j, 0] + 1)

        # Set nodal force
        nodal_force = {}
        for j in range(self.nelems_y):
            nodal_force[self.nodes2d[j, -1]] = [0.0, -1.0]

        return self.nodes, self.conn, self.X, dof_fixed, nodal_force
