import math
import numpy as np
import scipy.sparse as sp

import igl
import meshplot as mp
mp.offline()
from math import sqrt

v, f = igl.read_triangle_mesh("data/irr4-cyl2.off")
tt, _ = igl.triangle_triangle_adjacency(f)

c = np.loadtxt("data/irr4-cyl2.constraints")
cf = c[:, 0].astype(np.int64)
c = c[:, 1:]


def align_field(V, F, TT, soft_id, soft_value, llambda):
    assert (soft_id[0] > 0)
    assert (soft_id.shape[0] == soft_value.shape[0])

    # Edges
    e1 = V[F[:, 1], :] - V[F[:, 0], :]
    e2 = V[F[:, 2], :] - V[F[:, 0], :]

    # Compute the local reference systems for each face, T1, T2
    T1 = e1 / np.linalg.norm(e1, axis=1)[:, None]

    T2 = np.cross(T1, np.cross(T1, e2))
    T2 /= np.linalg.norm(T2, axis=1)[:, None]

    # Arrays for the entries of the matrix
    data = []
    ii = []
    jj = []

    index = 0
    for f in range(F.shape[0]):
        for ei in range(3):  # Loop over the edges

            # Look up the opposite face
            g = TT[f, ei]

            # If it is a boundary edge, it does not contribute to the energy
            # or avoid to count every edge twice
            if g == -1 or f > g:
                continue

            # Compute the complex representation of the common edge
            e = V[F[f, (ei + 1) % 3], :] - V[F[f, ei], :]

            vef = np.array([np.dot(e, T1[f, :]), np.dot(e, T2[f, :])])
            vef /= np.linalg.norm(vef)
            ef = (vef[0] + vef[1] * 1j).conjugate()

            veg = np.array([np.dot(e, T1[g, :]), np.dot(e, T2[g, :])])
            veg /= np.linalg.norm(veg)
            eg = (veg[0] + veg[1] * 1j).conjugate()

            # Add the term conj(f)^n*ui - conj(g)^n*uj to the energy matrix
            data.append(ef);
            ii.append(index);
            jj.append(f)
            data.append(-eg);
            ii.append(index);
            jj.append(g)

            index += 1

    sqrtl = sqrt(llambda)

    # Convert the constraints into the complex polynomial coefficients and add them as soft constraints

    # Rhs of the system
    b = np.zeros(index + soft_id.shape[0], dtype=complex)

    for ci in range(soft_id.shape[0]):
        f = soft_id[ci]
        v = soft_value[ci, :]

        # Project on the local frame
        c = np.dot(v, T1[f, :]) + np.dot(v, T2[f, :]) * 1j

        data.append(sqrtl);
        ii.append(index);
        jj.append(f)
        b[index] = c * sqrtl

        index += 1

    assert (b.shape[0] == index)

    # Solve the linear system
    A = sp.coo_matrix((data, (ii, jj)), shape=(index, F.shape[0])).asformat("csr")
    u = sp.linalg.spsolve(A.H @ A, A.H @ b)

    R = T1 * u.real[:, None] + T2 * u.imag[:, None]

    return R


def align_field_Q(V, F, TT, soft_id, soft_value, llambda):
    assert (soft_id[0] > 0)
    assert (soft_id.shape[0] == soft_value.shape[0])

    # Edges
    e1 = V[F[:, 1], :] - V[F[:, 0], :]
    e2 = V[F[:, 2], :] - V[F[:, 0], :]

    # Compute the local reference systems for each face, T1, T2
    T1 = e1 / np.linalg.norm(e1, axis=1)[:, None]

    T2 = np.cross(T1, np.cross(T1, e2))
    T2 /= np.linalg.norm(T2, axis=1)[:, None]

    # Arrays for the entries of the matrix
    data = []
    ii = []
    jj = []

    index = 0
    for f in range(F.shape[0]):
        for ei in range(3):  # Loop over the edges

            # Look up the opposite face
            g = TT[f, ei]

            # If it is a boundary edge, it does not contribute to the energy
            # or avoid to count every edge twice
            if g == -1 or f > g:
                continue

            # Compute the complex representation of the common edge
            e = V[F[f, (ei + 1) % 3], :] - V[F[f, ei], :]

            vef = np.array([np.dot(e, T1[f, :]), np.dot(e, T2[f, :])])
            vef /= np.linalg.norm(vef)
            ef = (vef[0] + vef[1] * 1j).conjugate()

            veg = np.array([np.dot(e, T1[g, :]), np.dot(e, T2[g, :])])
            veg /= np.linalg.norm(veg)
            eg = (veg[0] + veg[1] * 1j).conjugate()

            # Add the term conj(f)^n*ui - conj(g)^n*uj to the energy matrix
            data.append(np.dot(ef,ef.conjugate()))
            ii.append(index)
            jj.append(f)

            data.append(np.dot(ef.conjugate(),-eg))
            ii.append(index)
            jj.append(g)

            index += 1

            data.append(np.dot(-ef, eg.conjugate()))
            ii.append(index)
            jj.append(f)

            data.append(np.dot(eg.conjugate(), eg))
            ii.append(index)
            jj.append(g)

            index += 1

    sqrtl = sqrt(llambda)

    # Convert the constraints into the complex polynomial coefficients and add them as soft constraints

    # Rhs of the system
    b = np.zeros(index + soft_id.shape[0], dtype=complex)

    for ci in range(soft_id.shape[0]):
        f = soft_id[ci]
        v = soft_value[ci, :]

        # Project on the local frame
        c = np.dot(v, T1[f, :]) + np.dot(v, T2[f, :]) * 1j

        data.append(sqrtl);
        ii.append(index);
        jj.append(f)
        b[index] = c * sqrtl

        index += 1

    assert (b.shape[0] == index)

    # Solve the linear system
    A = sp.coo_matrix((data, (ii, jj)), shape=(index, F.shape[0])).asformat("csr")
    u = sp.linalg.spsolve(A.H @ A, A.H @ b)

    R = T1 * u.real[:, None] + T2 * u.imag[:, None]

    return R

def plot_mesh_field(V, F, R, constrain_faces):
    # Highlight in red the constrained faces
    col = np.ones_like(f)
    col[constrain_faces, 1:] = 0

    # Scaling of the representative vectors
    avg = igl.avg_edge_length(V, F) / 2

    # Plot from face barycenters
    B = igl.barycenter(V, F)

    p = mp.plot(V, F, c=col)
    p.add_lines(B, B + R * avg)

    return p


R = align_field(v, f, tt, cf, c, 1e6)
plot_mesh_field(v, f, R, cf)