import math
import numpy as np
import scipy.sparse as sp
#np.set_printoptions(threshold=np.inf)
import igl
import meshplot as mp
mp.offline()
from math import sqrt

v, f = igl.read_triangle_mesh("data/irr4-cyl2.off")
tt, _ = igl.triangle_triangle_adjacency(f)

c = np.loadtxt("data/irr4-cyl2.constraints")
cf = c[:, 0].astype(np.int64)
c = c[:, 1:]


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
            ii.append(f)
            jj.append(f)

            data.append(np.dot(ef.conjugate(),-eg))
            ii.append(f)
            jj.append(g)

            data.append(np.dot(-ef, eg.conjugate()))
            ii.append(g)
            jj.append(f)

            data.append(np.dot(eg.conjugate(), eg))
            ii.append(g)
            jj.append(g)


    # Convert the constraints into the complex polynomial coefficients and add them as soft constraints
    Q = sp.coo_matrix((data, (ii, jj)), shape=(F.shape[0], F.shape[0])).asformat("lil")

    # Rhs of the system
    b = np.zeros(F.shape[0], dtype=complex)


    for ci in range(soft_id.shape[0]):
        f = soft_id[ci]
        v = soft_value[ci, :]
        # Project on the local frame
        c = np.dot(v, T1[f, :]) + np.dot(v, T2[f, :]) * 1j
        Q[f,:]=0
        Q[f,f]=1
        b[f]=c

    for ci in range(soft_id.shape[0]):
        f = soft_id[ci]
        v = soft_value[ci, :]

        # Project on the local frame
        c = np.dot(v, T1[f, :]) + np.dot(v, T2[f, :]) * 1j

        for fj in range(F.shape[0]):
            b[fj] -= np.dot(Q[fj,f],c)

    #assert (b.shape[0] == index)

    # Solve the linear system
    A = Q.asformat("csr")
    u = sp.linalg.spsolve(A, b)
    for ci in range(soft_id.shape[0]):
        f = soft_id[ci]
        v = soft_value[ci, :]
        # Project on the local frame
        c = np.dot(v, T1[f, :]) + np.dot(v, T2[f, :]) * 1j
        u[f] = c

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


R2 = align_field_Q(v, f, tt, cf, c, 1e6)
plot_mesh_field(v, f, R2, cf)

def scalar_field(v,f,vector_field):
    G=igl.grad(v,f)
    area=igl.doublearea(v,f)/2
    At=np.diag(np.tile(area, 3))
    ut=vector_field.T.flatten()
    K=G.T@At@G
    b=G.T@At@ut
    K[0,:]=0
    K[0,0]=1
    b[0]=0

    si=sp.linalg.spsolve(K, b)
    gt=(G@si).reshape(3,-1).T
    poisson_error1=np.linalg.norm(gt-vector_field,axis=1)
    poisson_error2=np.linalg.norm(gt-vector_field)
    return si,poisson_error1,poisson_error2


def plot_mesh_scalar_field(V, F, R, s_field):
    avg = igl.avg_edge_length(V, F) / 2
    B = igl.barycenter(V, F)
    p = mp.plot(V, F, c=s_field)

    p.add_lines(B, B + R * avg)
    return p

s_field,p_error1,p_error2 = scalar_field(v,f,R2)
plot_mesh_scalar_field(v,f,R2,s_field)

def plot_possion_error(V,F,error):
    p=mp.plot(V,F,c=error)
    return p

s_field = scalar_field(v,f,R2)
plot_possion_error(v,f,p_error1)


print("The sum of error of every face:")
print(sum(p_error1))
print("norm the total:")
print(p_error2)

def harmonic_para(v,f):
    boundaryloop=igl.boundary_loop(f)
    boundaryuv=igl.map_vertices_to_circle(v,boundaryloop)
    uv = igl.harmonic(v,f,boundaryloop, boundaryuv,1)
    return uv


uv,v_grad=harmonic_para(v,f)
G = igl.grad(v, f)
v_grad = (G @ uv[:, 1]).reshape(3, -1).T
p = mp.subplot(v, f, uv=uv, s=[1, 2, 0])
mp.subplot(uv, f, shading={"wireframe": True}, data=p, s=[1,2,1])
plot_mesh_scalar_field(v,f,v_grad,uv[:,1])



def LSCM_para(v,f):
    boundaryloop = igl.boundary_loop(f)
    b = np.array([boundaryloop[0], boundaryloop[int(len(boundaryloop) / 2)]])
    bc = np.array([[0., 0.], [1., 0.]])
    _,uv = igl.lscm(v, f,b,bc)
    return uv

uv=harmonic_para(v,f)
G = igl.grad(v, f)
v_grad = (G @ uv[:, 1]).reshape(3, -1).T
p = mp.subplot(v, f, uv=uv, s=[1, 2, 0])
mp.subplot(uv, f, shading={"wireframe": True}, data=p, s=[1,2,1])
plot_mesh_scalar_field(v,f,v_grad,uv[:,1])






v, f = igl.read_triangle_mesh("data/irr4-cyl2.off")
tt, _ = igl.triangle_triangle_adjacency(f)
c = np.loadtxt("data/irr4-cyl2.constraints")
cf = c[:, 0].astype(np.int64)
c = c[:, 1:]
R = align_field_Q(v, f, tt, cf, c, 1e6)
s_field = scalar_field(v,f,R)
uv=harmonic_para(v,f)

mp.plot(uv, f, uv=uv, shading={"wireframe": True})

uv_e=uv
uv_e[:,1]=s_field
p = mp.subplot(v, f, uv=uv, s=[1, 2, 0])
mp.plot(uv, f, uv=uv, shading={"wireframe": True})