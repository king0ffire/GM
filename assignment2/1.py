import numpy as np
import igl
import meshplot as mp


# Utility function to generate a tet grid
# n is a 3-tuple with the number of cell in every direction
# mmin/mmax are the grid bounding box corners

def tet_grid(n, mmin, mmax):
    nx = n[0]
    ny = n[1]
    nz = n[2]

    delta = mmax - mmin

    deltax = delta[0] / (nx - 1)
    deltay = delta[1] / (ny - 1)
    deltaz = delta[2] / (nz - 1)

    T = np.zeros(((nx - 1) * (ny - 1) * (nz - 1) * 6, 4), dtype=np.int64)
    V = np.zeros((nx * ny * nz, 3))

    mapping = -np.ones((nx, ny, nz), dtype=np.int64)

    index = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                mapping[i, j, k] = index
                V[index, :] = [i * deltax, j * deltay, k * deltaz]
                index += 1
    assert (index == V.shape[0])

    tets = np.array([
        [0, 1, 3, 4],
        [5, 2, 6, 7],
        [4, 1, 5, 3],
        [4, 3, 7, 5],
        [3, 1, 5, 2],
        [2, 3, 7, 5]
    ])

    index = 0
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                indices = [
                    (i, j, k),
                    (i + 1, j, k),
                    (i + 1, j + 1, k),
                    (i, j + 1, k),

                    (i, j, k + 1),
                    (i + 1, j, k + 1),
                    (i + 1, j + 1, k + 1),
                    (i, j + 1, k + 1),
                ]

                for t in range(tets.shape[0]):
                    tmp = [mapping[indices[ii]] for ii in tets[t, :]]
                    T[index, :] = tmp
                    index += 1

    assert (index == T.shape[0])

    V += mmin
    return V, T


pi, v = igl.read_triangle_mesh("data/cat.off")
pi /= 10
ni = igl.per_vertex_normals(pi, v)
mp.offline()
mp.plot(pi, shading={"point_size": 8})


def find_closed_point(point, points):
    distance=np.linalg.norm(points-point,axis=1)
    return np.argmin(distance)


eps=igl.bounding_box_diagonal(pi)*0.01
piplus=np.zeros_like(pi)
piminus=np.zeros_like(pi)
for (i,point) in enumerate(pi):
    temp=point+eps*ni[i]
    neweps=eps
    while find_closed_point(temp,pi)!=i:
        neweps=neweps/2
        temp=point+neweps*ni[i]
    piplus[i]=temp
for (i,point) in enumerate(pi):
    temp=point-eps*ni[i]
    neweps=eps
    while find_closed_point(temp,pi)!=i:
        neweps=neweps/2
        temp=point-neweps*ni[i]
    piminus[i]=temp


print(np.array([1,0,0]).size)
p=mp.plot(pi,c=np.array([1,0,0]), shading={"point_size": 8})
p.add_points(piplus,c=np.array([1,0,0]), shading={"point_size": 8})
p.add_points(piminus,c=np.array([0,1,0]), shading={"point_size": 8})