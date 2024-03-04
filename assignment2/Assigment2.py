import numpy as np
import igl
import meshplot as mp


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


# Implementing a spatial index to accelerate neighbor calculations
def findgridcell(point, bbox_min, gridlength):  # bbox_min (1,3) ndarray. point is position
    return ((point - bbox_min) / gridlength).astype(np.int64)  # (,3) ndarray


def constructspatialgrid(points, bbox_min, bbox_max,
                         gridlength):  # points is the obj. Pure cube grid. should contain all obj&tet_grid points. length is determined by wendlandradius
    gridcellnumbers = findgridcell(bbox_max, bbox_min, gridlength) + 1
    gridcells = []  # shape=(gridcellsnumbers[0],gridcellsnumbers[1],gridcellsnumbers[2],*)
    for i in range(gridcellnumbers[0]):
        gridcells.append([])
        for j in range(gridcellnumbers[1]):
            gridcells[i].append([])
            for k in range(gridcellnumbers[2]):
                gridcells[i][j].append([])
    for (i, point) in enumerate(points):
        index = findgridcell(point, bbox_min, gridlength)
        gridcells[index[0]][index[1]][index[2]].append(i)
    return gridcells, gridcellnumbers


def find_closed_point_spatial(point, points, bbox_min, gridcells, gridcellnumbers,
                              gridlength, adjindices):  # point is position. points is positions of obj constraints
    cell = findgridcell(point, bbox_min, gridlength)
    adjcelllist = []
    for i, j, k in adjindices:
        i1, i2, i3 = cell[0] + i, cell[1] + j, cell[2] + k
        if 0 <= i1 < gridcellnumbers[0] and 0 <= i2 < gridcellnumbers[1] and 0 <= i3 < gridcellnumbers[2]:
            adjcelllist.extend(gridcells[i1][i2][i3])
    return adjcelllist[find_closed_point(point, points[adjcelllist])]


def closest_points_spatial(point, points, bbox_min, gridcells, gridcellnumbers, gridlength,
                           wendlandRadius, adjindices):  # point is position. Return like closest_points
    cell = findgridcell(point, bbox_min, gridlength)
    adjcelllist = []
    for i, j, k in adjindices:
        i1, i2, i3 = cell[0] + i, cell[1] + j, cell[2] + k
        if 0 <= i1 < gridcellnumbers[0] and 0 <= i2 < gridcellnumbers[1] and 0 <= i3 < gridcellnumbers[2]:
            adjcelllist.extend(gridcells[i1][i2][i3])
    if len(adjcelllist) == 0:
        return np.array([], dtype=np.int64)
    else:
        adjcelllist = np.array(adjcelllist)
        return adjcelllist[closest_points(point, points[adjcelllist], wendlandRadius)]


def normalize(points, min, max):
    return ((points - np.min(points, axis=0)) / (np.max(np.max(points, axis=0) - np.min(points, axis=0)))) * (
            max - min) - (max - min) / 2


def find_closed_point(point, points):  # assume the return is always a int
    distance = np.linalg.norm(points - point, axis=1)
    return np.argmin(distance)


def closest_points(point, points, h):
    distance = np.linalg.norm(points - point, axis=1)
    res = np.argwhere(distance < h)
    return res.squeeze(axis=1)


def wendlandweight(p1, p2, h):
    distance = np.linalg.norm(p1 - p2)
    return (1 - distance / h) ** 4 * (4 * distance / h + 1)


def functioninvector_k2(x, y, z):
    return np.array([1, x, y, z, x ** 2, x * y, x * z, y ** 2, y * z, z ** 2])


def functioninvector_k1(x, y, z):
    return np.array([1, x, y, z])


def functioninvector_k0(x, y, z):
    return np.array([1])


def functioninvector(x, y, z, k):
    if k == 2:
        return functioninvector_k2(x, y, z)
    elif k == 1:
        return functioninvector_k1(x, y, z)
    elif k == 0:
        return functioninvector_k0(x, y, z)


def readMesh(filelocation):
    pi, v = igl.read_triangle_mesh(filelocation)
    pi /= 10
    #ni = igl.per_vertex_normals(pi, v)
    return pi, v


def normalizeMesh(v,normalizationmethod, pi, spatial, gridlength):
    if normalizationmethod == 0:
        bbox_min = np.array([-1., -1., -1.])
        bbox_max = np.array([1., 1., 1.])
        bbox_diag = np.linalg.norm(bbox_max - bbox_min)
        npi = normalize(pi, np.min(bbox_min), np.max(bbox_max))
        bbox_min = bbox_min - 0.05 * bbox_diag
        bbox_max = bbox_max + 0.05 * bbox_diag
    if normalizationmethod == 1:
        bv, bf = igl.bounding_box(pi)
        bbox_min = np.min(np.array(bv), axis=0)
        bbox_max = np.max(np.array(bv), axis=0)
        bbox_diag = np.linalg.norm(bbox_max - bbox_min)
        bbox_min = bbox_min - 0.05 * bbox_diag
        bbox_max = bbox_max + 0.05 * bbox_diag
        npi = pi
    if normalizationmethod == 2:
        centered = pi - np.mean(pi, axis=0)
        cov = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        if np.linalg.det(sorted_eigenvectors) < 0:
            sorted_eigenvectors[:, 0] *= -1
        npi = centered @ sorted_eigenvectors
        bv, bf = igl.bounding_box(npi)
        bbox_min = np.min(np.array(bv), axis=0)
        bbox_max = np.max(np.array(bv), axis=0)
        bbox_diag = np.linalg.norm(bbox_max - bbox_min)
        bbox_min = bbox_min - 0.05 * bbox_diag
        bbox_max = bbox_max + 0.05 * bbox_diag

    ni = igl.per_vertex_normals(npi, v)
    if spatial == True:
        gridcells, gridcellnumbers = constructspatialgrid(npi, bbox_min, bbox_max, gridlength)
    else:
        gridcells = None
        gridcellnumbers = None
    return npi, bbox_min, bbox_max, gridcells, gridcellnumbers,ni


def constructConstraint(npi, ni, spatial, bbox_min, gridcells, gridcellnumbers, gridlength,adjindices):
    eps = igl.bounding_box_diagonal(npi) * 0.01
    piplus = np.zeros_like(npi)
    piminus = np.zeros_like(npi)
    for (i, point) in enumerate(npi):
        temp = point + eps * ni[i]
        neweps = eps
        while True:
            if spatial == False:
                closest = find_closed_point(temp, npi)
            else:
                closest = find_closed_point_spatial(temp, npi, bbox_min, gridcells, gridcellnumbers, gridlength,adjindices)
            if closest == i:
                break
            neweps = neweps / 2
            temp = point + neweps * ni[i]
        piplus[i] = temp
    for (i, point) in enumerate(npi):
        temp = point - eps * ni[i]
        neweps = eps
        while True:
            if spatial == False:
                closest = find_closed_point(temp, npi)
            else:
                closest = find_closed_point_spatial(temp, npi, bbox_min, gridcells, gridcellnumbers, gridlength,adjindices)
            if closest == i:
                break
            neweps = neweps / 2
            temp = point - neweps * ni[i]
        piminus[i] = temp
    npiplus = piplus
    npiminus = piminus
    return npiplus, npiminus


def calculatefx(x, npi, npiplus, npiminus, ni, bbox_min, coefficientnumber, bbox_max, gridlength, spatial,
                wendlandRadius, k,adjindices):
    funcargs = np.zeros((x.shape[0], coefficientnumber))  # 1,x,y,z,x^2,xy,xz,y^2,yz,z^2
    points = np.concatenate((npi, npiplus, npiminus))
    if spatial == True:
        gridcells, gridcellnumbers = constructspatialgrid(points, bbox_min, bbox_max, gridlength)
    fx = np.zeros((x.shape[0], 1))
    for (i, xi) in enumerate(x):
        if spatial == False:
            adjlist = np.array(closest_points(xi, points, wendlandRadius))
        else:
            adjlist = np.array(
                closest_points_spatial(xi, points, bbox_min, gridcells, gridcellnumbers, gridlength, wendlandRadius,adjindices))
        a = np.zeros((adjlist.size, coefficientnumber))
        b = np.zeros((adjlist.size, 1))
        w = np.zeros((adjlist.size, adjlist.size))  # diagonal matrix
        if (len(adjlist) < coefficientnumber):
            fx[i] = 10000
            continue
        for (j, adj) in enumerate(adjlist):
            adj_position = points[adj]
            originalpiindex = adj % (npi.shape[0])
            originalpi_position = points[originalpiindex]
            eps = (adj_position - originalpi_position)[0] / ni[originalpiindex, 0]
            a[j, :] = functioninvector(adj_position[0], adj_position[1], adj_position[2], k)
            b[j] = eps  # what would be range?
            w[j, j] = wendlandweight(adj_position, xi, wendlandRadius)
        funcargs[i] = np.linalg.solve(a.T @ w @ a, a.T @ w @ b).T
        fx[i] = (functioninvector(xi[0], xi[1], xi[2], k) @ funcargs[i]).T
    return fx, funcargs


def virtualizeinout(fx):
    ind = np.zeros_like(fx)
    ind[fx >= 0] = 1  # yellow
    ind[fx < 0] = -1  # black
    return ind


def marching(x, T, fx):
    sv, sf, _, _ = igl.marching_tets(x, T, fx, 0)
    components = igl.facet_components(sf)
    index, count = np.unique(components, return_counts=True)
    index = np.where(count == np.max(count))[0][0]  # argwhere?
    filteredf = sf[components == index]
    return sv, filteredf


def MyWholeConstructionInOneFunc(filelocation, normalizationmethod, spatial, plotconstrain, plotinout, n,
                                 point_size, plotgrid, coefficientnumber, plotreconstructed, gridlength, k,
                                 wendlandRadius,adjindices):  # functioninvector
    pi, v= readMesh(filelocation)
    npi, bbox_min, bbox_max, gridcells, gridcellnumbers, ni = normalizeMesh(v,normalizationmethod, pi, spatial, gridlength)
    npiplus, npiminus = constructConstraint(npi, ni, spatial, bbox_min, gridcells, gridcellnumbers, gridlength,adjindices)
    if plotconstrain == True:
        print("The following plot is the required output for Setting up the Constraints section")
        p = mp.plot(npi, c=np.ones_like(npi) * np.array([0, 0, 1]), shading={"point_size": point_size})
        p.add_points(npiplus, c=np.ones_like(npi) * np.array([1, 0, 0]), shading={"point_size": point_size})
        p.add_points(npiminus, c=np.ones_like(npi) * np.array([0, 1, 0]), shading={"point_size": point_size})
    x, T = tet_grid([n[0], n[1], n[2]], bbox_min, bbox_max)
    if plotgrid == True:
        print("The following plot is the required output for Using a non-axis-aligned grid section")
        p = mp.plot(npi, v)
        p.add_points(x, shading={"point_size": point_size})
    fx, funcargs = calculatefx(x, npi, npiplus, npiminus, ni, bbox_min, coefficientnumber, bbox_max, gridlength,
                               spatial, wendlandRadius, k,adjindices)
    ind = virtualizeinout(fx)
    if plotinout == True:
        print("The following plot is the required output for \"Use MLS interpolation to extend to function f\" section")
        p = mp.plot(x, c=ind, shading={"point_size": point_size, "width": 800, "height": 800})
    sv, filteredf = marching(x, T, fx)
    if plotreconstructed == True:
        mp.plot(sv, filteredf, shading={"wireframe": True})

