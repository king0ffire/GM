import numpy as np
import igl
import meshplot

bunny_v, bunny_f = igl.read_triangle_mesh("data/bunny.off")
cube_v, cube_f = igl.read_triangle_mesh("data/cube.obj")
sphere_v, sphere_f = igl.read_triangle_mesh("data/sphere.obj")

v,f=sphere_v, sphere_f
n=igl.per_face_normals(v,f,np.array([0.,0.,0.]))
colors = np.zeros((f.shape[0], 3))*2

explode_v=np.zeros((f.size,3))
explode_f=np.arange(f.size).reshape((-1,3))
explode_normals=np.zeros_like(explode_v)
for (i,k) in enumerate(f):
    for (j,l) in enumerate(k):
        explode_v[i*3+j]=v[l]
        explode_normals[i*3+j]=n[i]


print(explode_v.shape)
print(explode_normals.shape)
meshplot.offline()
meshplot.plot(explode_v, explode_f,n=explode_normals,shading={"flat": False,"colormap": 'Greys'})
#p.add_lines(explode_v, explode_v + 0.1*explode_normals)