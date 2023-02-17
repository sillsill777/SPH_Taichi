import taichi as ti

ti.init(arch=ti.gpu)

a=ti.field(dtype=ti.i32,shape=10)
b=ti.field(dtype=ti.i32,shape=10)

c=ti.Vector.field(3, dtype=ti.f32, shape=4)
d=ti.Vector.field(3, dtype=ti.f32, shape=4)

@ti.kernel
def func():
    for i in a:
        print(i)
        a[i] = int(i+2)
    for i in ti.grouped(b):
        print(i)
        b[i] = 3


@ti.kernel
def ff():
    for i in ti.grouped(a):
        print(a[i])


func()
ff()
print("---------------------------")
@ti.kernel
def fff():
    for i in c:
        print(i)
        c[i]=ti.Vector([1.,2.,3.])
    for j in ti.grouped(d):
        print(j)
        d[j]=ti.Vector([4.,5.,6.])
@ti.kernel
def f():
    for i in ti.grouped(c):
        print(c[i])
    for j in d:
        print(d[j])
fff()
f()
print("========================================")
grid_size=2
@ti.func
def pos_to_index(pos):
    return (pos / grid_size).cast(int)


print(a.shape[0])
print("======================")
ff()
exe=ti.algorithms.PrefixSumExecutor(a.shape[0])

exe.run(a)
print("======================")
ff()

@ti.kernel
def aa():
    for i in c:
        print(pos_to_index(c[i]))

aa()
z=ti.cast(0.0,ti.f32)
print(z)
