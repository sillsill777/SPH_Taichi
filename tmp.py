import taichi as ti

ti.init(arch=ti.gpu)
a=1
@ti.kernel
def func():
   print(a)

@ti.kernel
def func2():
    print(a)

func()
a=2
func()
func2()


c=ti.Vector.field(3,dtype=int,shape=2)
c[0]=[1,1,1]

@ti.kernel
def ff():
    print(c[0])

@ti.kernel
def fff():
    print(c[0])

ff()
c[0][0]=100
ff()
fff()



b=ti.field(int,shape=())



@ti.kernel
def func3():
    print(b[None])
    b=15

@ti.kernel
def func4():
    print(b[None])

func3()
b[None]=10
func3()
func4()

@ti.kernel
def ffff():
    ttmp = ti.math.vec3(1,1,1)

    print(ttmp)

ffff()


class BaseClass:
    def __init__(self):
        self.n = 10
        self.num = ti.field(dtype=ti.i32, shape=(self.n, ))

    @ti.kernel
    def count(self) -> ti.i32:
        ret = 0
        for i in range(self.n):
            ret += self.num[i]
        return ret

    @ti.kernel
    def add(self, d: ti.i32):
        for i in range(self.n):
            self.num[i] += d


@ti.data_oriented
class DataOrientedClass(BaseClass):
    pass

class DeviatedClass(DataOrientedClass):
    @ti.kernel
    def sub(self, d: ti.i32):
        for i in range(self.n):
            self.num[i] -= d


a = DeviatedClass()
a.add(1)
a.sub(1)
print(a.count())  # 0


b = DataOrientedClass()
b.add(3)
print(b.count())  # 30


@ti.data_oriented
class Counter:
    num_ = ti.field(dtype=ti.i32, shape=(32, ))
    def __init__(self, data_range):
        self.range = data_range
        self.add(data_range[0], data_range[1], 1)

    @classmethod
    @ti.kernel
    def add(cls, l: ti.i32, r: ti.i32, d: ti.i32):
        for i in range(l, r):
            cls.num_[i] += d

    @ti.kernel
    def num(self) -> ti.i32:
        ret = 0
        for i in range(self.range[0], self.range[1]):
            ret += self.num_[i]
        return ret

a = Counter((0, 5))
print(a.num())  # 5
print(a.num())
b = Counter((3, 10))
print(a.num())  # 6
print(b.num())  # 7