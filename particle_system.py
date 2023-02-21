import taichi as ti
import numpy as np
import trimesh as tm
from functools import reduce
from config_builder import SimConfig
from WCSPH import WCSPHSolver
from DFSPH import DFSPHSolver
from scan_single_buffer import parallel_prefix_sum_inclusive_inplace


@ti.data_oriented
class ParticleSystem:
    def __init__(self, config: SimConfig, GGUI=False):
        self.cfg = config
        self.GGUI = GGUI

        self.domain_start = np.array([0.0, 0.0, 0.0])
        self.domain_start = np.array(self.cfg.get_cfg("domainStart"))

        self.domain_end = np.array([1.0, 1.0, 1.0])
        self.domian_end = np.array(self.cfg.get_cfg("domainEnd"))  # [5, 3, 2]

        self.domain_size = self.domian_end - self.domain_start

        self.dim = len(self.domain_size)
        assert self.dim > 1
        # Simulation method
        self.simulation_method = self.cfg.get_cfg("simulationMethod")  # 0

        # Material
        self.material_solid = 0
        self.material_fluid = 1

        self.particle_radius = 0.01  # particle radius
        self.particle_radius = self.cfg.get_cfg("particleRadius")  # 0.01

        self.particle_diameter = 2 * self.particle_radius  # 2r
        self.support_radius = self.particle_radius * 4.0  # support radius
        self.m_V0 = 0.8 * self.particle_diameter ** self.dim  # 0.8*(diameter**3)
        # (4/3)*pi/8=0.52...  m_V0 -> volume with margin?

        self.particle_num = ti.field(int, shape=())

        # Grid related properties
        self.grid_size = self.support_radius  # 4r
        self.grid_num = np.ceil(self.domain_size / self.grid_size).astype(int)
        # [125, 75, 50]
        '''
        a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
        np.ceil(a)
        array([-1., -1., -0.,  1.,  2.,  2.,  2.])
        '''
        print("grid size: ", self.grid_num)  # [125, 75, 50]
        self.padding = self.grid_size  # 4r

        # All objects id and its particle num
        self.object_collection = dict()
        self.object_id_rigid_body = set()

        # ========== Compute number of particles ==========#
        #### Process Fluid Blocks ####
        fluid_blocks = self.cfg.get_fluid_blocks()
        fluid_particle_num = 0
        for fluid in fluid_blocks:
            particle_num = self.compute_cube_particle_num(fluid["start"], fluid["end"])
            """
            fluid['start'] , fluid['end'] 로 정의되는 직육면체에서
            각 그리드의 변의 길이를 particle의 지름으로 잡았을 때 그런 그리드 정육면체가 몇개가 들어가는지 
            """

            fluid["particleNum"] = particle_num  # 423,500
            print('fluid particla num: ', particle_num)
            self.object_collection[fluid["objectId"]] = fluid
            fluid_particle_num += particle_num

        #### Process Rigid Blocks #### -> Not in this example(dragon bath)
        rigid_blocks = self.cfg.get_rigid_blocks()  # []

        rigid_particle_num = 0
        for rigid in rigid_blocks:
            particle_num = self.compute_cube_particle_num(rigid["start"], rigid["end"])
            rigid["particleNum"] = particle_num
            self.object_collection[rigid["objectId"]] = rigid
            rigid_particle_num += particle_num

        #### Process Rigid Bodies ####
        rigid_bodies = self.cfg.get_rigid_bodies()
        for rigid_body in rigid_bodies:
            voxelized_points_np = self.load_rigid_body(rigid_body)
            rigid_body["particleNum"] = voxelized_points_np.shape[0]
            rigid_body["voxelizedPoints"] = voxelized_points_np
            self.object_collection[rigid_body["objectId"]] = rigid_body
            rigid_particle_num += voxelized_points_np.shape[0]

        self.fluid_particle_num = fluid_particle_num
        self.solid_particle_num = rigid_particle_num
        self.particle_max_num = fluid_particle_num + rigid_particle_num  # total particle num(fluid+solid)
        self.num_rigid_bodies = len(rigid_blocks) + len(rigid_bodies)

        #### TODO: Handle the Particle Emitter ####
        # self.particle_max_num += emitted particles
        print(f"Current particle num: {self.particle_num[None]}, Particle max num: {self.particle_max_num}")

        # ========== Allocate memory ==========#
        # Rigid body properties
        if self.num_rigid_bodies > 0:
            # TODO: Here we actually only need to store rigid boides, however the object id of rigid may not start from 0, so allocate center of mass for all objects
            self.rigid_rest_cm = ti.Vector.field(self.dim, dtype=float, shape=self.num_rigid_bodies + len(fluid_blocks))

        # Particle num of each grid###################################################################################
        """
        # Grid related properties
        self.grid_size = self.support_radius  # 4r
        self.grid_num = np.ceil(self.domain_size / self.grid_size).astype(int)
        # [125, 75, 50]
        """

        self.grid_particles_num = ti.field(int, shape=int(self.grid_num[0] * self.grid_num[1] * self.grid_num[2]))
        self.grid_particles_num_temp = ti.field(int, shape=int(self.grid_num[0] * self.grid_num[1] * self.grid_num[2]))
        # 468,750

        self.prefix_sum_executor = ti.algorithms.PrefixSumExecutor(self.grid_particles_num.shape[0])
        ##############################################################################################################
        # Particle related properties################################################################################
        self.object_id = ti.field(dtype=int, shape=self.particle_max_num)  # fluid+rigid body

        self.x = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.x_0 = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.v = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.acceleration = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)

        self.m_V = ti.field(dtype=float, shape=self.particle_max_num)
        self.m = ti.field(dtype=float, shape=self.particle_max_num)
        self.density = ti.field(dtype=float, shape=self.particle_max_num)
        self.pressure = ti.field(dtype=float, shape=self.particle_max_num)
        self.material = ti.field(dtype=int, shape=self.particle_max_num)  # int

        self.color = ti.Vector.field(3, dtype=int, shape=self.particle_max_num)  # int
        self.is_dynamic = ti.field(dtype=int, shape=self.particle_max_num)  # int
        # ------------------------------------- particle related properties total 12 of them

        if self.cfg.get_cfg("simulationMethod") == 4:  # Not in dragon_bath simulation
            self.dfsph_factor = ti.field(dtype=float, shape=self.particle_max_num)
            self.density_adv = ti.field(dtype=float, shape=self.particle_max_num)

        # Buffer for sort###################################################################################
        self.object_id_buffer = ti.field(dtype=int, shape=self.particle_max_num)

        self.x_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.x_0_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.v_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.acceleration_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)

        self.m_V_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.m_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.density_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.pressure_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.material_buffer = ti.field(dtype=int, shape=self.particle_max_num)

        self.color_buffer = ti.Vector.field(3, dtype=int, shape=self.particle_max_num)
        self.is_dynamic_buffer = ti.field(dtype=int, shape=self.particle_max_num)

        if self.cfg.get_cfg("simulationMethod") == 4:
            self.dfsph_factor_buffer = ti.field(dtype=float, shape=self.particle_max_num)
            self.density_adv_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        #######################################################################################################

        # Grid id for each particle
        self.grid_ids = ti.field(int, shape=self.particle_max_num)
        self.grid_ids_buffer = ti.field(int, shape=self.particle_max_num)
        self.grid_ids_new = ti.field(int, shape=self.particle_max_num)

        self.x_vis_buffer = None
        if self.GGUI:
            self.x_vis_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
            self.color_vis_buffer = ti.Vector.field(3, dtype=float, shape=self.particle_max_num)

        # ========== Initialize particles ==========#

        # Fluid block
        for fluid in fluid_blocks:
            obj_id = fluid["objectId"]
            offset = np.array(fluid["translation"])
            start = np.array(fluid["start"]) + offset
            end = np.array(fluid["end"]) + offset
            scale = np.array(fluid["scale"])  # [1,1,1]
            velocity = fluid["velocity"]  # [0,-1,0]
            density = fluid["density"]  # 1000
            color = fluid["color"]  # [50, 100, 200]
            self.add_cube(object_id=obj_id,
                          lower_corner=start,
                          cube_size=(end - start) * scale,
                          velocity=velocity,
                          density=density,
                          is_dynamic=1,  # enforce fluid dynamic
                          color=color,
                          material=1)  # 1 indicates fluid

        # TODO: Handle rigid block
        # Rigid block  Not in this example(dragon bath)
        for rigid in rigid_blocks:
            obj_id = rigid["objectId"]
            offset = np.array(rigid["translation"])
            start = np.array(rigid["start"]) + offset
            end = np.array(rigid["end"]) + offset
            scale = np.array(rigid["scale"])
            velocity = rigid["velocity"]
            density = rigid["density"]
            color = rigid["color"]
            is_dynamic = rigid["isDynamic"]
            self.add_cube(object_id=obj_id,
                          lower_corner=start,
                          cube_size=(end - start) * scale,
                          velocity=velocity,
                          density=density,
                          is_dynamic=is_dynamic,
                          color=color,
                          material=0)  # 1 indicates solid

        # Rigid bodies
        for rigid_body in rigid_bodies:
            obj_id = rigid_body["objectId"]
            self.object_id_rigid_body.add(obj_id)
            num_particles_obj = rigid_body["particleNum"]
            voxelized_points_np = rigid_body["voxelizedPoints"]
            is_dynamic = rigid_body["isDynamic"]  # false
            if is_dynamic:
                velocity = np.array(rigid_body["velocity"], dtype=np.float32)
            else:
                velocity = np.array([0.0 for _ in range(self.dim)], dtype=np.float32)
            density = rigid_body["density"]  # 1000
            color = np.array(rigid_body["color"], dtype=np.int32)  # [255,255,255]
            self.add_particles(obj_id,
                               num_particles_obj,
                               np.array(voxelized_points_np, dtype=np.float32),  # position
                               np.stack([velocity for _ in range(num_particles_obj)]),  # velocity
                               density * np.ones(num_particles_obj, dtype=np.float32),  # density
                               np.zeros(num_particles_obj, dtype=np.float32),  # pressure
                               np.array([0 for _ in range(num_particles_obj)], dtype=np.int32),  # material is solid
                               is_dynamic * np.ones(num_particles_obj, dtype=np.int32),  # is_dynamic=0
                               np.stack([color for _ in range(num_particles_obj)]))  # color

    def build_solver(self):  # WCSPH
        solver_type = self.cfg.get_cfg("simulationMethod")
        if solver_type == 0:
            return WCSPHSolver(self)
        elif solver_type == 4:
            return DFSPHSolver(self)
        else:
            raise NotImplementedError(f"Solver type {solver_type} has not been implemented.")

    @ti.func
    def add_particle(self, p, obj_id, x, v, density, pressure, material, is_dynamic, color):
        """
        self.add_particle(p, object_id, x, v,
                          new_particle_density[p - self.particle_num[None]],
                          new_particle_pressure[p - self.particle_num[None]],
                          new_particles_material[p - self.particle_num[None]],
                          new_particles_is_dynamic[p - self.particle_num[None]],
                          ti.Vector([new_particles_color[p - self.particle_num[None], i] for i in range(3)])
                          )
        """
        self.object_id[p] = obj_id
        self.x[p] = x  # Vec
        self.x_0[p] = x  # Vec
        self.v[p] = v  # Vec
        self.density[p] = density
        self.m_V[p] = self.m_V0
        """
        self.m_V0 = 0.8 * self.particle_diameter ** self.dim  # 0.8*(diameter**3)
        # (4/3)*pi/8=0.52...  m_V0 -> volume with margin?
        """
        self.m[p] = self.m_V0 * density
        self.pressure[p] = pressure
        self.material[p] = material
        self.is_dynamic[p] = is_dynamic
        self.color[p] = color  # Vec
        # acceleration is not here.

    def add_particles(self,
                      object_id: int,
                      new_particles_num: int,
                      new_particles_positions: ti.types.ndarray(),
                      new_particles_velocity: ti.types.ndarray(),
                      new_particle_density: ti.types.ndarray(),
                      new_particle_pressure: ti.types.ndarray(),
                      new_particles_material: ti.types.ndarray(),
                      new_particles_is_dynamic: ti.types.ndarray(),
                      new_particles_color: ti.types.ndarray()
                      ):

        self._add_particles(object_id,
                            new_particles_num,
                            new_particles_positions,
                            new_particles_velocity,
                            new_particle_density,
                            new_particle_pressure,
                            new_particles_material,
                            new_particles_is_dynamic,
                            new_particles_color
                            )

    # directly calling self._add_particles also works. Why using add_particles???

    @ti.kernel
    def _add_particles(self,
                       object_id: int,
                       new_particles_num: int,
                       new_particles_positions: ti.types.ndarray(),
                       new_particles_velocity: ti.types.ndarray(),
                       new_particle_density: ti.types.ndarray(),
                       new_particle_pressure: ti.types.ndarray(),
                       new_particles_material: ti.types.ndarray(),
                       new_particles_is_dynamic: ti.types.ndarray(),
                       new_particles_color: ti.types.ndarray()):
        for p in range(self.particle_num[None], self.particle_num[None] + new_particles_num):  # [0, 423500)
            v = ti.Vector.zero(float, self.dim)
            x = ti.Vector.zero(float, self.dim)
            """
            v = ti.Vector([0.,0.,0.])
            x = ti.Vector([0.,0.,0.])
            -----------------------------------also works-----------
            """
            for d in ti.static(range(self.dim)):
                v[d] = new_particles_velocity[p - self.particle_num[None], d]
                x[d] = new_particles_positions[p - self.particle_num[None], d]
            """
            But instead of upper for loop to initialize v and x. It is impossible to do it like below
            
            v= new_particles_velocity[p - self.particle_num[None]]  -> new_particle_velocity has dim=2 
            but in indexing, only 1 dimension info is given
            
            Also v =  new_particles_velocity[p - self.particle_num[None], :] -> error
            since slicing is not supported.
            """
            self.add_particle(p, object_id, x, v,
                              new_particle_density[p - self.particle_num[None]],
                              new_particle_pressure[p - self.particle_num[None]],
                              new_particles_material[p - self.particle_num[None]],
                              new_particles_is_dynamic[p - self.particle_num[None]],
                              ti.Vector([new_particles_color[p - self.particle_num[None], i] for i in range(3)])
                              )
        self.particle_num[None] += new_particles_num

    @ti.func
    def pos_to_index(self, pos):
        return (pos / self.grid_size).cast(int)  # floor

    @ti.func
    def flatten_grid_index(self, grid_index):
        return grid_index[0] * self.grid_num[1] * self.grid_num[2] + grid_index[1] * self.grid_num[2] + grid_index[2]

    @ti.func
    def get_flatten_grid_index(self, pos):
        return self.flatten_grid_index(self.pos_to_index(pos))

    @ti.func
    def is_static_rigid_body(self, p):
        """
        # Material
        self.material_solid = 0
        self.material_fluid = 1
        """
        return self.material[p] == self.material_solid and (not self.is_dynamic[p])

    @ti.func
    def is_dynamic_rigid_body(self, p):
        return self.material[p] == self.material_solid and self.is_dynamic[p]

    @ti.kernel
    def update_grid_id(self):
        for I in ti.grouped(self.grid_particles_num):
            self.grid_particles_num[I] = 0
        for I in ti.grouped(self.x):
            """
            self.grid_ids = ti.field(int, shape=self.particle_max_num)
            self.grid_particles_num = ti.field(int, shape=int(self.grid_num[0] * self.grid_num[1] * self.grid_num[2]))
            """
            grid_index = self.get_flatten_grid_index(self.x[I])
            self.grid_ids[I] = grid_index
            ti.atomic_add(self.grid_particles_num[grid_index], 1)
        for I in ti.grouped(self.grid_particles_num):
            self.grid_particles_num_temp[I] = self.grid_particles_num[I]

    @ti.kernel
    def counting_sort(self):
        # FIXME: make it the actual particle num
        for i in range(self.particle_max_num):
            I = self.particle_max_num - 1 - i
            base_offset = 0
            if self.grid_ids[I] - 1 >= 0:
                base_offset = self.grid_particles_num[self.grid_ids[I] - 1]
            self.grid_ids_new[I] = ti.atomic_sub(self.grid_particles_num_temp[self.grid_ids[I]], 1) - 1 + base_offset

            """
            'grid_ids[i]' contains for every particle i in the simulator(particle_max_num), what grid index 
            that the particle i belongs to.
            For example grid_ids=[1,0,1,3,2,2,2]-> particle 0 belongs to grid index 1,
            particle 1 belongs to grid index 0, particle 2 belongs to grid index 1 ...
            
            grid index 0 -> there is one particle exists.
            grid index 1-> there are two particle exists.
            grid index 2-> there are three particle exists.
            grid index 3-> there is one particle exists.
            
            If we want to sort grid_id then result will be [0,1,1,2,2,2,3]. To achieve this,
            particle 0 must go to index 1, particle 1 must go to index 0, particle 2 must go to index 2,
            particle 3 must go to 6, particle 4 must go to 3, ... ->[1,0,2,6,3,4,5]
            
            where as 'grid_ids_new[i]' contains for every particle i in the simulator, what index should it be
            for grid_ids to be sorted. Above example it will be 'grid_idx_new'=[1,0,2,6,3,4,5]
            
            """

        for I in ti.grouped(self.grid_ids):
            new_index = self.grid_ids_new[I]
            self.grid_ids_buffer[new_index] = self.grid_ids[I]
            self.object_id_buffer[new_index] = self.object_id[I]
            self.x_0_buffer[new_index] = self.x_0[I]
            self.x_buffer[new_index] = self.x[I]
            self.v_buffer[new_index] = self.v[I]
            self.acceleration_buffer[new_index] = self.acceleration[I]
            self.m_V_buffer[new_index] = self.m_V[I]
            self.m_buffer[new_index] = self.m[I]
            self.density_buffer[new_index] = self.density[I]
            self.pressure_buffer[new_index] = self.pressure[I]
            self.material_buffer[new_index] = self.material[I]
            self.color_buffer[new_index] = self.color[I]
            self.is_dynamic_buffer[new_index] = self.is_dynamic[I]

            if ti.static(self.simulation_method == 4):
                self.dfsph_factor_buffer[new_index] = self.dfsph_factor[I]
                self.density_adv_buffer[new_index] = self.density_adv[I]

        for I in ti.grouped(self.x):
            self.grid_ids[I] = self.grid_ids_buffer[I]
            self.object_id[I] = self.object_id_buffer[I]
            self.x_0[I] = self.x_0_buffer[I]
            self.x[I] = self.x_buffer[I]
            self.v[I] = self.v_buffer[I]
            self.acceleration[I] = self.acceleration_buffer[I]
            self.m_V[I] = self.m_V_buffer[I]
            self.m[I] = self.m_buffer[I]
            self.density[I] = self.density_buffer[I]
            self.pressure[I] = self.pressure_buffer[I]
            self.material[I] = self.material_buffer[I]
            self.color[I] = self.color_buffer[I]
            self.is_dynamic[I] = self.is_dynamic_buffer[I]

            if ti.static(self.simulation_method == 4):
                self.dfsph_factor[I] = self.dfsph_factor_buffer[I]
                self.density_adv[I] = self.density_adv_buffer[I]
        """
        There are total particle_max_num particles(0, 1, 2, ... particle_max_num-1)
        And each particle has property grid_ids, object_id, x_0, x, v, acceleration ...
        Now due to counting sort, grid_ids is sorted. For example grid_ids=[0,0,0,1,1,2,3]
        -> particle 0,1,2 belongs to grid index 0, particle 3,4 belongs to grid index 1,
        particle 5 belongs to grid index 2, particle 6 belongs to grid index 3.
        
        particle 0 has property object_id[0], x_0[0], x[0], v[0], acc[0], m_V[0], m[0],... with grid index 0
        particle 1 has property object_id[1], x_0[1], x[1], v[1], acc[1], m_V[1], m[1],... with grid index 0
        particle 2 has property object_id[2], x_0[2], x[2], v[2], acc[2], m_V[2], m[2],... with grid index 0
        
        particle 3 has property object_id[3], x_0[3], x[3], v[3], acc[3], m_V[3], m[3],... with grid index 1
        particle 4 has property object_id[4], x_0[4], x[4], v[4], acc[4], m_V[4], m[4],... with grid index 1
        
        particle 5 has property object_id[5], x_0[5], x[5], v[5], acc[5], m_V[5], m[5],... with grid index 2
        
        particle 6 has property object_id[6], x_0[6], x[6], v[6], acc[6], m_V[6], m[6],... with grid index 3
        """
    def initialize_particle_system(self):
        self.update_grid_id()
        self.prefix_sum_executor.run(self.grid_particles_num)
        self.counting_sort()

    @ti.func
    def for_all_neighbors(self, p_i, task: ti.template(), ret: ti.template()):
        center_cell = self.pos_to_index(self.x[p_i])
        for offset in ti.grouped(ti.ndrange(*(((-1, 2),) * self.dim))):
            """
            -offset
            [-1, -1, -1]
            [-1, -1, 0]
            [-1, -1, 1]
            [-1, 0, -1]
            [-1, 0, 0]
            [-1, 0, 1]
            [-1, 1, -1]
            ...
            [1, 0, 1]
            [1, 1, -1]
            [1, 1, 0]
            [1, 1, 1]
            """
            grid_index = self.flatten_grid_index(center_cell + offset)
            start_idx=0 if grid_index==0 else self.grid_particles_num[grid_index-1]
            for p_j in range(start_idx, self.grid_particles_num[grid_index]):
                if p_i[0] != p_j and (self.x[p_i] - self.x[p_j]).norm() < self.support_radius:
                    task(p_i, p_j, ret)

    @ti.kernel
    def copy_to_numpy(self, np_arr: ti.types.ndarray(), src_arr: ti.template()):
        for i in range(self.particle_num[None]):
            np_arr[i] = src_arr[i]

    def copy_to_vis_buffer(self, invisible_objects=[]):
        if len(invisible_objects) != 0:
            self.x_vis_buffer.fill(0.0)
            self.color_vis_buffer.fill(0.0)
        for obj_id in self.object_collection:
            if obj_id not in invisible_objects:
                self._copy_to_vis_buffer(obj_id)

    @ti.kernel
    def _copy_to_vis_buffer(self, obj_id: int):
        assert self.GGUI
        # FIXME: make it equal to actual particle num
        for i in range(self.particle_max_num):
            if self.object_id[i] == obj_id:
                self.x_vis_buffer[i] = self.x[i]
                self.color_vis_buffer[i] = self.color[i] / 255.0

    def dump(self, obj_id):
        np_object_id = self.object_id.to_numpy()
        mask = (np_object_id == obj_id).nonzero()
        np_x = self.x.to_numpy()[mask]
        np_v = self.v.to_numpy()[mask]

        return {
            'position': np_x,
            'velocity': np_v
        }

    def load_rigid_body(self, rigid_body):
        obj_id = rigid_body["objectId"]
        mesh = tm.load(rigid_body["geometryFile"])
        mesh.apply_scale(rigid_body["scale"])
        # 각 25007 개의 꼭지점 좌표에 해당 스케일 값을 곱함
        offset = np.array(rigid_body["translation"])

        angle = rigid_body["rotationAngle"] / 360 * 2 * 3.1415926
        direction = rigid_body["rotationAxis"]
        rot_matrix = tm.transformations.rotation_matrix(angle, direction, mesh.vertices.mean(axis=0))
        """
        rotation_matrix(angle, direction, point=None)
        Return matrix to rotate about axis defined by point and
        direction.
        
        Parameters
        -------------
        angle     : float, or sympy.Symbol
          Angle, in radians or symbolic angle
        direction : (3,) float
          Unit vector along rotation axis
        point     : (3, ) float, or None
          Origin point of rotation axis
        """
        mesh.apply_transform(rot_matrix)
        mesh.vertices += offset

        # Backup the original mesh for exporting obj
        mesh_backup = mesh.copy()
        rigid_body["mesh"] = mesh_backup
        rigid_body["restPosition"] = mesh_backup.vertices
        rigid_body["restCenterOfMass"] = mesh_backup.vertices.mean(axis=0)
        # is_success = tm.repair.fill_holes(mesh)
        # print("Is the mesh successfully repaired? ", is_success)  # false
        voxelized_mesh = mesh.voxelized(pitch=self.particle_diameter)
        voxelized_mesh = mesh.voxelized(pitch=self.particle_diameter).fill()
        # voxelized_mesh = mesh.voxelized(pitch=self.particle_diameter).hollow()
        voxelized_points_np = voxelized_mesh.points
        print(f"rigid body {obj_id} num: {voxelized_points_np.shape[0]}")

        return voxelized_points_np  # (18496,3)

    def compute_cube_particle_num(self, start, end):
        num_dim = []
        for i in range(self.dim):
            num_dim.append(
                np.arange(start[i], end[i], self.particle_diameter))  # end[i] is not included but start[i] is included
        print('num_dim: ', [len(n) for n in num_dim])  # [55,140,55]
        return reduce(lambda x, y: x * y,
                      [len(n) for n in num_dim])

    def add_cube(self,
                 object_id,
                 lower_corner,
                 cube_size,
                 material,
                 is_dynamic,
                 color=(0, 0, 0),
                 density=None,
                 pressure=None,
                 velocity=None):

        """
        :param object_id: 0
        :param lower_corner: start
        :param cube_size: (end-start)*scale
        :param material: 1 -> 1 indicate fluid
        :param is_dynamic: 1 -> enforce fluid dynamic
        :param color: [50,100,200]
        :param density: 1000.0
        :param pressure: None
        :param velocity: [0,-1,0]
        :return:
        """

        num_dim = []
        for i in range(self.dim):
            num_dim.append(
                np.arange(lower_corner[i], lower_corner[i] + cube_size[i],
                          self.particle_diameter))
        num_new_particles = reduce(lambda x, y: x * y,
                                   [len(n) for n in num_dim])  # if scale=1 then same as compute_cube_particle_num()
        print('particle num ', num_new_particles)  # 423,500
        # num_dim's element num -> [55,140,55]

        new_positions = np.array(np.meshgrid(*num_dim, sparse=False, indexing='ij'), dtype=np.float32)  # [3,55,140,55]

        new_positions = new_positions.reshape(-1,
                                              reduce(lambda x, y: x * y, list(new_positions.shape[1:]))).transpose()

        print("new position shape ", new_positions.shape)  # (423500, 3)
        """
        [
        [ x_start, y_start, z_start],
        [x_start, y_start, z_start + 0.02],
        [x_start, y_start, z_start + 0.04]
        ...
        [x_start, y_start, z_end],
        [x_start, y_start + 0.02, z_start],
        [x_start, y_start + 0.02, z_start + 0.02],
        ...
        [x_start, y_end, z_end],
        [x_start + 0.02, y_start, z_start],
        [x_start + 0.02, y_start, z_start + 0.02],
        ...
        [x_end, y_end, z_end]
        ]
        """
        if velocity is None:
            velocity_arr = np.full_like(new_positions, 0, dtype=np.float32)
        else:
            velocity_arr = np.array([velocity for _ in range(num_new_particles)], dtype=np.float32)  # (423500, 3)

        material_arr = np.full_like(np.zeros(num_new_particles, dtype=np.int32), material)  # 1
        is_dynamic_arr = np.full_like(np.zeros(num_new_particles, dtype=np.int32), is_dynamic)  # 1

        color_arr = np.stack([np.full_like(np.zeros(num_new_particles, dtype=np.int32), c) for c in color], axis=1)

        density_arr = np.full_like(np.zeros(num_new_particles, dtype=np.float32),
                                   density if density is not None else 1000.)
        pressure_arr = np.full_like(np.zeros(num_new_particles, dtype=np.float32),
                                    pressure if pressure is not None else 0.)  # pressure = 0
        print(color_arr.shape, density_arr.shape, pressure_arr.shape)  # (423500, 3) (423500,) (423500,)

        self.add_particles(object_id, num_new_particles, new_positions, velocity_arr, density_arr, pressure_arr,
                           material_arr, is_dynamic_arr, color_arr)

        # self._add_particles also works. Why using self.add_particles??
