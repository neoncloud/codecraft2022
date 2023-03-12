from typing import List
import numpy as np
import sys

######### 运动学超参数 #########
K_W = 10
K_V = 5
S = 6

class Workbench:
    def __init__(self, type_id: int, x: float, y: float, remaining_time: int, material_state: int, product_state: bool):
        self.type_id = type_id
        self.coord = np.array([x, y])
        self.remaining_time = remaining_time
        self.material_state = [j for j, bit in enumerate(
            reversed(bin(int(material_state))[2:])) if bit == "1"]
        self.product_state = product_state

    def update(self, remaining_time:int=None, material_state:int=None, product_state:bool=None):
        self.remaining_time = remaining_time
        self.material_state = [j for j, bit in enumerate(
                reversed(bin(int(material_state))[2:])) if bit == "1"]
        self.product_state = product_state


class Robot:
    def __init__(self, workbench_id: int, carrying_item: int, time_value_coeff: float, collision_value_coeff: float,
                 angular_velocity: float, linear_velocity_x: float, linear_velocity_y: float, heading: float, x: float, y: float):
        self.workbench_id = workbench_id
        self.carrying_item = carrying_item
        self.time_value_coeff = time_value_coeff
        self.collision_value_coeff = collision_value_coeff
        self.angular_velocity = angular_velocity
        self.linear_velocity = np.array([linear_velocity_x, linear_velocity_y])
        self.heading = heading
        self.coord = np.array([x, y])
        self.task = None
        self.task_coord = None #用于存每次的任务 该robot要去买的坐标|该robot要去卖的坐标
        # self.behav_buy = False
        # self.behav_sale = False
        # self.behav_destroy = False
        # self.behav_w = 0
        # self.behav_v = 0
        
    def update(self, workbench_id:int=None, carrying_item:int=None, time_value_coeff:float=None, collision_value_coeff:float=None,
               angular_velocity:float=None, linear_velocity_x: float=None, linear_velocity_y: float=None, heading:float=None, x:float=None, y:float=None):
        self.workbench_id = workbench_id
        self.carrying_item = carrying_item
        self.time_value_coeff = time_value_coeff
        self.collision_value_coeff = collision_value_coeff
        self.angular_velocity = angular_velocity
        self.linear_velocity = np.array([linear_velocity_x, linear_velocity_y])
        self.heading = heading
        self.coord = np.array([x, y])
        if self.workbench_id == self.task[1] and self.carrying_item==0:
            # 位于任务末端，且卖完了东西，则清空任务
            self.task = None
            self.task_coord = None
    
    def get_action(self):
        def w_v_fun(delta_dir:np.ndarray, distance:np.ndarray):
            w = K_W * delta_dir
            v = np.minimum(distance,S)
            v = K_V*(1-(np.absolute(w)/np.pi))*v
            return w, v
        
        def get_w_v(task:np.ndarray):
            tar_cur_dir = task-self.coord
            distance = np.linalg.norm(tar_cur_dir, axis=-1)
            tar_dir = np.arctan2(tar_cur_dir[1], tar_cur_dir[0]) # 每帧更新
            delta_dir = tar_dir - self.heading
            if np.absolute(delta_dir) > np.pi : delta_dir += 2*np.pi # w的范围是[-pi,pi]
            # print('cur_dir=',self.heading,'tar_dir=',tar_dir,'delta_dir=',delta_dir,'distance=',distance)
            return w_v_fun(delta_dir = delta_dir, distance = distance)
        
        sell, buy, destroy = False, False, False
        if self.task_coord is None:
            w, v = 0, 0
        else:
            if self.carrying_item == 0:
                if self.workbench_id == self.task[0]:
                    buy = True
                w,v = get_w_v(self.task_coord[0,:])
            else:
                if self.workbench_id == self.task[1]:
                    sell = True
                w,v = get_w_v(self.task_coord[1,:])
        return sell, buy, destroy, w, v

class Map:
    def __init__(self, frame_num: int, money: int, workbenches: List[Workbench], robots:List[Robot]):
        self.frame_num = frame_num
        self.money = money
        self.workbenches = workbenches
        self.robots = robots 
        self.adj_mat = self._get_adj_mat() # 邻接矩阵
        self.num_robots = len(robots)
        self.num_workbenches = len(workbenches)

    def update(self, input_string:str):
        lines = input_string.strip().split('\n')
        self.frame_num, self.money = map(int, lines[0].split())
        workbench_count = int(lines[1])
        # for i in range(2, 2 + workbench_count):
        for i, w in enumerate(self.workbenches, 2):
            workbench_data = list(map(float, lines[i].split()))
            w.update(int(workbench_data[3]), int(workbench_data[4]), bool(workbench_data[5]))
        for i, r in enumerate(self.robots, 2 + workbench_count):
            robot_data = list(map(float, lines[i].split()))
            r.update(int(robot_data[0]), int(robot_data[1]), robot_data[2], robot_data[3], robot_data[4], robot_data[5], robot_data[6], robot_data[7], robot_data[8], robot_data[9])

    def output(self):
        output = f"{self.frame_num}\n"
        for i,r in enumerate(self.robots):
            sell, buy, destroy, w, v = r.get_action()
            output += f"forward {i} {v}\nrotate {i} {w}\n"
            if sell:
                output += f"sell {i}\n"
            if buy:
                output += f"buy {i}\n"
            if destroy:
                output += f"destroy {i}\n"
        output += "OK"
        return output
    
    def _get_adj_mat(self):
        xy = np.stack([w.coord for w in self.workbenches], axis=0)
        XY = xy[:, None] - xy[None, :]
        XY = np.linalg.norm(XY,2,-1)
        return XY


def parse_init_frame(input_string: str) -> Map:
    lines = input_string.strip().split('\n')
    frame_num, money = map(int, lines[0].split())
    workbench_count = int(lines[1])
    workbenches = []
    for i in range(2, 2 + workbench_count):
        workbench_data = list(map(float, lines[i].split()))
        workbench = Workbench(int(workbench_data[0]), workbench_data[1], workbench_data[2], int(
            workbench_data[3]), int(workbench_data[4]), bool(workbench_data[5]))
        workbenches.append(workbench)
    robots = []
    for i in range(2 + workbench_count, 2 + workbench_count + 4):
        robot_data = list(map(float, lines[i].split()))
        robot = Robot(int(robot_data[0]), int(robot_data[1]), robot_data[2], robot_data[3],
                      robot_data[4], robot_data[5], robot_data[6], robot_data[7], robot_data[8], robot_data[9])
        robots.append(robot)
    map_obj = Map(frame_num, money, workbenches, robots)
    return map_obj
