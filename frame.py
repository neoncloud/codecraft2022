from typing import List, Tuple
import numpy as np


class Workbench:
    def __init__(self, type_id: int, x: float, y: float, remaining_time: int, material_state: int, product_state: bool):
        self.type_id = type_id
        self.coord = np.array([x, y])
        self.remaining_time = remaining_time
        self.material_state = [j for j, bit in enumerate(
            reversed(bin(int(material_state))[2:])) if bit == "1"]
        self.product_state = product_state

    def update(self, remaining_time:int=None, material_state:int=None, product_state:bool=None):
        if remaining_time is not None:
            self.remaining_time = remaining_time
        if material_state is not None:
            self.material_state = [j for j, bit in enumerate(
                reversed(bin(int(material_state))[2:])) if bit == "1"]
        if product_state is not None:
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

    def update(self, workbench_id:int=None, carrying_item:int=None, time_value_coeff:float=None, collision_value_coeff:float=None,
               angular_velocity:float=None, linear_velocity_x: float=None, linear_velocity_y: float=None, heading:float=None, x:float=None, y:float=None):
        if workbench_id is not None:
            self.workbench_id = workbench_id
        if carrying_item is not None:
            self.carrying_item = carrying_item
        if time_value_coeff is not None:
            self.time_value_coeff = time_value_coeff
        if collision_value_coeff is not None:
            self.collision_value_coeff = collision_value_coeff
        if angular_velocity is not None:
            self.angular_velocity = angular_velocity
        if linear_velocity_x is not None and linear_velocity_y is not None:
            self.linear_velocity = np.array([linear_velocity_x, linear_velocity_y])
        if heading is not None:
            self.heading = heading
        if x is not None and y is not None:
            self.coord = np.array([x, y])

class Map:
    def __init__(self, frame_num: int, money: int, workbenches: List[Workbench], robots:List[Robot], workbench_dict:dict):
        self.frame_num = frame_num
        self.money = money
        self.workbenches = workbenches
        self.robots = robots
        self.adj_mat = self._get_adj_mat()

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
    
    def _get_adj_mat(self):
        xy = np.stack([w.coord for w in self.workbenches], axis=0)
        XY = xy[:, None] - xy[None, :]
        XY = np.linalg.norm(XY,2,-1)
        return XY


def parse_input(input_string: str) -> Map:
    lines = input_string.strip().split('\n')
    frame_num, money = map(int, lines[0].split())
    workbench_count = int(lines[1])
    workbenches = []
    workbench_dict = {i:[] for i in range(1,10)}
    for i in range(2, 2 + workbench_count):
        workbench_data = list(map(float, lines[i].split()))
        workbench = Workbench(int(workbench_data[0]), workbench_data[1], workbench_data[2], int(
            workbench_data[3]), int(workbench_data[4]), bool(workbench_data[5]))
        workbenches.append(workbench)
        workbench_dict[int(workbench_data[0])].append(i-2)
    robots = []
    for i in range(2 + workbench_count, 2 + workbench_count + 4):
        robot_data = list(map(float, lines[i].split()))
        robot = Robot(int(robot_data[0]), int(robot_data[1]), robot_data[2], robot_data[3],
                      robot_data[4], robot_data[5], robot_data[6], robot_data[7], robot_data[8], robot_data[9])
        robots.append(robot)
    map_obj = Map(frame_num, money, workbenches, robots, workbench_dict)
    return map_obj
