from typing import List, Tuple
import numpy as np


class Workbench:
    def __init__(self, id: int, x: float, y: float, remaining_time: int, material_state: int, product_state: bool):
        self.id = id
        self.coord = np.array([x, y])
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


class Map:
    def __init__(self, frame_num: int, money: int, workbenches: List[Workbench]):
        self.frame_num = frame_num
        self.money = money
        self.workbenches = workbenches


class Agent:
    def __init__(self, robots: List[Robot]):
        self.robots = robots


def parse_input(input_string: str) -> Tuple[Map, Agent]:
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
    map_obj = Map(frame_num, money, workbenches)
    agent = Agent(robots)
    return map_obj, agent
