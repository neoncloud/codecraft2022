from typing import List
import numpy as np
import sys

######### 运动学超参数 #########
# # 势场参数
k_att = 10.0  # 表示机器人与目标之间的引力常数
obs_k_rep = 350.0  # 表示机器人与障碍物之间的斥力常数
bound_k_rep = 25000.0  # 表示机器人与边界之间的斥力常数
obs_rep_range = 25.0  # 表示障碍物斥力的作用范围
bound_rep_range = 1.5  # 表示边界斥力的作用范围
targe_att_range = 4.0  # 表示目的地的引力作用范围

EPSILON = 0.00001
time_horizon = 0.5
time_step = 0.02
max_speed = 6.02


class Line:
    def __init__(self):
        self.direction = np.array([0.0, 0.0])
        self.point = np.array([0.0, 0.0])

def abs_sq(vector):
    return np.dot(vector, vector)

def square(scalar):
    return scalar * scalar

def linear_program1(lines, lineNo, radius, optVelocity, directionOpt):
    dotProduct = np.dot(lines[lineNo].point, lines[lineNo].direction)
    discriminant = np.square(
        dotProduct) + np.square(radius) - np.square(np.linalg.norm(lines[lineNo].point))

    if discriminant < 0.0:
        return False, None

    sqrtDiscriminant = np.sqrt(discriminant)
    tLeft = -dotProduct - sqrtDiscriminant
    tRight = -dotProduct + sqrtDiscriminant

    for i in range(lineNo):
        denominator = np.linalg.det(np.column_stack(
            (lines[lineNo].direction, lines[i].direction)))
        numerator = np.linalg.det(np.column_stack(
            (lines[i].direction, lines[lineNo].point - lines[i].point)))

        if np.abs(denominator) <= EPSILON:
            if numerator < 0.0:
                return False, None
            continue

        t = numerator / denominator

        if denominator >= 0.0:
            tRight = min(tRight, t)
        else:
            tLeft = max(tLeft, t)

        if tLeft > tRight:
            return False, None

    if directionOpt:
        if np.dot(optVelocity, lines[lineNo].direction) > 0.0:
            result = lines[lineNo].point + \
                tRight * lines[lineNo].direction
        else:
            result = lines[lineNo].point + \
                tLeft * lines[lineNo].direction
    else:
        t = np.dot(lines[lineNo].direction,
                    (optVelocity - lines[lineNo].point))

        if t < tLeft:
            result = lines[lineNo].point + \
                tLeft * lines[lineNo].direction
        elif t > tRight:
            result = lines[lineNo].point + \
                tRight * lines[lineNo].direction
        else:
            result = lines[lineNo].point + t * lines[lineNo].direction

    return True, result

def linear_program2(lines, radius, optVelocity, directionOpt, result):
    if directionOpt:
        result = optVelocity * radius
    elif np.square(np.linalg.norm(optVelocity)) > np.square(radius):
        result = (optVelocity / np.linalg.norm(optVelocity)) * radius
    else:
        result = optVelocity

    for i in range(len(lines)):
        if np.linalg.det(np.column_stack((lines[i].direction, lines[i].point - result))) > 0.0:
            tempResult = result
            success, result = linear_program1(
                lines, i, radius, optVelocity, directionOpt)
            if not success:
                result = tempResult
                return i, result

    return len(lines), result

def linear_program3(lines, numObstLines, beginLine, radius, result):
    distance = 0.0

    for i in range(beginLine, len(lines)):
        if np.linalg.det(np.column_stack((lines[i].direction, lines[i].point - result))) > distance:
            projLines = []

            for ii in range(numObstLines):
                projLines.append(lines[ii])

            for j in range(numObstLines, i):
                line = Line()
                determinant = np.linalg.det(np.column_stack(
                    (lines[i].direction, lines[j].direction)))

                if np.abs(determinant) <= EPSILON:
                    if np.dot(lines[i].direction, lines[j].direction) > 0.0:
                        continue
                    else:
                        line.point = 0.5 * \
                            (lines[i].point + lines[j].point)
                else:
                    line.point = lines[i].point + (np.linalg.det(np.column_stack(
                        (lines[j].direction, lines[i].point - lines[j].point))) / determinant) * lines[i].direction
                line.direction = (lines[j].direction - lines[i].direction) / np.linalg.norm(
                    lines[j].direction - lines[i].direction)
                projLines.append(line)

            tempResult = result
            lineFail, result = linear_program2(projLines, radius, np.array(
                [-lines[i].direction[1], lines[i].direction[0]]), True, result)
            if lineFail < len(projLines):
                result = tempResult

            distance = np.linalg.det(np.column_stack(
                (lines[i].direction, lines[i].point - result)))
    return result


def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def de_normalize_angle(angle):
    if angle < 0:
        angle = 2*np.pi + angle
    return angle

class Workbench:
    def __init__(self, type_id: int, x: float, y: float, remaining_time: int, material_state: int, product_state: bool, index: int):
        self.index = index
        self.type_id = type_id
        self.coord = np.array([x, y])
        self.remaining_time = remaining_time
        self.material_state = [j for j, bit in enumerate(
            reversed(bin(int(material_state))[2:])) if bit == "1"]
        self.product_state = product_state
        self.assigned_task = 0  # 预约了节点
        self.incoming = []
        self.outcoming = False

    def update(self, remaining_time: int = None, material_state: int = None, product_state: bool = None):
        self.product_state = product_state
        self.remaining_time = remaining_time
        self.material_state = [j for j, bit in enumerate(
            reversed(bin(int(material_state))[2:])) if bit == "1"]


class Robot:
    def __init__(self, workbench_id: int, carrying_item: int, time_value_coeff: float, collision_value_coeff: float,
                 angular_velocity: float, linear_velocity_x: float, linear_velocity_y: float, heading: float, x: float, y: float, index: int):
        self.index = index  # 机器人的唯一索引
        self.workbench_id = workbench_id
        self.carrying_item = carrying_item
        self.time_value_coeff = time_value_coeff
        self.collision_value_coeff = collision_value_coeff
        self.angular_velocity = angular_velocity
        self.linear_velocity = np.array([linear_velocity_x, linear_velocity_y])
        self.heading = heading
        self.coord = np.array([x, y])
        self.task = None
        self.task_coord = None  # 用于存每次的任务 该 robot 要去买的坐标 | 该 robot 要去卖的坐标
        self.last_carry = 0
        self.buy_done = True
        self.sell_done = True
        self.freezed = 0
        self.destroy = False

    def update(self, workbench_id: int = None, carrying_item: int = None, time_value_coeff: float = None, collision_value_coeff: float = None,
               angular_velocity: float = None, linear_velocity_x: float = None, linear_velocity_y: float = None, heading: float = None, x: float = None, y: float = None):
        self.workbench_id = workbench_id
        self.carrying_item = carrying_item
        self.time_value_coeff = time_value_coeff
        self.collision_value_coeff = collision_value_coeff
        self.angular_velocity = angular_velocity
        self.linear_velocity = np.array([linear_velocity_x, linear_velocity_y])
        self.heading = heading
        self.coord = np.array([x, y])
        if self.task_coord is not None and self.task is not None:
            self.task_coord[1] = self.task[1].workbench.coord

    def compute_orca_lines(self, other_robots_coord, other_robots_linearv, other_robots_carry, time_horizon, time_step):
        invTimeHorizon = 1.0 / time_horizon
        orca_lines = []

        for i in range(3):
            relativePosition = other_robots_coord[i] - self.coord
            relativeVelocity = self.linear_velocity - \
                other_robots_linearv[i]

            distSq = abs_sq(relativePosition)
            if self.carrying_item == 0:
                agentradius = 0.45
            else:
                agentradius = 0.53
            if other_robots_carry[i] == 0:
                otherradius = 0.45
            else:
                otherradius = 0.53
            combinedRadius = agentradius + otherradius
            combinedRadiusSq = square(combinedRadius)

            line = Line()
            u = np.array([0.0, 0.0])

            if distSq > combinedRadiusSq:
                w = relativeVelocity - invTimeHorizon * relativePosition

                wLengthSq = abs_sq(w)
                dotProduct1 = np.dot(w, relativePosition)

                if dotProduct1 < 0.0 and square(dotProduct1) > combinedRadiusSq * wLengthSq:
                    wLength = np.sqrt(wLengthSq)
                    unitW = w / wLength

                    line.direction = np.array([unitW[1], -unitW[0]])
                    u = (combinedRadius * invTimeHorizon - wLength) * unitW
                else:
                    leg = np.sqrt(distSq - combinedRadiusSq)

                    if np.cross(relativePosition, w) > 0.0:
                        line.direction = np.array([(relativePosition[0] * leg - relativePosition[1] * combinedRadius), (
                            relativePosition[0] * combinedRadius + relativePosition[1] * leg)]) / distSq
                    else:
                        line.direction = -np.array([(relativePosition[0] * leg + relativePosition[1] * combinedRadius),
                                                    (-relativePosition[0] * combinedRadius + relativePosition[1] * leg)]) / distSq

                    dotProduct2 = np.dot(relativeVelocity, line.direction)
                    u = dotProduct2 * line.direction - relativeVelocity
            else:
                invTimeStep = 1.0 / time_step

                w = relativeVelocity - invTimeStep * relativePosition

                wLength = np.linalg.norm(w)
                unitW = w / wLength

                line.direction = np.array([unitW[1], -unitW[0]])
                u = (combinedRadius * invTimeStep - wLength) * unitW

            line.point = self.linear_velocity + 0.5 * u
            orca_lines.append(line)

        return orca_lines


    def calculate_target_att(self, task_index):
        tar_d = np.linalg.norm(np.float64(
            self.coord - self.task_coord[task_index, :]))
        if tar_d > targe_att_range:
            att = k_att * \
                (self.task_coord[task_index, :] - self.coord) / tar_d
        else:
            att = k_att * targe_att_range**6 * \
                (self.task_coord[task_index, :] - self.coord)
        return att

    def calculate_bound_rep(self):
        boundary_rep = np.float64(np.array([0, 0]))
        if self.coord[0] <= bound_rep_range:
            boundary_rep[0] = bound_k_rep * \
                (1/(self.coord[0]+1e-6) - 1 /
                    bound_rep_range) / (self.coord[0]+1e-6)**2
        elif self.coord[0] > 50 - bound_rep_range:
            boundary_rep[0] = -bound_k_rep * (
                1/(50-self.coord[0]+1e-6) - 1/bound_rep_range) / (50-self.coord[0]+1e-6)**2
        if self.coord[1] <= bound_rep_range:
            boundary_rep[1] = bound_k_rep * \
                (1/(self.coord[1]+1e-6) - 1 /
                    bound_rep_range) / (self.coord[1]+1e-6)**2
        elif self.coord[1] > 50 - bound_rep_range:
            boundary_rep[1] = -bound_k_rep * (
                1/(50-self.coord[1]+1e-6) - 1/bound_rep_range) / (50-self.coord[1]+1e-6)**2
        return boundary_rep

    def final_version(self, task_index, robot_coord, robot_linear_v, robot_carrying_item):
        # # 得到目标方向
        tar_att = self.calculate_target_att(task_index)  # 计算目标引力
        bound_rep = self.calculate_bound_rep()  # 计算边界斥力
        target_vector = tar_att + bound_rep  # 计算合力
        pref_velocity = target_vector / \
            np.linalg.norm(target_vector) * max_speed

        other_robots_coord = np.delete(robot_coord, self.index, axis=0)
        other_robots_linearv = np.delete(
            robot_linear_v, self.index, axis=0)
        other_robots_carry = np.delete(
            robot_carrying_item, self.index, axis=0)

        orca_lines = self.compute_orca_lines(
            other_robots_coord, other_robots_linearv, other_robots_carry, time_horizon, time_step)

        new_velocity = np.array([0.0, 0.0])
        lineFail, new_velocity = linear_program2(
            orca_lines, max_speed, pref_velocity, False, new_velocity)
        if lineFail < len(orca_lines):
            new_velocity = linear_program3(orca_lines, len(
                orca_lines), lineFail, max_speed, new_velocity)
        return new_velocity

    def move(self, best_velocity):
        KWW = 85
        # KVV1 = 0.1
        # KWW1 = 300
        # KVV = 1.3
        v = 6
        desired_angle = de_normalize_angle(
            np.arctan2(best_velocity[1], best_velocity[0]))
        heading = de_normalize_angle(self.heading)
        delta_w = normalize_angle(desired_angle - heading)
        # delta_w = bound_turn_round(self,delta_dir=delta_w,bound_range=1)
        w = KWW * delta_w

        if np.abs(delta_w) > 0.30*np.pi:
            v = np.linalg.norm(best_velocity, axis=-1)/7
        else:
            v = np.linalg.norm(best_velocity, axis=-1)
        # v = np.linalg.norm(best_velocity, axis=-1)
        return w, v

    def get_action(self, robot_coord: np.ndarray, robot_linear_v: np.ndarray, robot_carrying_item: np.ndarray):
        sell, buy, destroy = False, False, self.destroy
        if self.task_coord is None:
            w, v = 0, 0
        else:
            if not self.buy_done:
                if self.workbench_id == self.task[0].workbench.index:
                    if self.carrying_item == 0:
                        if not self.task[0].workbench.product_state:
                            # 失败，对面还没生产
                            self.buy_done = True
                            self.sell_done = True
                            # self.task[0].done = False
                        else:
                            buy = True
                            self.task[0].workbench.outcoming = False
                            # self.task[0].done = True
                    else:
                        self.buy_done = True
                        self.task[0].workbench.assigned_task -= 1
                best_velocity = self.final_version(task_index=0, robot_coord=robot_coord, robot_linear_v=robot_linear_v, robot_carrying_item=robot_carrying_item)
                w, v = self.move(best_velocity)
            elif not self.sell_done:
                if self.workbench_id == self.task[1].workbench.index:
                    if self.carrying_item != 0:
                        if self.carrying_item not in self.task[1].workbench.material_state:
                            sell = True
                            if self.carrying_item in self.task[1].workbench.incoming:
                                self.task[1].workbench.incoming.remove(
                                    self.carrying_item)
                        else:
                            delta = np.array([25, 25])-self.task_coord[1]
                            self.task_coord[1] += 3*delta/np.linalg.norm(delta)
                    else:
                        self.task[0].workbench.assigned_task -= 1
                        self.sell_done = True
                        self.buy_done = True
                    # self.task_coord = None
                best_velocity = self.final_version(task_index=1, robot_coord=robot_coord, robot_linear_v=robot_linear_v, robot_carrying_item=robot_carrying_item)
                w, v = self.move(best_velocity)
            else:
                self.task_coord[1] += np.array([1, 1])
                best_velocity = self.final_version(task_index=1, robot_coord=robot_coord, robot_linear_v=robot_linear_v, robot_carrying_item=robot_carrying_item)
                w, v = self.move(best_velocity)
        return sell, buy, destroy, w, v


class Map:
    def __init__(self, frame_num: int, money: int, workbenches: List[Workbench], robots: List[Robot]):
        self.frame_num = frame_num
        self.money = money
        self.workbenches = workbenches
        self.robots = robots
        self.workbench_adj_mat = self._get_workbench_adj_mat()  # 邻接矩阵
        self.robot_adj_mat = self._get_robot_adj_mat()  # 邻接矩阵
        self.robot_heading = [r.heading for r in self.robots]
        self.robot_coord = np.array([r.coord for r in self.robots])
        self.wb_coord = np.array([w.coord for w in self.workbenches])
        self.num_robots = len(robots)
        self.num_workbenches = len(workbenches)
        self.robot_linear_v = np.array(
            [r.linear_velocity for r in self.robots])
        self.robot_carrying_item = np.array(
            [r.carrying_item for r in self.robots])

    def update(self, input_string: str):
        lines = input_string.strip().split('\n')
        self.frame_num, self.money = map(int, lines[0].split())
        workbench_count = int(lines[1])
        # for i in range(2, 2 + workbench_count):
        for i, r in enumerate(self.robots, 2 + workbench_count):
            robot_data = list(map(float, lines[i].split()))
            r.update(int(robot_data[0]), int(robot_data[1]), robot_data[2], robot_data[3],
                     robot_data[4], robot_data[5], robot_data[6], robot_data[7], robot_data[8], robot_data[9])
        self.robot_coord = np.array([r.coord for r in self.robots])
        self.robot_adj_mat = self._get_robot_adj_mat()
        self.robot_heading = [r.heading for r in self.robots]
        self.robot_linear_v = np.array(
            [r.linear_velocity for r in self.robots])
        self.robot_carrying_item = np.array(
            [r.carrying_item for r in self.robots])
        for i, w in enumerate(self.workbenches, 2):
            workbench_data = list(map(float, lines[i].split()))
            w.update(int(workbench_data[3]), int(
                workbench_data[4]), bool(workbench_data[5]))
            # print(f"Workbench {w.index} assigned buy: {w.assigned_buy}, assigned sell: {w.assigned_sell}", file=sys.stderr)

    def output(self):
        output = f"{self.frame_num}\n"
        for i, r in enumerate(self.robots):
            sell, buy, destroy, w, v = r.get_action(self.robot_coord, self.robot_linear_v, self.robot_carrying_item)
            output += f"forward {i} {v}\nrotate {i} {w}\n"
            if sell:
                output += f"sell {i}\n"
            if buy and self.frame_num < 8900:
                output += f"buy {i}\n"
            if destroy:
                output += f"destroy {i}\n"
        output += "OK"
        return output

    def _get_adj_mat(self, xy):
        XY = xy[:, None] - xy[None, :]
        XY = np.linalg.norm(XY, 2, -1)
        return XY

    def _get_workbench_adj_mat(self):
        xy = np.stack([w.coord for w in self.workbenches], axis=0)
        return self._get_adj_mat(xy)

    def _get_robot_adj_mat(self):
        xy = np.stack([r.coord for r in self.robots], axis=0)
        return self._get_adj_mat(xy)


def parse_init_frame(input_string: str) -> Map:
    lines = input_string.strip().split('\n')
    frame_num, money = map(int, lines[0].split())
    workbench_count = int(lines[1])
    workbenches = []
    for i in range(2, 2 + workbench_count):
        workbench_data = list(map(float, lines[i].split()))
        workbench = Workbench(int(workbench_data[0]), workbench_data[1], workbench_data[2], int(
            workbench_data[3]), int(workbench_data[4]), bool(workbench_data[5]), i-2)
        workbenches.append(workbench)
    robots = []
    for i in range(2 + workbench_count, 2 + workbench_count + 4):
        robot_data = list(map(float, lines[i].split()))
        robot = Robot(int(robot_data[0]), int(robot_data[1]), robot_data[2], robot_data[3],
                      robot_data[4], robot_data[5], robot_data[6], robot_data[7], robot_data[8], robot_data[9], i-2-workbench_count)
        robots.append(robot)
    map_obj = Map(frame_num, money, workbenches, robots)
    map_obj.update(input_string)
    return map_obj
