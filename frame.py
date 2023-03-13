from typing import List
import numpy as np
import sys

######### 运动学超参数 #########
# ################################################
# K_W = 4 ## 正常转向时
# K_W2 = 10 ##墙壁周围转向时
# K_V = 4
# S = 6
# PZBJ = 4#碰撞半径，当机器人距离小于这个值时触发规避
# PZGB_w = 60/180*np.pi
# PZGB_v = 1
# WALL_dis = 2
# ################################################
# 势场常数
# k_att = 300.0 # 表示机器人与目标之间的引力常数
# obs_k_rep = 1000.0 # 表示机器人与障碍物之间的斥力常数
# bound_k_rep = 600.0 # 表示机器人与边界之间的斥力常数
# obs_rep_range = 15.0 # 表示障碍物斥力的作用范围
# bound_rep_range = 3.0 # 表示边界斥力的作用范围
# KW = 10 # 超参数
# KV = 1 # 超参数

# k_att = 20.0 # 表示机器人与目标之间的引力常数
# obs_k_rep = 600.0 # 表示机器人与障碍物之间的斥力常数
# bound_k_rep = 500.0 # 表示机器人与边界之间的斥力常数
# obs_rep_range = 30.0 # 表示障碍物斥力的作用范围
# bound_rep_range = 2.0 # 表示边界斥力的作用范围
# targe_att_range =5.0 # 表示目的地的引力作用范围
# KW =2.5 # 超参数
# KV = 1.2 # 超参数
k_att = 10.0 # 表示机器人与目标之间的引力常数
obs_k_rep = 400.0 # 表示机器人与障碍物之间的斥力常数
bound_k_rep = 2000.0 # 表示机器人与边界之间的斥力常数
obs_rep_range = 20.0 # 表示障碍物斥力的作用范围
bound_rep_range = 1.5 # 表示边界斥力的作用范围
targe_att_range =4.0 # 表示目的地的引力作用范围
KW =4 # 超参数
KV = 1.3 # 超参数

class Workbench:
    def __init__(self, type_id: int, x: float, y: float, remaining_time: int, material_state: int, product_state: bool, index:int):
        self.index = index
        self.type_id = type_id
        self.coord = np.array([x, y])
        self.remaining_time = remaining_time
        self.material_state = [j for j, bit in enumerate(
            reversed(bin(int(material_state))[2:])) if bit == "1"]
        self.product_state = product_state
        self.assigned_buy = False
        self.assigned_sell = []

    def update(self, remaining_time:int=None, material_state:int=None, product_state:bool=None):
        self.remaining_time = remaining_time
        self.material_state = [j for j, bit in enumerate(
                reversed(bin(int(material_state))[2:])) if bit == "1"]
        self.product_state = product_state


class Robot:
    def __init__(self, workbench_id: int, carrying_item: int, time_value_coeff: float, collision_value_coeff: float,
                 angular_velocity: float, linear_velocity_x: float, linear_velocity_y: float, heading: float, x: float, y: float, index:int):
        self.index =  index # 机器人的唯一索引
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
        self.last_carry = 0
        self.buy_done = True
        self.sell_done = True
        self.freezed = 0
        self.destroy = False

        
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
        self.destroy = False

    def get_action(self, adj_mat:np.ndarray, headin_glist:np.ndarray, robot_coord:np.ndarray):
        # print('adj_mat=',adj_mat,'self.coord=',self.coord,'headin_glist=',headin_glist, 'robot_coord=', robot_coord, file=sys.stderr)
    # ############################################################################################
        # def w_v_fun(delta_dir:np.ndarray, distance:np.ndarray, task:np.ndarray):
        #     if task[0] < WALL_dis or task[1] < WALL_dis or (50-task[1]) < WALL_dis or (50-task[0]) < WALL_dis:
        #         w = K_W2 * delta_dir
        #         v = np.minimum(distance,S)
        #     else:
        #         w = K_W * delta_dir
        #         v = S
        #     v = K_V*(1-(np.absolute(w)/np.pi))*v
        #     return w, v  
    #     def get_w_v(task:np.ndarray):
    #         ##判断task是否接近墙
    #         tar_cur_dir = task-self.coord
    #         distance = np.linalg.norm(tar_cur_dir, axis=-1)
    #         tar_dir = np.arctan2(tar_cur_dir[1], tar_cur_dir[0]) # 每帧更新
    #         delta_dir = tar_dir - self.heading
    #         if delta_dir > np.pi:
    #             delta_dir -= 2*np.pi
    #         elif delta_dir < -np.pi:
    #             delta_dir += 2*np.pi
    #         return w_v_fun(delta_dir = delta_dir, distance = distance, task = task)
    # ##########################################################################################   
        
        def potential_field(self,task_index:int):
            """
            计算机器人在当前位置的势场
            """
            tar_d= np.linalg.norm(np.float64(self.coord - self.task_coord[task_index,:]))
            if tar_d > targe_att_range:
                att = k_att * (self.task_coord[task_index,:] - self.coord) * tar_d
            else:
                att = k_att * targe_att_range**6 * (self.task_coord[task_index,:] - self.coord)
            
            # att = k_att * (self.task_coord[task_index,:] - self.coord) # 计算机器人当前位置和目标位置之间的引力
            
            obs_rep = np.float64(np.array([0, 0])) # 用于计算机器人当前位置与障碍物之间的斥力
            obstacles = np.delete(robot_coord, self.index)
            for obs in obstacles:
                obs_vec = np.float64(self.coord - np.array(obs))
                obs_d = np.float64(np.linalg.norm(obs_vec))
                if obs_d <= obs_rep_range:
                    obs_rep += -obs_k_rep * (1/obs_d - 1/obs_rep_range) * (1/obs_d**2) * obs_vec / obs_d
                    # obs_rep += obs_k_rep * (1/obs_d - 1/obs_rep_range) * (1/obs_d**2) * obs_vec
                    # obs_rep += obs_k_rep * (1/obs_d - 1/obs_rep_range) * (1/obs_d**2) * obs_vec  / obs_d

            # 添加地图边界限制条件
            boundary_rep = np.float64(np.array([0, 0]))
            if self.coord[0] < bound_rep_range:
                boundary_rep[0] = bound_k_rep * (1/(self.coord[0]+1e-6) - 1/bound_rep_range) / (self.coord[0]+1e-6)**2
            elif self.coord[0] > 50 - bound_rep_range:
                boundary_rep[0] = -bound_k_rep * (1/(50-self.coord[0]+1e-6) - 1/bound_rep_range) / (50-self.coord[0]+1e-6)**2
            if self.coord[1] < bound_rep_range:
                boundary_rep[1] = bound_k_rep * (1/(self.coord[1]+1e-6) - 1/bound_rep_range) / (self.coord[1]+1e-6)**2
            elif self.coord[1] > 50 - bound_rep_range:
                boundary_rep[1] = -bound_k_rep * (1/(50-self.coord[1]+1e-6) - 1/bound_rep_range) / (50-self.coord[1]+1e-6)**2
            # print('att = ', att, 'rep = ', rep, 'boundary_rep = ', boundary_rep)
            new_vel = att + obs_rep + boundary_rep
            # angle_needed = np.arctan2(np.cross(self.linear_velocity, new_vel), np.dot(self.linear_velocity, new_vel))
            tar_dir = np.arctan2(new_vel[1], new_vel[0])
            
            delta_dir = tar_dir - self.heading
            
            # if np.abs(delta_dir) >= np.pi: print('delta_dir1大于pi',delta_dir, file=sys.stderr)
            
            if delta_dir > np.pi:
                delta_dir -= 2*np.pi
            elif delta_dir < -np.pi:
                delta_dir += 2*np.pi
            
            # if np.abs(delta_dir) >= np.pi: print('delta_dir2大于pi',delta_dir, file=sys.stderr)
            if np.abs(delta_dir) > 0.6 * np.pi or self.last_carry != self.carrying_item:
            # if self.last_carry != self.carrying_item:
                print('触发', file=sys.stderr)
                if self.coord[1] > 50 - bound_rep_range:
                    if self.heading > 0 and self.heading <= 0.5*np.pi and delta_dir > 0:
                        delta_dir = delta_dir - 2 * np.pi
                    elif self.heading > 0.5 * np.pi and self.heading < np.pi and delta_dir < 0:
                        delta_dir = 2 * np.pi + delta_dir
                elif self.coord[1] < bound_rep_range:
                    if self.heading < 0 and self.heading >= -0.5*np.pi and delta_dir < 0:
                        delta_dir = 2 * np.pi + delta_dir
                    elif self.heading < -0.5*np.pi and self.heading > -np.pi and delta_dir > 0:
                        delta_dir = delta_dir - 2 * np.pi
                elif self.coord[0] < bound_rep_range:
                    if self.heading < np.pi and self.heading >= 0.5*np.pi and delta_dir > 0:
                        delta_dir = delta_dir - 2 * np.pi
                    elif self.heading > -np.pi and self.heading <= -0.5*np.pi and delta_dir < 0:
                        delta_dir = 2 * np.pi + delta_dir
                elif self.coord[0] > 50 - bound_rep_range:
                    if self.heading > 0 and self.heading <= 0.5*np.pi and delta_dir < 0:
                        delta_dir = 2 * np.pi + delta_dir
                    elif self.heading < 0 and self.heading >= -0.5*np.pi and delta_dir > 0:
                        delta_dir = delta_dir - 2 * np.pi
                    
            w = KW * delta_dir

            tar_cur_dir = self.task_coord[task_index,:]-self.coord
            distance = np.linalg.norm(tar_cur_dir, axis=-1)
            v = np.minimum(distance+2,6)
            v = KV*(1-(np.absolute(delta_dir)/np.pi))*v
            # v = KV*np.linalg.norm(new_vel)
            # print('w=',w,'v=',v, file=sys.stderr)
            self.last_carry = self.carrying_item
            return w, v
        
        sell, buy, destroy = False, False, self.destroy
        ##判断是否接近墙
        # if self.coord[0] < WALL_dis or self.coord[1] < WALL_dis or (50-self.coord[1]) < WALL_dis or (50-self.coord[0]) < WALL_dis:
        #     v = 0.1
        if self.task_coord is None:
            w, v = 0, 0
        else:
            if not self.buy_done:
                if self.workbench_id == self.task[0]:
                    buy = True
                    self.buy_done = True
                w,v = potential_field(self,task_index=0)
            else:
                if self.workbench_id == self.task[1]:
                    sell = True
                    self.sell_done = True
                w,v = potential_field(self,task_index=1)
        
            # #####防碰撞
            # min_index = np.where(adj_mat[self.index] != 0)[0].min()  # 找到距离最小的
            # min_dis = adj_mat[self.index][min_index]
            # if  self.carrying_item != 0 and min_dis <= PZBJ: # 若最近的机器人距离小于碰撞半径，触发防碰撞
            #     duifang_head = headin_glist[min_index]
            #     del_dir = duifang_head - self.heading
            #     if del_dir > np.pi:
            #         del_dir -= 2*np.pi
            #     elif del_dir < -np.pi:
            #         del_dir += 2*np.pi
            #     if np.abs(del_dir) < 0.5*np.pi:
            #         # if self.index < min_index:
            #         if self.coord[1] >= robot_coord[min_index][0]:
            #             # print('self.coord[1]=',self.coord[1],'robot_coord[min_index][1]=',robot_coord[min_index][0], file=sys.stderr)
            #             v = v * ((min_dis-1.5)/PZBJ)
            #     else:
            #         # w -= K_W*PZGB_w ## PZGB_w希望可以动态调整
            #         w -= (1-(min_dis-3/PZBJ)) * PZGB_w
                    
                # w,v = get_w_v(self.task_coord[0,:])
            #     w,v = potential_field(self,task_index=0)
            # else:
            #     if self.workbench_id == self.task[1]:
            #         sell = True
            #     # w,v = get_w_v(self.task_coord[1,:])
            #     w,v = potential_field(self,task_index=1)
                
        # ##############################################################################################
        #     #####防碰撞
        #     min_index = np.where(adj_mat[self.index] != 0)[0].min()  # 找到距离最小的
        #     min_dis = adj_mat[self.index][min_index]
        #     if  self.carrying_item != 0 and min_dis <= PZBJ: # 若最近的机器人距离小于碰撞半径，触发防碰撞
        #         duifang_head = headin_glist[min_index]
        #         del_dir = duifang_head - self.heading
        #         if del_dir > np.pi:
        #             del_dir -= 2*np.pi
        #         elif del_dir < -np.pi:
        #             del_dir += 2*np.pi
        #         if np.abs(del_dir) < 0.5*np.pi:
        #             # if self.index < min_index:
        #             if self.coord[1] >= robot_coord[min_index][0]:
        #                 # print('self.coord[1]=',self.coord[1],'robot_coord[min_index][1]=',robot_coord[min_index][0], file=sys.stderr)
        #                 v = v * ((min_dis-1.5)/PZBJ)
        #         else:
        #             # w -= K_W*PZGB_w ## PZGB_w希望可以动态调整
        #             w -= (1-(min_dis-3/PZBJ)) * PZGB_w
        # #############################################################################################           
        return sell, buy, destroy, w, v

class Map:
    def __init__(self, frame_num: int, money: int, workbenches: List[Workbench], robots:List[Robot]):
        self.frame_num = frame_num
        self.money = money
        self.workbenches = workbenches
        self.robots = robots 
        self.workbench_adj_mat = self._get_workbench_adj_mat() # 邻接矩阵
        self.robot_adj_mat = self._get_robot_adj_mat() # 邻接矩阵
        self.robot_heading = [r.heading for r in self.robots]
        self.robot_coord = np.array([r.coord for r in self.robots])
        self.wb_coord = np.array([w.coord for w in self.workbenches])
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
        self.robot_adj_mat = self._get_robot_adj_mat()
        self.robot_heading = [r.heading for r in self.robots]

    def output(self):
        output = f"{self.frame_num}\n"
        for i,r in enumerate(self.robots):
            sell, buy, destroy, w, v = r.get_action(self.robot_adj_mat, self.robot_heading, self.robot_coord)
            output += f"forward {i} {v}\nrotate {i} {w}\n"
            if sell:
                output += f"sell {i}\n"
            if buy:
                output += f"buy {i}\n"
            if destroy:
                output += f"destroy {i}\n"
        output += "OK"
        return output
    
    def _get_adj_mat(self, xy):
        XY = xy[:, None] - xy[None, :]
        XY = np.linalg.norm(XY,2,-1)
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
    return map_obj
