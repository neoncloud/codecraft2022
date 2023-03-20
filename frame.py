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
# # 势场参数
k_att = 10.0 # 表示机器人与目标之间的引力常数
obs_k_rep = 350.0 # 表示机器人与障碍物之间的斥力常数
bound_k_rep = 2500.0 # 表示机器人与边界之间的斥力常数
obs_rep_range = 25.0 # 表示障碍物斥力的作用范围
bound_rep_range = 1.5 # 表示边界斥力的作用范围
targe_att_range =4.0 # 表示目的地的引力作用范围
# k_att = 1000.0 # 表示机器人与目标之间的引力常数
# obs_k_rep = 13300.0 # 表示机器人与障碍物之间的斥力常数
# bound_k_rep = 23300.0 # 表示机器人与边界之间的斥力常数
# obs_rep_range = 2.6 # 表示障碍物斥力的作用范围
# bound_rep_range = 0.9 # 表示边界斥力的作用范围
# targe_att_range =1.3 # 表示目的地的引力作用范围
KW = 4# 超参数
KV = 1.3 # 超参数
VO_distance_range = 15.0
max_speed = 6.0
First_frame = True

class Workbench:
    def __init__(self, type_id: int, x: float, y: float, remaining_time: int, material_state: int, product_state: bool, index:int):
        self.index = index
        self.type_id = type_id
        self.coord = np.array([x, y])
        self.remaining_time = remaining_time
        self.material_state = [j for j, bit in enumerate(
            reversed(bin(int(material_state))[2:])) if bit == "1"]
        self.product_state = product_state
        self.assigned_buy = False  # 预约了购买
        self.assigned_task = False # 预约了节点
        self.assigned_sell = []    # 预约了出售

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
        self.task_coord = None #用于存每次的任务 该 robot 要去买的坐标 | 该 robot 要去卖的坐标
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
        # print('self.linear_velocity=',np.linalg.norm(self.linear_velocity), file=sys.stderr)
        # print('self.angular_velocity=',self.angular_velocity, file=sys.stderr)
        
        def potential_field(self,task_index:int):
            # 计算引力
            tar_d= np.linalg.norm(np.float64(self.coord - self.task_coord[task_index,:]))
            # att = k_att * (1/tar_d) * (self.task_coord[task_index,:] - self.coord)
            if tar_d > targe_att_range:
                # att = k_att * (self.task_coord[task_index,:] - self.coord) * tar_d
                att = k_att * (self.task_coord[task_index,:] - self.coord) / tar_d
            else:
                att = k_att * targe_att_range**6 * (self.task_coord[task_index,:] - self.coord)
            
            # 计算障碍斥力
            obs_rep = np.float64(np.array([0, 0]))
            obstacles = np.delete(robot_coord, self.index)
            for obs in obstacles:
                obs_vec = np.float64(self.coord - np.array(obs))
                obs_d = np.float64(np.linalg.norm(obs_vec))
                if obs_d <= obs_rep_range:
                    # obs_rep += -obs_k_rep * (1/obs_d - 1/obs_rep_range) * (1/obs_d**2) * obs_vec / obs_d
                    obs_rep += obs_k_rep * (1/obs_d - 1/obs_rep_range) * (1/obs_d**2) * obs_vec / obs_d
                    
            # 计算边界斥力
            boundary_rep = np.float64(np.array([0, 0]))
            if self.coord[0] <= bound_rep_range:
                boundary_rep[0] = bound_k_rep * (1/(self.coord[0]+1e-6) - 1/bound_rep_range) / (self.coord[0]+1e-6)**2
            elif self.coord[0] > 50 - bound_rep_range:
                boundary_rep[0] = -bound_k_rep * (1/(50-self.coord[0]+1e-6) - 1/bound_rep_range) / (50-self.coord[0]+1e-6)**2
            if self.coord[1] <= bound_rep_range:
                boundary_rep[1] = bound_k_rep * (1/(self.coord[1]+1e-6) - 1/bound_rep_range) / (self.coord[1]+1e-6)**2
            elif self.coord[1] > 50 - bound_rep_range:
                boundary_rep[1] = -bound_k_rep * (1/(50-self.coord[1]+1e-6) - 1/bound_rep_range) / (50-self.coord[1]+1e-6)**2
            # print('att = ', att, 'rep = ', rep, 'boundary_rep = ', boundary_rep)
            new_vel = att + obs_rep + boundary_rep
            # angle_needed = np.arctan2(np.cross(self.linear_velocity, new_vel), np.dot(self.linear_velocity, new_vel))
            tar_dir = np.arctan2(new_vel[1], new_vel[0])
            # tar_dir = velocity_obstacle2(self,tar_dir)
            
            # # Calculate the velocity obstacles
            # other_robots = np.delete(robot_coord, self.index, axis=0)
            # velocity_obstacles = [calculate_velocity_obstacle(self, other_robot) for other_robot in other_robots]
            # best_orientation = None
            # for angle in np.linspace(0, 2*np.pi, 180):
            #     # print('angle=',angle)
            #     available = True
            #     candidate_orientation = tar_dir - angle
            #     for vo in velocity_obstacles:
            #         if vo[0] < candidate_orientation and candidate_orientation < vo[1]:
            #             available = False
            #             break
            #     if available:
            #         tar_dir = candidate_orientation
            #         break
            
            delta_dir = tar_dir - self.heading
            if delta_dir > np.pi:
                delta_dir -= 2*np.pi
            elif delta_dir < -np.pi:
                delta_dir += 2*np.pi
            
            if np.abs(delta_dir) > 0.6 * np.pi or self.last_carry != self.carrying_item:
                print('触发', file=sys.stderr)
                if self.coord[1] > 50 - 2:
                    if self.heading > 0 and self.heading <= 0.5*np.pi and delta_dir > 0:
                        delta_dir = delta_dir - 2 * np.pi
                    elif self.heading > 0.5 * np.pi and self.heading < np.pi and delta_dir < 0:
                        delta_dir = 2 * np.pi + delta_dir
                elif self.coord[1] < 2:
                    if self.heading < 0 and self.heading >= -0.5*np.pi and delta_dir < 0:
                        delta_dir = 2 * np.pi + delta_dir
                    elif self.heading < -0.5*np.pi and self.heading > -np.pi and delta_dir > 0:
                        delta_dir = delta_dir - 2 * np.pi
                elif self.coord[0] < 2:
                    if self.heading < np.pi and self.heading >= 0.5*np.pi and delta_dir > 0:
                        delta_dir = delta_dir - 2 * np.pi
                    elif self.heading > -np.pi and self.heading <= -0.5*np.pi and delta_dir < 0:
                        delta_dir = 2 * np.pi + delta_dir
                elif self.coord[0] > 50 - 2:
                    if self.heading > 0 and self.heading <= 0.5*np.pi and delta_dir < 0:
                        delta_dir = 2 * np.pi + delta_dir
                    elif self.heading < 0 and self.heading >= -0.5*np.pi and delta_dir > 0:
                        delta_dir = delta_dir - 2 * np.pi
            
            w = KW * delta_dir

            # tar_cur_dir = self.task_coord[task_index,:]-self.coord
            distance = np.linalg.norm(self.task_coord[task_index,:]-self.coord, axis=-1)
            v = np.minimum(distance+3,6)
            # v = np.minimum(distance+4,6)
            # v = KV * v
            v = KV*(1-(np.absolute(delta_dir)/np.pi))*v
            # v = KV*np.linalg.norm(new_vel)
            # v = KV * (1-np.abs(self.angular_velocity)/np.pi)
            # print('w=',w,'v=',v, file=sys.stderr)
            self.last_carry = self.carrying_item
            return w, v
        
        def velocity_obstacle(self,task_index:int):
            # vo_left = [0, 0]
            # vo_right = [0, 0]
            other_robots = np.delete(robot_coord, self.index, axis=0)
            vo_boundary = []
            for robot in other_robots:
                obs_relative_pos = np.float64(np.array(robot) - self.coord)
                obs_distance = np.linalg.norm(obs_relative_pos)
                if obs_distance > VO_distance_range:
                    continue
                obs_dir = np.arctan2(obs_relative_pos[1], obs_relative_pos[0]) # [-pi,pi]
                if obs_dir < 0: obs_dir = 2*np.pi + obs_dir # [0,2pi]
                obs_distance = max(obs_distance, 4*0.53)
                alpha = np.arcsin(4*0.53/obs_distance)
                left_boundary = obs_dir + alpha
                right_boundary = obs_dir - alpha
                vo_boundary.append([right_boundary, left_boundary])
            vo_boundary = sorted(vo_boundary, key=lambda x: x[0])
            # print('vo_boundary=',vo_boundary, file=sys.stderr)
            merged_vo_boundary = []
            for boundary in vo_boundary:
                if not merged_vo_boundary:
                    merged_vo_boundary.append(boundary)
                else:
                    last = merged_vo_boundary[-1]
                    if boundary[0] <= last[1]:
                        last[1] = max(last[1], boundary[1])
                    else:
                        merged_vo_boundary.append(boundary)
            
            tar_relative_pos = np.float64(self.task_coord[task_index,:] - self.coord)
            tar_distance = np.linalg.norm(tar_relative_pos)
            tar_dir = np.arctan2(tar_relative_pos[1], tar_relative_pos[0]) # [-pi,pi]
            if tar_dir < 0: tar_dir = 2*np.pi + tar_dir # [0,2pi]
            
            # print('tar_dir=',tar_dir,'boundary=',merged_vo_boundary, file=sys.stderr)
            for boundary in merged_vo_boundary:
                if tar_dir <= boundary[0] or tar_dir >= boundary[1]:
                    continue
                print('触发', file=sys.stderr)
                tar_dir = boundary[0] - 0.25 * np.pi
                
            delta_dir = tar_dir - self.heading
            # print('delta_dir=',delta_dir,'tar_dir=',tar_dir,'heading=',self.heading, file=sys.stderr)
            if delta_dir > np.pi:
                delta_dir = delta_dir - 2*np.pi
            elif delta_dir < -np.pi:
                delta_dir = delta_dir + 2*np.pi
            
            if self.coord[1] > 50 - 1.3:
                if self.heading > 0 and self.heading <= 0.5*np.pi and delta_dir > 0:
                    delta_dir = delta_dir - 2 * np.pi
                elif self.heading > 0.5 * np.pi and self.heading < np.pi and delta_dir < 0:
                    delta_dir = 2 * np.pi + delta_dir
            elif self.coord[1] < 1.3:
                if self.heading < 0 and self.heading >= -0.5*np.pi and delta_dir < 0:
                    delta_dir = 2 * np.pi + delta_dir
                elif self.heading < -0.5*np.pi and self.heading > -np.pi and delta_dir > 0:
                    delta_dir = delta_dir - 2 * np.pi
            elif self.coord[0] < 1.3:
                if self.heading < np.pi and self.heading >= 0.5*np.pi and delta_dir > 0:
                    delta_dir = delta_dir - 2 * np.pi
                elif self.heading > -np.pi and self.heading <= -0.5*np.pi and delta_dir < 0:
                    delta_dir = 2 * np.pi + delta_dir
            elif self.coord[0] > 50 - 1.3:
                if self.heading > 0 and self.heading <= 0.5*np.pi and delta_dir < 0:
                    delta_dir = 2 * np.pi + delta_dir
                elif self.heading < 0 and self.heading >= -0.5*np.pi and delta_dir > 0:
                    delta_dir = delta_dir - 2 * np.pi
                    
            w = KW * delta_dir
            
            dis_two_tar = np.linalg.norm(np.float64(self.task_coord[0,:] - self.task_coord[1,:]))
            if dis_two_tar > 20: 
                temp = 4
            elif dis_two_tar > 10:
                temp = 2
            else:
                temp = 1.5
            v = np.minimum(tar_distance+temp,6)
            v = KV*(1-(np.absolute(delta_dir)/np.pi))*v
            
            return w,v
        def velocity_obstacle2(self,tar_dir):
            other_robots = np.delete(robot_coord, self.index, axis=0)
            vo_boundary = []
            for robot in other_robots:
                obs_relative_pos = np.float64(np.array(robot) - self.coord)
                obs_distance = np.linalg.norm(obs_relative_pos)
                if obs_distance > VO_distance_range:
                    continue
                obs_dir = np.arctan2(obs_relative_pos[1], obs_relative_pos[0]) # [-pi,pi]
                if obs_dir < 0: obs_dir = 2*np.pi + obs_dir # [0,2pi]
                obs_distance = max(obs_distance, 4*0.53)
                alpha = np.arcsin(4*0.53/obs_distance)
                left_boundary = obs_dir + alpha
                right_boundary = obs_dir - alpha
                vo_boundary.append([right_boundary, left_boundary])
            vo_boundary = sorted(vo_boundary, key=lambda x: x[0])
            # print('vo_boundary=',vo_boundary, file=sys.stderr)
            merged_vo_boundary = []
            for boundary in vo_boundary:
                if not merged_vo_boundary:
                    merged_vo_boundary.append(boundary)
                else:
                    last = merged_vo_boundary[-1]
                    if boundary[0] <= last[1]:
                        last[1] = max(last[1], boundary[1])
                    else:
                        merged_vo_boundary.append(boundary)
            for boundary in merged_vo_boundary:
                if tar_dir <= boundary[0] or tar_dir >= boundary[1]:
                    continue
                print('触发', file=sys.stderr)
                tar_dir = boundary[0] - 0.25 * np.pi
            return tar_dir
        
        def normalize_angle(angle):
            return (angle + np.pi) % (2 * np.pi) - np.pi
        
        def calculate_velocity_obstacle(self, other_robot):
            # Calculate relative position and velocity
            relative_position = other_robot - self.coord
            distance = np.linalg.norm(relative_position, axis=-1)
            distance = max(2*0.53,distance)
            sum_radius = 2 * 0.53
            sin_alpha = sum_radius / distance

            # Calculate the range of angles of the VO
            alpha = np.arcsin(sin_alpha)
            beta = np.arctan2(relative_position[1], relative_position[0])
            # angle_min = normalize_angle(beta - 3*alpha)
            # angle_max = normalize_angle(beta + 3*alpha)
            if beta < 0: beta = 2*np.pi + beta
            # if distance >= 5: 
            #     angle_min = (beta - 3*alpha)%(2*np.pi)
            #     angle_max = (beta + 3*alpha)%(2*np.pi)
            # else:
            angle_min = (beta - alpha)%(2*np.pi)
            angle_max = (beta + alpha)%(2*np.pi)
            if 2.8 <= distance <= 30:
                return angle_min, angle_max
            else:
                return 0,0
        
        def boundary_delta_w(self,delta_dir):
            if self.coord[1] > 50 - 1.3:
                if self.heading > 0 and self.heading <= 0.5*np.pi and delta_dir > 0:
                    delta_dir = delta_dir - 2 * np.pi
                elif self.heading > 0.5 * np.pi and self.heading < np.pi and delta_dir < 0:
                    delta_dir = 2 * np.pi + delta_dir
            elif self.coord[1] < 1.3:
                if self.heading < 0 and self.heading >= -0.5*np.pi and delta_dir < 0:
                    delta_dir = 2 * np.pi + delta_dir
                elif self.heading < -0.5*np.pi and self.heading > -np.pi and delta_dir > 0:
                    delta_dir = delta_dir - 2 * np.pi
            elif self.coord[0] < 1.3:
                if self.heading < np.pi and self.heading >= 0.5*np.pi and delta_dir > 0:
                    delta_dir = delta_dir - 2 * np.pi
                elif self.heading > -np.pi and self.heading <= -0.5*np.pi and delta_dir < 0:
                    delta_dir = 2 * np.pi + delta_dir
            elif self.coord[0] > 50 - 1.3:
                if self.heading > 0 and self.heading <= 0.5*np.pi and delta_dir < 0:
                    delta_dir = 2 * np.pi + delta_dir
                elif self.heading < 0 and self.heading >= -0.5*np.pi and delta_dir > 0:
                    delta_dir = delta_dir - 2 * np.pi
            return delta_dir
        
        def v_obstacle(self,task_index):
            max_speed = 3.0
            # new_velocity = choose_velocity(robot, robots, delta_time, max_speed)
            # Calculate the desired velocity
            v_delta_time = 1
            w_delta_time = 0.02
            desired_velocity = (self.task_coord[task_index,:] - self.coord) / v_delta_time
            desired_speed = np.linalg.norm(desired_velocity)
            if desired_speed > max_speed:
                desired_velocity = desired_velocity * max_speed / desired_speed

            # Calculate the velocity obstacles
            other_robots = np.delete(robot_coord, self.index, axis=0)
            velocity_obstacles = [calculate_velocity_obstacle(self, other_robot) for other_robot in other_robots]
            # print('velocity_obstacles=',velocity_obstacles,file=sys.stderr)

            # Choose the best velocity
            best_velocity = None
            best_distance = float('inf')
            best_orientation = None
            for angle in np.linspace(0, 2*np.pi, 180):
                # print('angle=',angle)
                available = True
                candidate_orientation = np.arctan2(desired_velocity[1], desired_velocity[0]) - angle
                for vo in velocity_obstacles:
                    if vo[0] < candidate_orientation and candidate_orientation < vo[1]:
                        available = False
                        break
                if available:
                    best_orientation = candidate_orientation
                    break
            # print('vo=',velocity_obstacles,'desired_or=',np.arctan2(desired_velocity[1], desired_velocity[0]),'best_or=',best_orientation,file=sys.stderr)
            w = normalize_angle(best_orientation - self.heading) / w_delta_time
            if self.coord[0] <= 1.5 or self.coord[0] > 50 - 1.5 or self.coord[1] <= 1.5 or self.coord[1] > 50 - 1.5:
                max_speed = 1.0
            if self.buy_done == True:
                dis_two_tar = np.linalg.norm(np.float64(self.task_coord[0,:] - self.task_coord[1,:]))
                if dis_two_tar >= 10:
                    v = max_speed
                else:
                    v = dis_two_tar/10 * max_speed
            v = max(np.linalg.norm(desired_velocity),1) * max_speed
            # v = max_speed
            
            # for angle in np.linspace(0, np.pi, 180):
            #     candidate_velocity = np.array([np.cos(angle), np.sin(angle)]) * max_speed
            #     candidate_distance = np.linalg.norm(candidate_velocity - desired_velocity)

            #     # Check if candidate_velocity is outside of all the VOs
            #     for vo in velocity_obstacles:
            #         if vo[0] < angle < vo[1]:
            #             break
            #     else:
            #         if candidate_distance < best_distance:
            #             best_velocity = candidate_velocity
            #             best_distance = candidate_distance
            # new_velocity = best_velocity
            # for angle in np.linspace(-np.pi, np.pi, 180):
            #     candidate_velocity = np.array([np.cos(angle), np.sin(angle)]) * max_speed
            #     candidate_distance = np.linalg.norm(candidate_velocity - desired_velocity)

            #     # Check if candidate_velocity is outside of all the VOs
            #     for vo in velocity_obstacles:
            #         if vo[0] < angle < vo[1]:
            #             break
            #     else:
            #         if candidate_distance < best_distance:
            #             best_velocity = candidate_velocity
            #             best_distance = candidate_distance
            # new_velocity = best_velocity
            # print('best_v=',best_velocity,'desired_v=',desired_velocity, file=sys.stderr)
            # desired_orientation = np.arctan2(new_velocity[1], new_velocity[0])
            # w = normalize_angle(desired_orientation - self.heading) / w_delta_time
            # v = np.linalg.norm(new_velocity)
            # i = 0
            # for robot in robot_coord:
            #     relative_position = robot - self.coord
            #     distance = np.linalg.norm(relative_position)
            #     if distance != 0 and distance <= 2 * 0.53 and self.index < i:
            #         # print('index=',self.index,'i=',i, file=sys.stderr)
            #         w = 0
            #         v = 0
            #     i = i + 1
            return w, v
        
        def calculate_target_att(self,task_index):
            tar_d= np.linalg.norm(np.float64(self.coord - self.task_coord[task_index,:]))
            if tar_d > targe_att_range:
                att = k_att * (self.task_coord[task_index,:] - self.coord) / tar_d
            else:
                att = k_att * targe_att_range**6 * (self.task_coord[task_index,:] - self.coord)
            return att
        
        def calculate_bound_rep(self):
            boundary_rep = np.float64(np.array([0, 0]))
            if self.coord[0] <= bound_rep_range:
                boundary_rep[0] = bound_k_rep * (1/(self.coord[0]+1e-6) - 1/bound_rep_range) / (self.coord[0]+1e-6)**2
            elif self.coord[0] > 50 - bound_rep_range:
                boundary_rep[0] = -bound_k_rep * (1/(50-self.coord[0]+1e-6) - 1/bound_rep_range) / (50-self.coord[0]+1e-6)**2
            if self.coord[1] <= bound_rep_range:
                boundary_rep[1] = bound_k_rep * (1/(self.coord[1]+1e-6) - 1/bound_rep_range) / (self.coord[1]+1e-6)**2
            elif self.coord[1] > 50 - bound_rep_range:
                boundary_rep[1] = -bound_k_rep * (1/(50-self.coord[1]+1e-6) - 1/bound_rep_range) / (50-self.coord[1]+1e-6)**2
            return boundary_rep
        
        def final_version(self,task_index):
            # global First_frame
            # if First_frame == True: First_frame = False
            KWW = 50
            # # 得到目标方向
            # relative_with_target = self.task_coord[task_index,:] - self.coord
            # target_orientation = np.arctan2(relative_with_target[1], relative_with_target[0])
            # if target_orientation < 0: target_orientation = 2*np.pi + target_orientation
            
            # 计算目标引力
            tar_att = calculate_target_att(self,task_index)
            # 计算边界斥力
            bound_rep = calculate_bound_rep(self)
            # 计算合力
            target_vector = tar_att + bound_rep
            target_orientation = np.arctan2(target_vector[1], target_vector[0])
            # print('tar_att = ', tar_att, 'bound_rep = ', bound_rep, 'tar_vec=',target_vector, 'tar_ori=',target_orientation, file=sys.stderr)
            
            # 得到限制的方向
            other_robots = np.delete(robot_coord, self.index, axis=0)
            velocity_obstacles = [calculate_velocity_obstacle(self, other_robot) for other_robot in other_robots]
            # 得到最终方向
            if self.heading < 0: head = 2*np.pi + self.heading
            else: head = self.heading
            best_orientation = head
            for angle in np.linspace(0, 2*np.pi, 90):
                available = True
                candidate_orientation = (target_orientation - angle)%(2*np.pi)
                for vo in velocity_obstacles:
                    if vo[0] < candidate_orientation and candidate_orientation < vo[1]:
                        available = False
                        break
                if available:
                    best_orientation = candidate_orientation
                    break
            # print('target_orientation=',target_orientation,'best_orientation=',best_orientation,file=sys.stderr)
            # 得到角速度
            delta_dir = normalize_angle(best_orientation - head)
            # delta_dir = normalize_angle_test(self, best_orientation - self.heading)
            # delta_dir = boundary_delta_w(self,delta_dir)
            # if min(np.linalg.norm(self.task_coord[0,:] - self.coord),np.linalg.norm(self.task_coord[1,:] - self.coord)) < 5:
            #     KWW = 7
            if best_orientation != target_orientation:
                # print('target_orientation=',target_orientation,'best_orientation=',best_orientation,file=sys.stderr)
                KWW = 300
            w = KWW * delta_dir
            # 得到速度
            KVV = 1.3
            distance = np.linalg.norm(self.task_coord[task_index,:]-self.coord, axis=-1)
            v = np.minimum(distance+2.5,6)
            v = KVV * (1-(np.absolute(delta_dir)/np.pi))*v
            temp_coord = self.coord + self.linear_velocity * 0.5
            # if best_orientation != target_orientation:
            #     v = 4
                # print('w=',w,'v=',v,file=sys.stderr)
            # if self.coord[0] <= 0.8 or self.coord[0] > 50 - 0.8 or self.coord[1] <= 0.8 or self.coord[1] > 50 - 0.8:
            # # if temp_coord[0] <= 0 or temp_coord[0] > 50 or temp_coord[1] <= 0 or temp_coord[1] > 50:
            #     v = 2
            return w, v

        sell, buy, destroy = False, False, self.destroy
        if self.task_coord is None:
            w, v = 0, 0
        else:
            if not self.buy_done:
                if self.workbench_id == self.task[0].workbench.index:
                    if self.carrying_item == 0:
                        buy = True
                    self.buy_done = True
                    w = 0
                    v = 0
                else:
                # w, v = velocity_obstacle(self,task_index=0)
                # w,v = potential_field(self,task_index=0)
                # w, v = v_obstacle(self,task_index=0)
                    w, v = final_version(self,task_index=0)
            else:
                if self.workbench_id == self.task[1].workbench.index:
                    if self.carrying_item != 0:
                        sell = True
                    self.sell_done = True
                    w = 0
                    v = 0
                else:
                # w, v = velocity_obstacle(self,task_index=1)
                # w,v = potential_field(self,task_index=1)
                # w, v = v_obstacle(self,task_index=1)
                    w, v = final_version(self,task_index=1)                
        # print('w=',w,'v=',v, file=sys.stderr)
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
            if buy and self.frame_num < 8900:
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
