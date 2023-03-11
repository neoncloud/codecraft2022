from matplotlib import pyplot as plt
import numpy as np
from frame import Map, parse_input

map_txt_file_path = '/mnt/e/2023huawei/LinuxRelease/LinuxRelease/maps/1.txt'
def read_map():
    # 打开文件并读取每一行，并对每一行进行映射转换
    with open(map_txt_file_path, "r") as f:
        ori_matrix = np.array([list(line.strip()) for line in f])
    search_chars = ['A', '1', '2', '3', '4', '5', '6', '7','8']
    char_to_coord = {}
    # 查找符合条件的元素的下标，并将下标和对应的值储存在字典中
    for char in search_chars:
        char_to_coord[char] = {}
        indices = np.where(ori_matrix == char)
        for i in range(len(indices[0])):
            char_to_coord[char][i] = (indices[1][i]*0.5+0.25, 50-(indices[0][i]*0.5+0.25))
    return char_to_coord

def w_v_fun(delta_dir,distance):
    k_w = 1 # 可以调节的参数
    w = k_w * delta_dir
    if distance >= 6:
        v = 6
    else:
        v = distance
    k_v = 1
    v = k_v*(1-(np.absolute(w)/np.pi))*v
    return w, v

def kinema(current_direct,current_coord,target_coord):
    distance = np.linalg.norm(target_coord - current_coord)
    tar_cur_dir = [target_coord[0]-current_coord[0],target_coord[1]-current_coord[1]]
    tar_dir = np.arctan2(tar_cur_dir[1],tar_cur_dir[0]) # 每帧更新
    delta_dir = tar_dir - current_direct
    if np.absolute(delta_dir) > np.pi : delta_dir += 2*np.pi # w的范围是[-pi,pi]
    print('cur_dir=',current_direct,'tar_dir=',tar_dir,'delta_dir=',delta_dir,'distance=',distance)
    w, v = w_v_fun(delta_dir = delta_dir, distance = distance)
    return w,v
    
def get_robot_info(map_obj):
    # 遍历并打印Agent对象的所有属性
    for r in map_obj.robots:
        for attr in dir(r):
            if not callable(getattr(r, attr)) and not attr.startswith("__"):
                print(f"Agent对象属性 {attr}: {getattr(r, attr)}")

    
if __name__ == '__main__':
    # 读取地图txt获得各物体的坐标
    all_coord = read_map()
    print(all_coord)
    map_obj = parse_input(input_string)
    # 假设传入目前点和一个目标点
    cur_point = np.array([10,10])
    tar_point = np.array([5,10])
    cur_direct = np.pi
    robot_info = get_robot_info(map_obj)
    w,v = kinema(cur_direct, cur_point, tar_point)
    print('w=',w,'v=',v)