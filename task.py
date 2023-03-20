import itertools
import sys
from queue import Queue
from typing import List

import numpy as np

from frame import Map, Workbench, Robot

dependency_dict = {
    9: [(7, 6, 5, 4),],
    8: [7,],
    7: [6,5,4,],
    6: [2,3],
    5: [1,3],
    4: [1,2],
    3: [],
    2: [],
    1: []
}

dependency_T = {
    1:[4,5,9],
    2:[4,6,9],
    3:[5,6,9],
    4:[7,9],
    5:[7,9],
    6:[7,9],
    7:[8,9],
    8:[],
    9:[]
}

production_time = {
    9: 0,
    8: 0,
    7: 1000,
    6: 500,
    5: 500,
    4: 500,
    3: 50,
    2: 50,
    1: 50
}

profit = {
    1:3000,
    2:3200,
    3:3400,
    4:7100+3000+3200,
    5:7800+3000+3400,
    6:8300+3200+3400,
    7:29000+7100+3000+3200+7800+3000+3400+8300+3200+3400,
    8:1,
    9:1
}

def print_child(task_node):
    print(task_node.workbench.type_id, task_node.workbench.index, file=sys.stderr)
    if len(task_node.children) == 0:
        return
    print(f"children of {task_node.workbench.type_id}:", file=sys.stderr)
    for c in task_node.children:
        print_child(c)
    print("end", file=sys.stderr)

def flatten(lst:List):
    flattened_lst = []
    for elem in lst:
        if isinstance(elem, list) or isinstance(elem, tuple):
            flattened_lst.extend(flatten(elem))
        else:
            flattened_lst.append(elem)
    return flattened_lst

class DummyTaskNode:
    def __init__(self, workbench:Workbench) -> None:
        self.workbench = workbench

class TaskNode:
    def __init__(self, workbench:Workbench, wb_id_to_type:np.ndarray, adj_mat:np.ndarray, map_obj:Map) -> None:
        self.workbench = workbench # 实际对应的工作台
        self.blacklist = [] # 存储此时行不通的路
        self.children = self.make_children(wb_id_to_type, adj_mat, map_obj)
        self.done = False # 这个flag只能单向从false到true

    def make_children(self, wb_id_to_type:np.ndarray, adj_mat:np.ndarray, workbenches:List[Workbench]):
        dep_nodes = dependency_dict[self.workbench.type_id]
        children = []
        # failed = False
        for d in dep_nodes:
            # 这里不管任务之间是tuple(或关系) 还是int(与关系)，都能同一处理
            # 选出所有可用的工作台
            workbench_candidate = np.argwhere(np.isin(wb_id_to_type, np.array(d))).squeeze()
            workbench_candidate = np.atleast_1d(workbench_candidate) 
            # 筛选条件，要么没材料，要么已经产出了等待进行下一批（意味着材料将被清空）
            workbench_candidate = list(filter(lambda wb_id: wb_id not in self.blacklist and (len(workbenches[wb_id].material_state)==0 or workbenches[wb_id].remaining_time!=0), workbench_candidate))
            if len(workbench_candidate) == 0: # 没有可用的工作台
                return []
            # 查询最高效的
            # 因为不知道d是tuple还是int，因此通过wb_types反查类型id
            min_wb = adj_mat[self.workbench.index, workbench_candidate]
            min_wb = workbench_candidate[min_wb.argmin()]
            min_wb = workbenches[min_wb]
            # 实例化子任务节点
            child = TaskNode(min_wb, wb_id_to_type, adj_mat, workbenches)
            if len(child.children) == 0 and child.workbench.type_id not in (1,2,3): # 没有可用的工作台，则ban掉这个工作台，重新开始
                self.blacklist.append(min_wb.index)
                # failed = True
                break
            children.append(child)
        # if failed:
        #     children = self.make_children(wb_id_to_type, adj_mat, workbenches)
        return children

    def if_all_done(self):
        if len(self.children) == 0:
            return self.done
        return self.done and all([child.if_all_done() for child in self.children])
    
    def get_avail_sub_task(self, robot_coord:np.ndarray):
        # 当前节点任务完成，他的子节点一定生产完毕了
        if self.done:
            return []
        # 当前节点已经产出，或者即将产出
        min_time = np.linalg.norm(robot_coord - self.workbench.coord, 2, -1).min(0)/6.0
        if ((self.workbench.remaining_time <= min_time and self.workbench.remaining_time>=0) or self.workbench.product_state or self.workbench.type_id in (1,2,3)) and not self.workbench.assigned_buy:
            return self
        # 否则递归遍历子节点
        avail_sub_tasks = []
        for child in self.children:
            child_avail_tasks = child.get_avail_sub_task(robot_coord)
            # print(child_avail_tasks, file=sys.stderr)
            if isinstance(child_avail_tasks, list):
                # 返回的是一个列表，说明是中间节点
                avail_sub_tasks += child_avail_tasks
            else: # 返回的是一个workbench，说明是尾节点
                if child_avail_tasks.workbench.type_id in self.workbench.material_state:
                    # 目前被占用了，就先别急
                    continue
                avail_sub_tasks.append((child_avail_tasks, self))
        return avail_sub_tasks


class Scheduler2:
    def __init__(self, map_obj:Map) -> None:
        self.map = map_obj
        self.task_queue = Queue()
        self.sub_tasks = []
        self.wb_type_to_id = {i:[w.index for w in self.map.workbenches if w.type_id==i] for i in range(1,10)}
        self.wb_id_to_type = [w.type_id for w in self.map.workbenches]
        self.eff_adj_mat = self.make_eff_adj_mat()
        self.free_robots = []
        self.free_robots_coord = []
        self.task_queue.put(self.make_task())
        self.update_robot()
        self.update_task()


    def make_task(self):
        top_nodes = list(itertools.chain(*[self.wb_type_to_id[i] for i in (9,8)]))
        top_node = np.random.choice(top_nodes, 1).item()
        task = TaskNode(self.map.workbenches[top_node], self.wb_id_to_type, self.eff_adj_mat, self.map.workbenches)
        print("New TaskNode:", file=sys.stderr)
        print_child(task)
        if len(task.children)==0:
            return None
        else:
            return task

    def make_eff_adj_mat(self):
        # 计算质效比邻接矩阵 赶路时间/收益，不邻接的设为999
        eff_adj_mat = self.map.workbench_adj_mat.copy()
        for i, dists in enumerate(self.map.workbench_adj_mat):
            i_wb_type = self.wb_id_to_type[i]
            for j,_ in enumerate(dists):
                j_wb_type = self.wb_id_to_type[j]
                deps = flatten(dependency_dict[i_wb_type])
                if j_wb_type not in deps:
                    eff_adj_mat[i,j] = 999
                else:
                    eff_adj_mat[i,j] += 50 #补偿机器人加速减速时间
                    eff_adj_mat[i,j] /= profit[i_wb_type]
        return eff_adj_mat

    def update_task(self):
        # if len(self.free_robots) <= len(self.sub_tasks):
        #     # 还有活干，没必要更新
        #     return
        avail_sub_tasks = []
        temp = []
        while not self.task_queue.empty():
            task = self.task_queue.get()
            if task.if_all_done(): # 全干完了
                print(f"TaskNode {task} Done!", file=sys.stderr)
                self.task_queue.task_done()
                # del task
            else:
                avail_sub_tasks += task.get_avail_sub_task(self.map.robot_coord)
                temp.append(task)
        for i in temp:
            self.task_queue.put(i)
        # 有机器人无事可做了，就新建一个任务树
        if len(avail_sub_tasks) < len(self.free_robots): 
            task = self.make_task()
            if task is None:
                return
            self.task_queue.put(task)
        # avail_sub_tasks = list(filter(lambda task: task[0].workbench.type_id not in task[1].workbench.assigned_sell, avail_sub_tasks))
        self.sub_tasks = avail_sub_tasks
            # self.update_sub_task()

    def update_robot(self):
        self.free_robots = list(filter(lambda r: r.carrying_item == 0 and r.buy_done and r.sell_done , self.map.robots))
        self.free_robots_coord = np.array([r.coord for r in self.free_robots])
        for r in self.map.robots:
            if r in self.free_robots:
                continue
            src_task = r.task[0]
            tgt_task = r.task[1]
            if r.buy_done:
                if r.carrying_item != 0: #成功买到
                    src_task.workbench.assigned_buy = False
                else:
                    r.buy_done = True
                    r.sell_done = True
                if np.linalg.norm(tgt_task.workbench.coord - r.coord, 2 ,-1) < 2.0: # 快要靠近时发生冲突，要卖的东西已经被人抢卖了
                    if r.carrying_item in tgt_task.workbench.material_state:
                        self.reassign(r)
            if r.sell_done:
                if r.carrying_item == 0:
                    r.sell_done = True
                    if src_task.workbench.type_id in tgt_task.workbench.assigned_sell:
                        tgt_task.workbench.assigned_sell.remove(src_task.workbench.type_id)

    def reassign(self, robot:Robot):
        carry_item = robot.carrying_item
        target_candidate = dependency_T[carry_item] # 找个新目标
        target_candidate = list(filter(lambda task: task[1].workbench.type_id in target_candidate, self.sub_tasks))
        target_candidate = list(filter(lambda task: carry_item not in task[1].workbench.material_state, target_candidate))
        if len(target_candidate) == 0:
            return
        min_dist = np.linalg.norm(robot.coord-np.array([task[1].workbench.coord for task in target_candidate]), 2, -1).argmin(0) #选个最近的
        new_task = target_candidate[min_dist]
        robot.task[1].done = False
        if carry_item in robot.task[1].workbench.assigned_sell:
            robot.task[1].workbench.assigned_sell.remove(carry_item)
        robot.task = new_task
        robot.task_coord[1] = new_task[1].workbench.coord


    def dispatch(self):
        # 选择空闲机器人
        if len(self.free_robots) == 0:
            return
        if len(self.sub_tasks) == 0:
            return
        sub_task_src = [s[0] for s in self.sub_tasks]
        task_profit = np.array([[profit[s.workbench.type_id] for s in sub_task_src]]).T

        # 计算距离
        sub_task_src_coord = np.array([s.workbench.coord for s in sub_task_src])
        dists = list(map(lambda r_coord: np.linalg.norm(sub_task_src_coord - r_coord, axis=-1), self.free_robots_coord))
        effi = np.stack(dists, axis=-1)/task_profit # 最优效率

        # 任务分派，使用匈牙利算法
        assignment = linear_sum_assignment(effi)[1]
        assignment = assignment[:len(self.free_robots)]

        for i,r in enumerate(self.free_robots):
            if i >= len(sub_task_src): # 刚好分配了一个虚任务，就别干了
                continue
            task = self.sub_tasks[assignment[i]]
            src = task[0]
            dst = task[1]
            r.task = task
            r.task_coord = np.array([src.workbench.coord, dst.workbench.coord])
            r.buy_done = False
            r.sell_done = False
            src.done = True
            if src.workbench.type_id not in (1,2,3):
                src.workbench.assigned_buy = True
            if dst.workbench.type_id in (8,9):
                dst.done = True
            else:
                dst.workbench.assigned_sell.append(src.workbench.type_id)
            print(f"Robot {r.index} buy {src.workbench.index},{src.workbench.type_id} sell to {dst.workbench.index},{dst.workbench.type_id}", file=sys.stderr)

            # self.sub_tasks[assignment[i]] = None
        # self.sub_tasks = list(filter(lambda x: x is not None, self.sub_tasks))

    def step(self):
        self.update_task()
        self.update_robot()
        self.dispatch()


class Scheduler:
    def __init__(self, map_obj:Map) -> None:
        self.tasks = None
        self.tier_1, self.tier_2, self.tier_3 = [], [], []
        self.ongoing_task = -np.ones((4,2))
        self.map = map_obj
        self.dead_robot = np.ones((4,1))
        self.rule = {
            1:(4,5),
            2:(4,6),
            3:(5,6),
            4:(7,9),
            5:(7,9),
            6:(7,9),
            7:(8,9),
            8:(),
            9:()
        }
        self.wb_type_to_id = {i:[w.index for w in self.map.workbenches if w.type_id==i] for i in range(1,10)}
        self.wb_id_to_type = {w.index:w.type_id for w in self.map.workbenches}
        self.src_to_tgt = {i.index:[w for t in self.rule[i.type_id] for w in self.wb_type_to_id[t]] for i in self.map.workbenches}

    def get_all_task(self):
        # 每个工作台到最近的一个机器人所需时间
        robot_to_wb_dist = np.linalg.norm(self.map.robot_coord[:, None]-self.map.wb_coord[None, :], 2, -1).min(0)/6.0
        # 选择机器人赶过去能恰好赶上的工作台 且 没有被分配
        source = list(filter(lambda w: ((w.remaining_time/50.0 <= robot_to_wb_dist[w.index] and w.remaining_time>0) or w.product_state) and not w.assigned_buy, self.map.workbenches))
        # if self.map.frame_num >=8000:
        #     source = list(filter(lambda w: w.type_id in (1,2,3), source))
        if len(source) == 0:
            return
        tier_1,tier_2,tier_3 = [],[],[]
        for s in source:
            # 查找所有可能的目标
            target_candidate = self.src_to_tgt[s.index]
            # 过滤掉没空位的 和 已经被分配的
            target_candidate = list(filter(lambda w: s.type_id not in self.map.workbenches[w].material_state+self.map.workbenches[w].assigned_sell, target_candidate))
            target_candidate = np.array(target_candidate)
            if len(target_candidate)>0:
                dist = self.map.workbench_adj_mat[s.index, target_candidate]
                # 找最近的5个工作台 a[np.argpartition(a, -5)[-5:]]
                # TODO 将这里的 dist 加入优先级权重，例如工作台的缺件情况
                task = np.array([[s.index, target_candidate[np.argmin(dist)]]])
                if s.type_id in (7,):
                    tier_1.append(task)
                elif s.type_id in (6,5,4):
                    tier_2.append(task)
                else:
                    tier_3.append(task)
        self.tier_1 = tier_1
        self.tier_2 = tier_2
        self.tier_3 = tier_3
        return self.tasks
    
    def init_task(self):
        source = list(filter(lambda w: w.type_id in (1,2,3), self.map.workbenches))
        all_task = []
        for s in source:
            # 查找所有可能的目标
            target_candidate = self.src_to_tgt[s.index]
            # 过滤掉没空位的 和 已经被分配的
            target_candidate = list(filter(lambda w: s.type_id not in self.map.workbenches[w].material_state, target_candidate))
            target_candidate = np.array(target_candidate)
            if len(target_candidate)>0:
                dist = self.map.workbench_adj_mat[s.index, target_candidate]
                # 找最近的一个工作台
                all_task.append([[s.index, target_candidate[np.argmin(dist)]]])
        self.tier_3 = all_task

    def dispatch(self):
        for tier in (self.tier_1, self.tier_2, self.tier_3):
            if len(tier) == 0:
                continue
            tier = np.concatenate(tier, axis=0)
            
            # 选择空闲机器人
            # print(tier, file=sys.stderr)
            free_robots = list(filter(lambda r: r.carrying_item == 0 and r.buy_done and r.sell_done , self.map.robots))
            if len(free_robots) == 0:
                return

            # 计算机器人与各个任务起点的距离
            dists = list(map(lambda r: np.linalg.norm(self.map.wb_coord[tier[:,0]] - r.coord, axis=-1), free_robots)) 
            dists = np.stack(dists, axis=-1)

            # if len(tier) != 4:
            #     dists = self.pad_to_4(dists)
            # 任务分派，使用匈牙利算法
            assignment = linear_sum_assignment(dists)[0]
            assignment = assignment[:len(free_robots)]
            # print(assignment, file=sys.stderr)
            task_sorted = tier[assignment]

            for i,r in enumerate(free_robots):
                if i >= len(tier): # 刚好分配了一个虚任务，就别干了
                    continue
                task = task_sorted[i]
                self.map.workbenches[task[0]].assigned_buy = True
                # 给对应工作台注册一个即将进来的工件种类
                if self.wb_id_to_type[task[1]] not in (8,9):
                    # 8 号和 9号只进不产
                    self.map.workbenches[task[1]].assigned_sell.append(self.wb_id_to_type[task[0]])
                r.task = task
                r.task_coord = self.map.wb_coord[task]
                r.buy_done = False
                r.sell_done = False

    def clear_ongoing(self):
        for r in self.map.robots:
            src = self.map.workbenches[r.task[0]]
            tgt = self.map.workbenches[r.task[1]]
            if r.buy_done:
                if r.carrying_item != 0: #成功买到
                    src.assigned_buy = False
                else:
                    r.buy_done = True
                    r.sell_done = True
            if r.sell_done:
                if r.carrying_item == 0:
                    r.sell_done = True
                    if tgt.type_id not in (8,9):
                        # try:
                        if src.type_id in tgt.assigned_sell:
                            tgt.assigned_sell.remove(src.type_id)
                        elif src.type_id in tgt.material_state:
                            continue
                        # except:
                        #     print(f"remove {src.type_id} from {tgt.index} failed, assigned sell {tgt.assigned_sell}, type {tgt.type_id}, mat_state {tgt.material_state}", file=sys.stderr)
                        #     raise
                else:
                    target_candidate = self.rule[r.carrying_item]
                    target_candidate = [self.wb_type_to_id[tgt] for tgt in target_candidate]
                    target_candidate = list(filter(lambda w: r.carrying_item not in self.map.workbenches[w].material_state+self.map.workbenches[w].assigned_sell, itertools.chain(*target_candidate)))
                    if len(target_candidate) == 0:
                        continue
                    # 分配一个最近的可用目标
                    target_candidate = np.array(target_candidate)
                    dist = np.linalg.norm(self.map.wb_coord[target_candidate] - r.coord, axis=-1)
                    new_target = target_candidate[np.argmin(dist)]
                    r.task[0] = self.wb_type_to_id[r.carrying_item][0]
                    r.task[1] = new_target
                    r.task_coord[1] = self.map.wb_coord[new_target]
                    r.sell_done = False
                    if self.wb_id_to_type[new_target] not in (8,9):
                        # 给对应工作台注册一个即将进来的工件种类
                        self.map.workbenches[new_target].assigned_sell.append(r.carrying_item)

                # if tgt.type_id not in (8,9):
                #     if src.type_id in tgt.assigned_sell:
                #         tgt.assigned_sell.remove(src.type_id)
                #     elif r.carrying_item!=0: # 异常，要卖但是这个工作台没分配
                # if r.carrying_item == 0:
                #     r.freezed = 0
                #     r.destroy = False
                #     continue
                # # 没卖成功
                # print("freezed", file=sys.stderr)
                # r.freezed += 1
                # if r.freezed >= 5:
                #     # 几次都卖不出去，还没有接收目标就直接销毁
                #     r.destroy = True
                #     r.sell_done = True
                #     r.buy_done = True
                #     r.freezed = 0
                #     continue

    def pad_to_4(self, mat):
        """
        对输入的矩阵进行填充，使最短的维度等于4，填充值为9999
        """
        # 获取矩阵的维度
        rows, cols = mat.shape
        
        # 判断最小的维度是否小于4
        if min(rows, cols) < 4:
            # 计算需要填充的数量
            pad_rows = max(0, 4 - rows)
            pad_cols = max(0, 4 - cols)
            
            # 创建填充的矩阵
            padded_matrix = np.ones((rows+pad_rows, cols+pad_cols)) * 9999
            
            # 将输入矩阵的值复制到填充矩阵中
            padded_matrix[:rows, :cols] = mat
            
            # 返回填充后的矩阵
            return padded_matrix
        
        else:
            # 如果最小维度已经大于等于4，则直接返回原始矩阵
            return mat
    
    def step(self):
        self.clear_ongoing()
        self.get_all_task()
        self.dispatch()
        # print(self.ongoing_task, file=sys.stderr)


# 匈牙利算法求解任务分配问题
def linear_sum_assignment(cost_matrix):
    cost_matrix = np.asarray(cost_matrix)
    if len(cost_matrix.shape) != 2:
        raise ValueError("expected a matrix (2-d array), got a %r array"
                         % (cost_matrix.shape,))

    # The algorithm expects more columns than rows in the cost matrix.
    if cost_matrix.shape[1] < cost_matrix.shape[0]:
        cost_matrix = cost_matrix.T
        transposed = True
    else:
        transposed = False

    state = _Hungary(cost_matrix)

    # No need to bother with assignments if one of the dimensions
    # of the cost matrix is zero-length.
    step = None if 0 in cost_matrix.shape else _step1

    while step is not None:
        step = step(state)

    if transposed:
        marked = state.marked.T
    else:
        marked = state.marked
    return np.where(marked == 1)


class _Hungary(object):
    def __init__(self, cost_matrix):
        self.C = cost_matrix.copy()

        n, m = self.C.shape
        self.row_uncovered = np.ones(n, dtype=bool)
        self.col_uncovered = np.ones(m, dtype=bool)
        self.Z0_r = 0
        self.Z0_c = 0
        self.path = np.zeros((n + m, 2), dtype=int)
        self.marked = np.zeros((n, m), dtype=int)

    def _clear_covers(self):
        """Clear all covered matrix cells"""
        self.row_uncovered[:] = True
        self.col_uncovered[:] = True

def _step1(state):
    state.C -= state.C.min(axis=1)[:, np.newaxis]
    for i, j in zip(*np.where(state.C == 0)):
        if state.col_uncovered[j] and state.row_uncovered[i]:
            state.marked[i, j] = 1
            state.col_uncovered[j] = False
            state.row_uncovered[i] = False

    state._clear_covers()
    return _step3


def _step3(state):
    marked = (state.marked == 1)
    state.col_uncovered[np.any(marked, axis=0)] = False

    if marked.sum() < state.C.shape[0]:
        return _step4


def _step4(state):
    # We convert to int as numpy operations are faster on int
    C = (state.C == 0).astype(int)
    covered_C = C * state.row_uncovered[:, np.newaxis]
    covered_C *= np.asarray(state.col_uncovered, dtype=int)
    n = state.C.shape[0]
    m = state.C.shape[1]

    while True:
        # Find an uncovered zero
        row, col = np.unravel_index(np.argmax(covered_C), (n, m))
        if covered_C[row, col] == 0:
            return _step6
        else:
            state.marked[row, col] = 2
            # Find the first starred element in the row
            star_col = np.argmax(state.marked[row] == 1)
            if state.marked[row, star_col] != 1:
                # Could not find one
                state.Z0_r = row
                state.Z0_c = col
                return _step5
            else:
                col = star_col
                state.row_uncovered[row] = False
                state.col_uncovered[col] = True
                covered_C[:, col] = C[:, col] * (
                    np.asarray(state.row_uncovered, dtype=int))
                covered_C[row] = 0


def _step5(state):
    count = 0
    path = state.path
    path[count, 0] = state.Z0_r
    path[count, 1] = state.Z0_c

    while True:
        # Find the first starred element in the col defined by
        # the path.
        row = np.argmax(state.marked[:, path[count, 1]] == 1)
        if state.marked[row, path[count, 1]] != 1:
            # Could not find one
            break
        else:
            count += 1
            path[count, 0] = row
            path[count, 1] = path[count - 1, 1]

        # Find the first prime element in the row defined by the
        # first path step
        col = np.argmax(state.marked[path[count, 0]] == 2)
        if state.marked[row, col] != 2:
            col = -1
        count += 1
        path[count, 0] = path[count - 1, 0]
        path[count, 1] = col

    # Convert paths
    for i in range(count + 1):
        if state.marked[path[i, 0], path[i, 1]] == 1:
            state.marked[path[i, 0], path[i, 1]] = 0
        else:
            state.marked[path[i, 0], path[i, 1]] = 1

    state._clear_covers()
    # Erase all prime markings
    state.marked[state.marked == 2] = 0
    return _step3


def _step6(state):
    # the smallest uncovered value in the matrix
    if np.any(state.row_uncovered) and np.any(state.col_uncovered):
        minval = np.min(state.C[state.row_uncovered], axis=0)
        minval = np.min(minval[state.col_uncovered])
        state.C[~state.row_uncovered] += minval
        state.C[:, state.col_uncovered] -= minval
    return _step4