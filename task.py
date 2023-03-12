import numpy as np
from frame import Map
import sys

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
        source = list(filter(lambda w: w.remaining_time <= robot_to_wb_dist[w.index] and w.remaining_time>=0 and not w.assigned_buy, self.map.workbenches))
        if len(source) == 0:
            return

        tier_1,tier_2,tier_3 = [],[],[]
        for s in source:
            # 查找所有可能的目标
            target_candidate = self.src_to_tgt[s.index]
            # 过滤掉没空位的 和 已经被分配的
            target_candidate = list(filter(lambda w: s.type_id not in self.map.workbenches[w].material_state and not self.map.workbenches[w].assigned_sell, target_candidate))
            target_candidate = np.array(target_candidate)
            if len(target_candidate)>0:
                dist = self.map.workbench_adj_mat[s.index, target_candidate]
                # 找最近的一个工作台
                if s.index in (1,2,3):
                    task_list = tier_3
                if s.index in (4,5,6):
                    task_list = tier_2
                else:
                    task_list = tier_1
                task_list.append([s.index, target_candidate[np.argmin(dist)]])
        self.tier_1 = np.array(tier_1)
        self.tier_2 = np.array(tier_2)
        self.tier_3 = np.array(tier_3)
        return self.tasks
    
    def dispatch(self):
        for tier in (self.tier_1, self.tier_2, self.tier_3):
            if len(tier) == 0:
                continue
            free_robots = list(filter(lambda r: r.carrying_item == 0 and r.task_coord is None, self.map.robots)) # 选择空闲机器人
            if len(free_robots) == 0:
                return
            dists = list(map(lambda r: np.linalg.norm(self.map.wb_coord[tier[:,0]] - r.coord, axis=-1), self.map.robots)) # 计算机器人与各个任务起点的距离
            dists = np.stack(dists, axis=-1)
            dists = self.pad_to_4(dists)
            assignment = linear_sum_assignment(dists)[0]
            assignment = assignment[:self.map.num_robots]
            print(assignment, file=sys.stderr)
            for r in free_robots:
                if r.index >= len(tier): # 刚好分配了一个虚任务，就别干了
                    continue
                task = tier[assignment][r.index]
                self.map.workbenches[task[0]].assigned_buy = True
                if self.wb_id_to_type[task[1]] not in (8,9): # 8 号和 9号只进不产 
                    self.map.workbenches[task[1]].assigned = True
                r.task = task
                r.task_coord = self.map.wb_coord[task]
    
    def init_task(self):
        source = list(filter(lambda w: w.type_id in (1,2,3), self.map.workbenches))
        all_task = []
        for s in source:
            # 查找所有可能的目标
            target_candidate = self.src_to_tgt[s.index]
            # 过滤掉没空位的 和 已经被分配的
            target_candidate = list(filter(lambda w: s.type_id not in self.map.workbenches[w].material_state and not np.isin(w, self.ongoing_task[:, 1]).any(), target_candidate))
            target_candidate = np.array(target_candidate)
            if len(target_candidate)>0:
                dist = self.map.workbench_adj_mat[s.index, target_candidate]
                # 找最近的一个工作台
                all_task.append([s.index, target_candidate[np.argmin(dist)]])
        self.tier_1 = np.array(all_task)
    
    def clear_ongoing(self):
        for r in self.map.robots:
            if r.task is None:
                continue
            if r.workbench_id == r.task[1] and r.carrying_item==0:
            # 位于任务末端，且卖完了东西，则清空任务
                self.map.workbenches[r.task[0]].assigned_buy = False
                self.map.workbenches[r.task[1]].assigned_sell = False
                r.task_coord = None
                r.task = None
    
    def find_dead_robot(self):
        for r in self.map.robots:
            pass

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