import numpy as np
from frame import Map
import sys

class Scheduler:
    def __init__(self, map_obj:Map) -> None:
        self.tasks = None
        self.map = map_obj
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
        self.wb_type_to_id = {i:[w for w,_ in enumerate(self.map.workbenches) if _.type_id==i] for i in range(1,10)}
        self.wb_id_to_type = {i:w.type_id for i,w in enumerate(self.map.workbenches)}
        self.wb_coord = np.array([w.coord for w in self.map.workbenches])
        self.src_to_tgt = {i:[w for t in self.rule[_.type_id] for w in self.wb_type_to_id[t]] for i,_ in enumerate(self.map.workbenches)}

    def get_all_task(self):
        source = list(filter(lambda x: self.map.workbenches[x].product_state, range(self.map.num_workbenches)))    # 找到所有完成生产的工作台
        all_task = []
        for s in source:
            target_candidate = self.src_to_tgt[s]   # 查找所有可能的目标
            target_candidate = list(filter(lambda w: self.wb_id_to_type[s] not in self.map.workbenches[w].material_state, target_candidate))
            # 过滤掉没空位的
            if len(target_candidate)>0:
                dist = self.map.adj_mat[s,np.array(target_candidate)]
                all_task.append([s,target_candidate[np.argmin(dist)]]) # 找最近的一个工作台
        self.tasks = np.array(all_task)
        return self.tasks
    
    def dispatch(self):
        free_robots = list(filter(lambda r: r.carrying_item == 0 and r.task_coord is None, self.map.robots)) # 选择空闲机器人
        # dists = np.stack([r.coord for r in free_robots], axis=-1)
        # print(self.tasks, file=sys.stderr)
        # dists = np.linalg.norm(self.wb_coord[self.tasks[:,0]]-dists, 2, -1)
        dists = list(map(lambda r: np.linalg.norm(self.wb_coord[self.tasks[:,0]] - r.coord, axis=-1), free_robots)) # 计算机器人与各个任务起点的距离
        if len(dists) == 0:
            return
        dists = np.stack(dists, axis=-1)
        assignment = linear_sum_assignment(dists)[0]
        list(map(lambda t,r: setattr(r, 'task', t), self.tasks[assignment], free_robots)) # 根据分配重排任务列表，并分配到机器人上
        list(map(lambda t,r: setattr(r, 'task_coord', t), self.wb_coord[self.tasks[assignment]], free_robots))
    
    def init_task(self):
        source = list(filter(lambda x: self.map.workbenches[x].type_id in (1,2,3), range(self.map.num_workbenches)))    # 找到所有完成生产的工作台
        all_task = []
        for s in source:
            target_candidate = self.src_to_tgt[s]   # 查找所有可能的目标
            target_candidate = list(filter(lambda w: self.wb_id_to_type[s] not in self.map.workbenches[w].material_state, target_candidate))
            # 过滤掉没空位的
            if len(target_candidate)>0:
                dist = self.map.adj_mat[s,np.array(target_candidate)]
                all_task.append([s,target_candidate[np.argmin(dist)]]) # 找最近的一个工作台
        self.tasks = np.array(all_task)
        return self.tasks

    def step(self):
        task = self.get_all_task()
        if len(task) == 0:
            return
        self.dispatch()


# 匈牙利算法求解任务分配问题
def linear_sum_assignment(cost_matrix):
    cost_matrix = np.asarray(cost_matrix)
    if len(cost_matrix.shape) != 2:
        raise ValueError("expected a matrix (2-d array), got a %r array"
                         % (cost_matrix.shape,))

    if cost_matrix.shape[1] < cost_matrix.shape[0]:
        cost_matrix = cost_matrix.T
        transposed = True
    else:
        transposed = False

    state = _Hungary(cost_matrix)

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
        self.row_uncovered[:] = True
        self.col_uncovered[:] = True


def _step1(state):
    state.C -= np.apply_along_axis(np.min, 1, state.C)[:, np.newaxis]
    
    indices = filter(lambda x: state.col_uncovered[x[1]] and state.row_uncovered[x[0]], zip(*np.where(state.C == 0)))
    
    state.marked[tuple(zip(*indices))] = 1
    state.col_uncovered[list(set(x[1] for x in indices))] = False
    state.row_uncovered[list(set(x[0] for x in indices))] = False
    
    state._clear_covers()
    
    return _step3


def _step3(state):
    marked = (state.marked == 1)
    state.col_uncovered[np.any(marked, axis=0)] = False

    if marked.sum() < state.C.shape[0]:
        return _step4


def _step4(state):
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

        col = np.argmax(state.marked[path[count, 0]] == 2)
        if state.marked[row, col] != 2:
            col = -1
        count += 1
        path[count, 0] = path[count - 1, 0]
        path[count, 1] = col

    for i in range(count + 1):
        if state.marked[path[i, 0], path[i, 1]] == 1:
            state.marked[path[i, 0], path[i, 1]] = 0
        else:
            state.marked[path[i, 0], path[i, 1]] = 1

    state._clear_covers()
    state.marked[state.marked == 2] = 0
    return _step3


def _step6(state):
    if np.any(state.row_uncovered) and np.any(state.col_uncovered):
        minval = np.min(state.C[state.row_uncovered], axis=0)
        minval = np.min(minval[state.col_uncovered])
        state.C[~state.row_uncovered] += minval
        state.C[:, state.col_uncovered] -= minval
    return _step4