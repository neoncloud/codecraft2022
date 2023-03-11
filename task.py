import numpy as np
from frame import Map
from queue import Queue

class Scheduler:
    def __init__(self, map_obj:Map) -> None:
        self.tasks_queue = Queue()
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
        self.src_to_tgt = {i:[w for t in self.rule[_.type_id] for w in self.wb_type_to_id[t]] for i,_ in enumerate(self.map.workbenches)}

    def get_all_task(self):
        source = list(filter(lambda x: self.map.workbenches[x].product_state, range(len(self.map.workbenches))))    # 找到所有完成生产的工作台
        all_task = []
        for s in source:
            target_candidate = self.src_to_tgt[s]   # 查找所有可能的目标
            target_candidate = list(filter(lambda w: self.wb_id_to_type[s] not in self.map.workbenches[w].material_state, target_candidate))
            # 过滤掉没空位的
            if len(target_candidate)>0:
                dist = self.map.adj_mat[s,np.array(target_candidate)]
                all_task.append((s,target_candidate[np.argmin(dist)])) # 找最近的一个工作台
        return all_task

