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

    def update(self, map_obj=None):
        if map_obj is not None:
            self.map = map_obj
        source = list(filter(lambda x: self.map.workbenches[x].product_state, range(len(self.map.workbenches))))
        for s in source:
            target_candidate = self.src_to_tgt[s]
            target_candidate = list(filter(lambda w: self.wb_id_to_type[s] not in self.map.workbenches[w].material_state, target_candidate))
            if len(target_candidate)>0:
                dist = self.map.adj_mat[s,np.array(target_candidate)]
                self.tasks_queue.put((s,target_candidate[np.argmin(dist)]))

