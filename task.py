import numpy as np
from frame import Map

class Scheduler:
    def __init__(self, adj_mat:np.array, workbench:dict, map_obj:Map) -> None:
        self.adj_mat = adj_mat
        self.workbench = workbench
        self.map = map_obj

    def get_nearest_workbench(self, id:int, robot_pos:int):
        workbench_list = self.workbench[id]
        workbench_list = [i for i in workbench_list if self.map.workbenches[i].product_state]
        dist = self.adj_mat[workbench_list, robot_pos]
        return workbench_list[np.argmin(dist)]
    
    def update(self):
        pass
