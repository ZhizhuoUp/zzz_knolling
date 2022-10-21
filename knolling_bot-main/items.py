import numpy as np

class sort():
    
    def __init__(self):

        self.cube_2x2 = np.array([0.015, 0.015, 0.012])
        self.cube_2x3 = np.array([0.023, 0.015, 0.012])
        self.cube_2x4 = np.array([0.033, 0.015, 0.012])
        self.pencil = np.array([0.150, 0.012, 0.012])
        self.error_rate = 0.1
    
    def judge(self, item_xyz):

        list_2x2 = []
        list_2x3 = []
        list_2x4 = []
        list_pencil = []
        others = []

        for i in range(item_xyz.shape[0]):
            if abs(np.sum(item_xyz[i, :] - self.cube_2x2)) < np.sum(self.cube_2x2) * self.error_rate:
                list_2x2.append(item_xyz[i, :])
            
            elif abs(np.sum(item_xyz[i, :] - self.cube_2x3)) < np.sum(self.cube_2x3) * self.error_rate:
                list_2x3.append(item_xyz[i, :])
            
            elif abs(np.sum(item_xyz[i, :] - self.cube_2x4)) < np.sum(self.cube_2x4) * self.error_rate:
                list_2x4.append(item_xyz[i, :])
            
            elif abs(np.sum(item_xyz[i, :] - self.cube_2x3)) < np.sum(self.cube_2x3) * self.error_rate:
                list_pencil.append(item_xyz[i, :])
            
            else:
                others.append(item_xyz[i, :])

        list_2x2 = np.asarray(list_2x2, dtype=np.float32)
        list_2x3 = np.asarray(list_2x3, dtype=np.float32)
        list_2x4 = np.asarray(list_2x4, dtype=np.float32)
        list_pencil = np.asarray(list_pencil, dtype=np.float32)
        others = np.asarray(others, dtype=np.float32)

        return list_2x2, list_2x3, list_2x4, list_pencil, others

        print(list_2x2)
        print(list_2x3)
        print(list_2x4)
        print(list_pencil)
        print(others)
        