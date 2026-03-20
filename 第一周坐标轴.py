import json
import numpy as np

class CoordinateSystem:
    def __init__(self, ori_axis, vectors):
        """初始化向量"""
        self.axis = np.array(ori_axis) / np.linalg.norm(ori_axis, axis=1, keepdims=True) # 向量除以模长
        self.vectors = np.array(vectors) / np.linalg.norm(vectors, axis=1, keepdims=True) # np.linalg.norm的参数是(数组，按行/列取，要不要压缩至一维的向量)

    def transform(self, obj_axis):
        """
        坐标系转换和向量的移动
        这个函数我一整个周五和周六的凌晨都花在这上面了
        我也不知道为什么就是错了，明明别人相同的算法没错的啊，ai已经被我问的幻觉都出来了，我没招了
        学长学姐如果你看到这个请帮我看看这个代码的错误在哪 [感谢]
        """
        ori = self.axis
        obj = np.array(obj_axis)
        vec = self.vectors

        obj_norm = np.linalg.norm(obj, axis=1, keepdims=True)
        obj = obj / obj_norm

        ori_inv = (np.linalg.inv(ori))
        vec_new = obj @ np.dot(ori_inv, vec)

        self.vectors = vec_new
        self.axis = obj
        return self.vectors

    def get_scale(self):
        """面积/体积缩放倍数（行列式绝对值）"""
        return abs(np.linalg.det(self.axis))

    def get_projections(self):
        """计算在各自轴上的投影"""
        return np.dot(self.vectors, self.axis)

    def get_angles(self, vec):
        """计算与各自轴的夹角"""
        angles = []
        for j in range(self.axis.shape[0]):
            _axis = self.axis[j, :] # 取第j+1行——取第j+1个向量
            if np.linalg.norm(vec) == 0.0:
                cos = 0.0 # 会有norm的值为0而报错
            else:
                cos = np.dot(vec, _axis) / np.linalg.norm(vec)
                cos = np.clip(cos, -1, 1) # 限制cos的值只能在（-1,1）之间
            angles.append(np.arccos(cos))
        return angles

if __name__ == "__main__":
    """函数的实现过程"""
    with open("data(1).json", 'r') as f:
        data = json.load(f)
    for group in data:
        print("="*100)
        print(f"name: {group["group_name"]}")
        system = CoordinateSystem(group["ori_axis"], group["vectors"])
        for task in group["tasks"]:
            print(f"task: {task["type"]}")
            if task["type"] == "area":
                print(f"area: {system.get_scale()}")
            elif task["type"] == "axis_projection":
                print(f"投影为: \n{system.get_projections()}")
            elif task["type"] == "change_axis":
                print(f"新坐标系下的各向量: \n{system.transform(task["obj_axis"])}")
            elif task["type"] == "axis_angle":
                print("向量与轴的夹角分别为: ")
                for i in range(system.vectors.shape[0]):
                    vector = system.vectors[i,:]
                    print(system.get_angles(vector))