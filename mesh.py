"""
网格划分模块 - 二维三角形网格自动生成
"""

import numpy as np
from typing import Tuple, List, Optional
from config import ThermalConfig


class MeshGenerator:
    """二维三角形网格生成器"""

    def __init__(self, config: ThermalConfig):
        self.config = config
        self.nodes: Optional[np.ndarray] = None
        self.elements: Optional[np.ndarray] = None
        self.num_nodes: int = 0
        self.num_elements: int = 0
        self.boundary_edges: Optional[np.ndarray] = None

    def generate_structured_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成结构化三角形网格
        将矩形域划分为nx x ny个矩形，每个矩形分为两个三角形
        """
        nx = self.config.geometry.num_elements_x
        ny = self.config.geometry.num_elements_y
        w = self.config.geometry.width
        h = self.config.geometry.height

        num_nodes_x = nx + 1
        num_nodes_y = ny + 1
        self.num_nodes = num_nodes_x * num_nodes_y

        x_coords = np.linspace(0, w, num_nodes_x)
        y_coords = np.linspace(0, h, num_nodes_y)

        self.nodes = np.zeros((self.num_nodes, 2))
        for j in range(num_nodes_y):
            for i in range(num_nodes_x):
                idx = j * num_nodes_x + i
                self.nodes[idx, 0] = x_coords[i]
                self.nodes[idx, 1] = y_coords[j]

        self.num_elements = 2 * nx * ny
        self.elements = np.zeros((self.num_elements, 3), dtype=np.int32)

        elem_idx = 0
        for j in range(ny):
            for i in range(nx):
                n0 = j * num_nodes_x + i
                n1 = j * num_nodes_x + (i + 1)
                n2 = (j + 1) * num_nodes_x + i
                n3 = (j + 1) * num_nodes_x + (i + 1)

                self.elements[elem_idx, :] = [n0, n1, n3]
                elem_idx += 1
                self.elements[elem_idx, :] = [n0, n3, n2]
                elem_idx += 1

        self._find_boundary_edges()
        return self.nodes, self.elements

    def generate_unstructured_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成非结构化三角形网格（使用Delaunay三角剖分）
        """
        try:
            from scipy.spatial import Delaunay
        except ImportError:
            print("警告: scipy未安装，使用结构化网格代替")
            return self.generate_structured_mesh()

        nx = self.config.geometry.num_elements_x
        ny = self.config.geometry.num_elements_y
        w = self.config.geometry.width
        h = self.config.geometry.height

        num_boundary_x = nx + 1
        num_boundary_y = ny + 1

        boundary_points = []

        for i in range(num_boundary_x):
            boundary_points.append([i * w / nx, 0.0])
        for j in range(1, num_boundary_y):
            boundary_points.append([w, j * h / ny])
        for i in range(num_boundary_x - 2, -1, -1):
            boundary_points.append([i * w / nx, h])
        for j in range(num_boundary_y - 2, 0, -1):
            boundary_points.append([0.0, j * h / ny])

        num_internal = (nx - 1) * (ny - 1)
        internal_points = np.random.rand(num_internal, 2)
        internal_points[:, 0] = internal_points[:, 0] * (w - 0.2) + 0.1
        internal_points[:, 1] = internal_points[:, 1] * (h - 0.2) + 0.1

        all_points = np.vstack([np.array(boundary_points), internal_points])

        tri = Delaunay(all_points)
        self.nodes = all_points
        self.elements = tri.simplices.astype(np.int32)
        self.num_nodes = len(self.nodes)
        self.num_elements = len(self.elements)

        self._find_boundary_edges()
        return self.nodes, self.elements

    def _find_boundary_edges(self):
        """找出所有边界边"""
        edge_count = {}

        for elem in self.elements:
            edges = [
                tuple(sorted([elem[0], elem[1]])),
                tuple(sorted([elem[1], elem[2]])),
                tuple(sorted([elem[2], elem[0]])),
            ]
            for edge in edges:
                if edge in edge_count:
                    edge_count[edge] += 1
                else:
                    edge_count[edge] = 1

        boundary_edges_list = [edge for edge, count in edge_count.items() if count == 1]
        self.boundary_edges = np.array(boundary_edges_list, dtype=np.int32)

    def get_element_centroids(self) -> np.ndarray:
        """计算单元形心坐标"""
        centroids = np.zeros((self.num_elements, 2))
        for i in range(self.num_elements):
            elem_nodes = self.nodes[self.elements[i]]
            centroids[i] = np.mean(elem_nodes, axis=0)
        return centroids

    def get_element_areas(self) -> np.ndarray:
        """计算所有单元的面积"""
        areas = np.zeros(self.num_elements)
        for i in range(self.num_elements):
            n1, n2, n3 = self.elements[i]
            x1, y1 = self.nodes[n1]
            x2, y2 = self.nodes[n2]
            x3, y3 = self.nodes[n3]
            areas[i] = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        return areas

    def get_edge_length(self, edge: np.ndarray) -> float:
        """计算边的长度"""
        n1, n2 = edge
        dx = self.nodes[n1, 0] - self.nodes[n2, 0]
        dy = self.nodes[n1, 1] - self.nodes[n2, 1]
        return np.sqrt(dx * dx + dy * dy)

    def get_mesh_quality(self) -> dict:
        """计算网格质量指标"""
        areas = self.get_element_areas()
        qualities = np.zeros(self.num_elements)

        for i in range(self.num_elements):
            n1, n2, n3 = self.elements[i]
            p1, p2, p3 = self.nodes[n1], self.nodes[n2], self.nodes[n3]

            l1 = np.linalg.norm(p2 - p3)
            l2 = np.linalg.norm(p1 - p3)
            l3 = np.linalg.norm(p1 - p2)

            s = (l1 + l2 + l3) / 2.0
            area = np.sqrt(max(s * (s - l1) * (s - l2) * (s - l3), 0.0))

            if area > 0:
                R = l1 * l2 * l3 / (4 * area)
                r = 2 * area / (l1 + l2 + l3)
                qualities[i] = 2 * r / R
            else:
                qualities[i] = 0.0

        return {
            "num_nodes": self.num_nodes,
            "num_elements": self.num_elements,
            "min_quality": np.min(qualities),
            "max_quality": np.max(qualities),
            "avg_quality": np.mean(qualities),
            "min_area": np.min(areas),
            "max_area": np.max(areas),
            "avg_area": np.mean(areas),
        }

    def refine_mesh(self, refine_indices: Optional[np.ndarray] = None):
        """
        网格加密（均匀加密或局部加密）
        """
        if refine_indices is None:
            refine_indices = np.arange(self.num_elements)

        new_nodes_list = [self.nodes.copy()]
        new_elements_list = []
        node_offset = self.num_nodes

        for elem_idx in range(self.num_elements):
            n1, n2, n3 = self.elements[elem_idx]

            if elem_idx in refine_indices:
                mid12 = (self.nodes[n1] + self.nodes[n2]) / 2.0
                mid23 = (self.nodes[n2] + self.nodes[n3]) / 2.0
                mid31 = (self.nodes[n3] + self.nodes[n1]) / 2.0

                new_nodes_list.append(mid12.reshape(1, -1))
                new_nodes_list.append(mid23.reshape(1, -1))
                new_nodes_list.append(mid31.reshape(1, -1))

                m12 = node_offset
                m23 = node_offset + 1
                m31 = node_offset + 2
                node_offset += 3

                new_elements_list.append([n1, m12, m31])
                new_elements_list.append([m12, n2, m23])
                new_elements_list.append([m31, m23, n3])
                new_elements_list.append([m12, m23, m31])
            else:
                new_elements_list.append([n1, n2, n3])

        self.nodes = np.vstack(new_nodes_list)
        self.elements = np.array(new_elements_list, dtype=np.int32)
        self.num_nodes = len(self.nodes)
        self.num_elements = len(self.elements)

        self._find_boundary_edges()

    def info(self) -> str:
        """返回网格信息字符串"""
        quality = self.get_mesh_quality()
        info_str = "=== 网格信息 ===\n"
        info_str += f"节点数: {quality['num_nodes']}\n"
        info_str += f"单元数: {quality['num_elements']}\n"
        info_str += f"单元质量(最小/最大/平均): {quality['min_quality']:.4f} / {quality['max_quality']:.4f} / {quality['avg_quality']:.4f}\n"
        info_str += f"单元面积(最小/最大/平均): {quality['min_area']:.6f} / {quality['max_area']:.6f} / {quality['avg_area']:.6f} m²\n"
        return info_str


def generate_mesh(config: ThermalConfig, mesh_type: str = "structured") -> MeshGenerator:
    """
    便捷函数：生成网格
    """
    mesh = MeshGenerator(config)
    if mesh_type == "structured":
        mesh.generate_structured_mesh()
    elif mesh_type == "unstructured":
        mesh.generate_unstructured_mesh()
    else:
        raise ValueError(f"未知的网格类型: {mesh_type}")
    return mesh
