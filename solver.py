"""
方程求解模块 - 热传导方程有限元离散与温度场求解
"""

import numpy as np
from typing import Optional, Tuple
from config import ThermalConfig
from mesh import MeshGenerator


class FEMSolver:
    """有限元热传导求解器"""

    def __init__(self, config: ThermalConfig, mesh: MeshGenerator):
        self.config = config
        self.mesh = mesh
        self.K: Optional[np.ndarray] = None
        self.M: Optional[np.ndarray] = None
        self.F: Optional[np.ndarray] = None
        self.temperature: Optional[np.ndarray] = None
        self.heat_flux_elements: Optional[np.ndarray] = None

    def _element_stiffness_matrix(self, elem_idx: int) -> np.ndarray:
        """计算单元刚度矩阵（传导矩阵）"""
        n1, n2, n3 = self.mesh.elements[elem_idx]
        x1, y1 = self.mesh.nodes[n1]
        x2, y2 = self.mesh.nodes[n2]
        x3, y3 = self.mesh.nodes[n3]

        area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

        b = np.array([y2 - y3, y3 - y1, y1 - y2]) / (2 * area)
        c = np.array([x3 - x2, x1 - x3, x2 - x1]) / (2 * area)

        k = self.config.material.thermal_conductivity

        Ke = k * area * (np.outer(b, b) + np.outer(c, c))
        return Ke

    def _element_mass_matrix(self, elem_idx: int, lumped: bool = True) -> np.ndarray:
        """计算单元质量矩阵（热容矩阵）
        lumped=True: 集中质量矩阵（推荐，保持解的正性）
        """
        n1, n2, n3 = self.mesh.elements[elem_idx]
        x1, y1 = self.mesh.nodes[n1]
        x2, y2 = self.mesh.nodes[n2]
        x3, y3 = self.mesh.nodes[n3]

        area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

        rho = self.config.material.density
        cp = self.config.material.specific_heat

        if lumped:
            val = rho * cp * area / 3.0
            Me = np.diag([val, val, val])
        else:
            Me = rho * cp * area / 12.0 * np.array([
                [2.0, 1.0, 1.0],
                [1.0, 2.0, 1.0],
                [1.0, 1.0, 2.0]
            ])
        return Me

    def _assemble_stiffness_matrix(self):
        """组装整体刚度矩阵"""
        n = self.mesh.num_nodes
        self.K = np.zeros((n, n))

        for elem_idx in range(self.mesh.num_elements):
            Ke = self._element_stiffness_matrix(elem_idx)
            nodes = self.mesh.elements[elem_idx]
            for i in range(3):
                for j in range(3):
                    self.K[nodes[i], nodes[j]] += Ke[i, j]

    def _assemble_mass_matrix(self):
        """组装整体质量矩阵"""
        n = self.mesh.num_nodes
        self.M = np.zeros((n, n))

        for elem_idx in range(self.mesh.num_elements):
            Me = self._element_mass_matrix(elem_idx)
            nodes = self.mesh.elements[elem_idx]
            for i in range(3):
                for j in range(3):
                    self.M[nodes[i], nodes[j]] += Me[i, j]

    def _assemble_load_vector(self):
        """组装载荷向量"""
        n = self.mesh.num_nodes
        self.F = np.zeros(n)

        for source in self.config.heat_sources:
            x0, y0 = source.position
            if source.is_point_source:
                dists = np.sqrt(
                    (self.mesh.nodes[:, 0] - x0) ** 2 +
                    (self.mesh.nodes[:, 1] - y0) ** 2
                )
                nearest_idx = np.argmin(dists)
                self.F[nearest_idx] += source.magnitude
            else:
                r = source.radius
                for elem_idx in range(self.mesh.num_elements):
                    elem_nodes = self.mesh.elements[elem_idx]
                    centroid = np.mean(self.mesh.nodes[elem_nodes], axis=0)
                    dist = np.sqrt((centroid[0] - x0) ** 2 + (centroid[1] - y0) ** 2)
                    if dist < r:
                        n1, n2, n3 = elem_nodes
                        x1, y1 = self.mesh.nodes[n1]
                        x2, y2 = self.mesh.nodes[n2]
                        x3, y3 = self.mesh.nodes[n3]
                        area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                        q = source.magnitude * area / 3.0
                        self.F[n1] += q
                        self.F[n2] += q
                        self.F[n3] += q

    def _apply_dirichlet_bc(self):
        """施加Dirichlet边界条件（只修改行法，保持列不变）"""
        for bc in self.config.boundary_conditions:
            if bc.bc_type != "dirichlet":
                continue

            edge_nodes = self.config.get_edge_nodes(self.mesh.nodes, bc.edge)
            for node_idx in edge_nodes:
                self.K[node_idx, :] = 0.0
                self.K[node_idx, node_idx] = 1.0
                self.F[node_idx] = bc.temperature

    def _apply_neumann_bc(self):
        """施加Neumann边界条件"""
        for bc in self.config.boundary_conditions:
            if bc.bc_type != "neumann":
                continue

            edge_nodes = self.config.get_edge_nodes(self.mesh.nodes, bc.edge)
            edge_nodes_sorted = self._sort_edge_nodes(edge_nodes, bc.edge)

            for i in range(len(edge_nodes_sorted) - 1):
                n1 = edge_nodes_sorted[i]
                n2 = edge_nodes_sorted[i + 1]
                length = np.linalg.norm(self.mesh.nodes[n1] - self.mesh.nodes[n2])
                q = bc.heat_flux * length / 2.0
                self.F[n1] += q
                self.F[n2] += q

    def _apply_convection_bc(self):
        """施加对流边界条件（Robin边界）"""
        for bc in self.config.boundary_conditions:
            if bc.bc_type != "convection":
                continue

            edge_nodes = self.config.get_edge_nodes(self.mesh.nodes, bc.edge)
            edge_nodes_sorted = self._sort_edge_nodes(edge_nodes, bc.edge)
            h = bc.heat_transfer_coeff
            T_inf = bc.ambient_temperature

            for i in range(len(edge_nodes_sorted) - 1):
                n1 = edge_nodes_sorted[i]
                n2 = edge_nodes_sorted[i + 1]
                length = np.linalg.norm(self.mesh.nodes[n1] - self.mesh.nodes[n2])

                self.K[n1, n1] += h * length / 3.0
                self.K[n1, n2] += h * length / 6.0
                self.K[n2, n1] += h * length / 6.0
                self.K[n2, n2] += h * length / 3.0

                self.F[n1] += h * T_inf * length / 2.0
                self.F[n2] += h * T_inf * length / 2.0

    def _sort_edge_nodes(self, nodes: np.ndarray, edge: str) -> np.ndarray:
        """对边界节点进行排序"""
        if edge in ["left", "right"]:
            sorted_indices = np.argsort(self.mesh.nodes[nodes, 1])
        else:
            sorted_indices = np.argsort(self.mesh.nodes[nodes, 0])
        return nodes[sorted_indices]

    def solve_steady_state(self) -> np.ndarray:
        """求解稳态热传导问题"""
        self._assemble_stiffness_matrix()
        self._assemble_load_vector()

        self._apply_neumann_bc()
        self._apply_convection_bc()
        self._apply_dirichlet_bc()

        self.temperature = np.linalg.solve(self.K, self.F)
        return self.temperature

    def solve_transient(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        求解瞬态热传导问题
        使用向后欧拉法（隐式）
        采用行修改法施加Dirichlet边界条件
        """
        self._assemble_stiffness_matrix()
        self._assemble_mass_matrix()
        self._assemble_load_vector()

        self._apply_neumann_bc()
        self._apply_convection_bc()

        dt = self.config.solver.time_step
        total_time = self.config.solver.total_time
        num_steps = int(total_time / dt) + 1

        self.temperature = np.ones(self.mesh.num_nodes) * self.config.solver.initial_temperature

        time_points = np.zeros(num_steps)
        temperature_history = np.zeros((num_steps, self.mesh.num_nodes))
        temperature_history[0] = self.temperature.copy()

        A = self.M + dt * self.K
        F_load = dt * self.F.copy()

        dirichlet_nodes = []
        dirichlet_values = []
        for bc in self.config.boundary_conditions:
            if bc.bc_type == "dirichlet":
                nodes = self.config.get_edge_nodes(self.mesh.nodes, bc.edge)
                dirichlet_nodes.extend(nodes.tolist())
                dirichlet_values.extend([bc.temperature] * len(nodes))

        dirichlet_nodes = np.array(dirichlet_nodes, dtype=np.int32)
        dirichlet_values = np.array(dirichlet_values)

        A_bc = A.copy()
        for idx, node in enumerate(dirichlet_nodes):
            A_bc[node, :] = 0.0
            A_bc[node, node] = 1.0

        for step in range(1, num_steps):
            b = self.M @ self.temperature + F_load
            b[dirichlet_nodes] = dirichlet_values

            self.temperature = np.linalg.solve(A_bc, b)

            time_points[step] = step * dt
            temperature_history[step] = self.temperature.copy()

        return time_points, temperature_history

    def compute_heat_flux(self) -> np.ndarray:
        """计算每个单元的热流密度矢量"""
        self.heat_flux_elements = np.zeros((self.mesh.num_elements, 2))
        k = self.config.material.thermal_conductivity

        for elem_idx in range(self.mesh.num_elements):
            n1, n2, n3 = self.mesh.elements[elem_idx]
            x1, y1 = self.mesh.nodes[n1]
            x2, y2 = self.mesh.nodes[n2]
            x3, y3 = self.mesh.nodes[n3]

            area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

            b = np.array([y2 - y3, y3 - y1, y1 - y2]) / (2 * area)
            c = np.array([x3 - x2, x1 - x3, x2 - x1]) / (2 * area)

            T1 = self.temperature[n1]
            T2 = self.temperature[n2]
            T3 = self.temperature[n3]

            dTdx = b[0] * T1 + b[1] * T2 + b[2] * T3
            dTdy = c[0] * T1 + c[1] * T2 + c[2] * T3

            self.heat_flux_elements[elem_idx, 0] = -k * dTdx
            self.heat_flux_elements[elem_idx, 1] = -k * dTdy

        return self.heat_flux_elements

    def compute_nodal_heat_flux(self) -> np.ndarray:
        """计算节点热流密度（通过单元平均）"""
        if self.heat_flux_elements is None:
            self.compute_heat_flux()

        nodal_flux = np.zeros((self.mesh.num_nodes, 2))
        count = np.zeros(self.mesh.num_nodes)

        for elem_idx in range(self.mesh.num_elements):
            nodes = self.mesh.elements[elem_idx]
            for node in nodes:
                nodal_flux[node] += self.heat_flux_elements[elem_idx]
                count[node] += 1.0

        for i in range(self.mesh.num_nodes):
            if count[i] > 0:
                nodal_flux[i] /= count[i]

        return nodal_flux

    def get_temperature_gradient(self) -> np.ndarray:
        """计算温度梯度"""
        gradient = np.zeros((self.mesh.num_elements, 2))

        for elem_idx in range(self.mesh.num_elements):
            n1, n2, n3 = self.mesh.elements[elem_idx]
            x1, y1 = self.mesh.nodes[n1]
            x2, y2 = self.mesh.nodes[n2]
            x3, y3 = self.mesh.nodes[n3]

            area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

            b = np.array([y2 - y3, y3 - y1, y1 - y2]) / (2 * area)
            c = np.array([x3 - x2, x1 - x3, x2 - x1]) / (2 * area)

            T1 = self.temperature[n1]
            T2 = self.temperature[n2]
            T3 = self.temperature[n3]

            gradient[elem_idx, 0] = b[0] * T1 + b[1] * T2 + b[2] * T3
            gradient[elem_idx, 1] = c[0] * T1 + c[1] * T2 + c[2] * T3

        return gradient

    def verify_energy_conservation(self) -> dict:
        """
        验证稳态能量守恒
        计算通过边界的热流量与内热源功率是否平衡
        约定：流入域内为正，流出域外为负
        """
        if self.temperature is None:
            raise ValueError("请先求解温度场")

        if self.heat_flux_elements is None:
            self.compute_heat_flux()

        boundary_heat_in = 0.0
        boundary_inflow = 0.0
        boundary_outflow = 0.0
        source_power = 0.0

        edge_elem_map = {}
        for elem_idx in range(self.mesh.num_elements):
            n1, n2, n3 = self.mesh.elements[elem_idx]
            edges = [
                (tuple(sorted([n1, n2])), n3),
                (tuple(sorted([n2, n3])), n1),
                (tuple(sorted([n3, n1])), n2),
            ]
            for edge, opposite_node in edges:
                if edge not in edge_elem_map:
                    edge_elem_map[edge] = []
                edge_elem_map[edge].append((elem_idx, opposite_node))

        boundary_edges = []
        for edge, elems in edge_elem_map.items():
            if len(elems) == 1:
                boundary_edges.append((edge, elems[0]))

        for edge, (elem_idx, opp_node) in boundary_edges:
            n1, n2 = edge
            p1 = self.mesh.nodes[n1]
            p2 = self.mesh.nodes[n2]
            p_opp = self.mesh.nodes[opp_node]

            edge_vec = p2 - p1
            length = np.linalg.norm(edge_vec)

            outward_normal = np.array([edge_vec[1], -edge_vec[0]]) / length

            centroid = (p1 + p2 + p_opp) / 3.0
            if np.dot(p_opp - centroid, outward_normal) > 0:
                outward_normal = -outward_normal

            qx = self.heat_flux_elements[elem_idx, 0]
            qy = self.heat_flux_elements[elem_idx, 1]

            q_out = qx * outward_normal[0] + qy * outward_normal[1]
            q_in = -q_out
            heat_in = q_in * length

            boundary_heat_in += heat_in
            if heat_in > 0:
                boundary_inflow += heat_in
            else:
                boundary_outflow += -heat_in

        for source in self.config.heat_sources:
            if source.is_point_source:
                source_power += source.magnitude
            else:
                r = source.radius
                x0, y0 = source.position
                for elem_idx in range(self.mesh.num_elements):
                    elem_nodes = self.mesh.elements[elem_idx]
                    centroid = np.mean(self.mesh.nodes[elem_nodes], axis=0)
                    dist = np.sqrt((centroid[0] - x0) ** 2 + (centroid[1] - y0) ** 2)
                    if dist < r:
                        n1, n2, n3 = elem_nodes
                        x1, y1 = self.mesh.nodes[n1]
                        x2, y2 = self.mesh.nodes[n2]
                        x3, y3 = self.mesh.nodes[n3]
                        area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                        source_power += source.magnitude * area

        total_inflow = boundary_heat_in + source_power
        
        boundary_total = max(boundary_inflow, boundary_outflow)
        if abs(source_power) > 1e-10:
            reference = abs(source_power)
        elif boundary_total > 1e-10:
            reference = boundary_total
        else:
            reference = 1.0
        
        relative_error = abs(total_inflow) / reference * 100

        return {
            "boundary_heat_inflow": boundary_heat_in,
            "boundary_inflow": boundary_inflow,
            "boundary_outflow": boundary_outflow,
            "internal_heat_source": source_power,
            "total_inflow": total_inflow,
            "relative_error": relative_error
        }

    def compare_with_analytical(self, analytical_func) -> dict:
        """
        与解析解对比
        analytical_func: 接受(x, y)返回温度的函数
        """
        if self.temperature is None:
            raise ValueError("请先求解温度场")

        analytical_temp = np.zeros(self.mesh.num_nodes)
        for i in range(self.mesh.num_nodes):
            analytical_temp[i] = analytical_func(self.mesh.nodes[i, 0], self.mesh.nodes[i, 1])

        error = self.temperature - analytical_temp
        l2_error = np.sqrt(np.mean(error ** 2))
        max_error = np.max(np.abs(error))
        avg_error = np.mean(np.abs(error))

        return {
            "l2_error": l2_error,
            "max_error": max_error,
            "avg_error": avg_error,
            "analytical_temperature": analytical_temp,
            "error_field": error
        }

    def info(self) -> str:
        """返回求解器信息"""
        info_str = "=== 求解器信息 ===\n"
        info_str += f"求解类型: {'稳态' if self.config.solver.steady_state else '瞬态'}\n"
        if self.temperature is not None:
            info_str += f"温度范围: {np.min(self.temperature):.2f} ~ {np.max(self.temperature):.2f} °C\n"
            info_str += f"平均温度: {np.mean(self.temperature):.2f} °C\n"
        return info_str


def solve_thermal_problem(config: ThermalConfig, mesh: MeshGenerator) -> FEMSolver:
    """
    便捷函数：求解热传导问题
    """
    solver = FEMSolver(config, mesh)

    if config.solver.steady_state:
        solver.solve_steady_state()
    else:
        solver.solve_transient()

    solver.compute_heat_flux()
    return solver
