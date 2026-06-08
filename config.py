"""
参数配置模块 - 材料热物性参数与边界条件配置
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Union


BoundaryCondition = Union["DirichletBC", "NeumannBC", "ConvectionBC"]


@dataclass
class MaterialProperties:
    """材料热物性参数"""
    thermal_conductivity: float = 100.0
    density: float = 7850.0
    specific_heat: float = 500.0
    name: str = "steel"

    def thermal_diffusivity(self) -> float:
        return self.thermal_conductivity / (self.density * self.specific_heat)


@dataclass
class GeometryConfig:
    """求解域几何配置"""
    width: float = 1.0
    height: float = 1.0
    num_elements_x: int = 20
    num_elements_y: int = 20


@dataclass
class DirichletBC:
    """Dirichlet边界条件（固定温度）"""
    temperature: float
    edge: str
    bc_type: str = "dirichlet"


@dataclass
class NeumannBC:
    """Neumann边界条件（热流密度）"""
    heat_flux: float
    edge: str
    bc_type: str = "neumann"


@dataclass
class ConvectionBC:
    """对流边界条件（Robin边界）"""
    heat_transfer_coeff: float
    ambient_temperature: float
    edge: str
    bc_type: str = "convection"


@dataclass
class HeatSource:
    """内热源配置"""
    magnitude: float
    position: Tuple[float, float]
    radius: float = 0.0
    is_point_source: bool = True


@dataclass
class SolverConfig:
    """求解器配置"""
    steady_state: bool = True
    time_step: float = 0.01
    total_time: float = 1.0
    initial_temperature: float = 25.0


class ThermalConfig:
    """热传导模拟完整配置"""

    def __init__(self):
        self.material = MaterialProperties()
        self.geometry = GeometryConfig()
        self.solver = SolverConfig()
        self.boundary_conditions: List[BoundaryCondition] = []
        self.heat_sources: List[HeatSource] = []
        self._setup_default_bcs()

    def _setup_default_bcs(self):
        """设置默认边界条件"""
        self.boundary_conditions = [
            DirichletBC(temperature=100.0, edge="left"),
            DirichletBC(temperature=25.0, edge="right"),
        ]

    def set_material(self, thermal_conductivity: float, density: float,
                     specific_heat: float, name: str = "custom"):
        """设置材料参数"""
        self.material = MaterialProperties(
            thermal_conductivity=thermal_conductivity,
            density=density,
            specific_heat=specific_heat,
            name=name
        )

    def set_geometry(self, width: float, height: float,
                     nx: int, ny: int):
        """设置几何参数"""
        self.geometry = GeometryConfig(
            width=width,
            height=height,
            num_elements_x=nx,
            num_elements_y=ny
        )

    def add_dirichlet_bc(self, temperature: float, edge: str):
        """添加Dirichlet边界条件"""
        valid_edges = ["left", "right", "top", "bottom"]
        if edge not in valid_edges:
            raise ValueError(f"边必须是以下之一: {valid_edges}")
        self.boundary_conditions.append(
            DirichletBC(temperature=temperature, edge=edge)
        )

    def add_neumann_bc(self, heat_flux: float, edge: str):
        """添加Neumann边界条件"""
        valid_edges = ["left", "right", "top", "bottom"]
        if edge not in valid_edges:
            raise ValueError(f"边必须是以下之一: {valid_edges}")
        self.boundary_conditions.append(
            NeumannBC(heat_flux=heat_flux, edge=edge)
        )

    def add_convection_bc(self, h: float, T_ambient: float, edge: str):
        """添加对流边界条件"""
        valid_edges = ["left", "right", "top", "bottom"]
        if edge not in valid_edges:
            raise ValueError(f"边必须是以下之一: {valid_edges}")
        self.boundary_conditions.append(
            ConvectionBC(heat_transfer_coeff=h, ambient_temperature=T_ambient, edge=edge)
        )

    def add_point_heat_source(self, magnitude: float, x: float, y: float):
        """添加点热源"""
        self.heat_sources.append(
            HeatSource(magnitude=magnitude, position=(x, y), is_point_source=True)
        )

    def add_area_heat_source(self, magnitude: float, x: float, y: float, radius: float):
        """添加面热源"""
        self.heat_sources.append(
            HeatSource(magnitude=magnitude, position=(x, y),
                       radius=radius, is_point_source=False)
        )

    def clear_boundary_conditions(self):
        """清除所有边界条件"""
        self.boundary_conditions = []

    def clear_heat_sources(self):
        """清除所有热源"""
        self.heat_sources = []

    def get_edge_nodes(self, nodes: np.ndarray, edge: str) -> np.ndarray:
        """获取指定边上的节点索引"""
        x_coords = nodes[:, 0]
        y_coords = nodes[:, 1]

        tolerance = 1e-6
        w = self.geometry.width
        h = self.geometry.height

        if edge == "left":
            return np.where(np.abs(x_coords) < tolerance)[0]
        elif edge == "right":
            return np.where(np.abs(x_coords - w) < tolerance)[0]
        elif edge == "bottom":
            return np.where(np.abs(y_coords) < tolerance)[0]
        elif edge == "top":
            return np.where(np.abs(y_coords - h) < tolerance)[0]
        else:
            raise ValueError(f"未知的边: {edge}")

    def set_solver_params(self, steady_state: bool = True, time_step: float = 0.01,
                          total_time: float = 1.0, initial_temp: float = 25.0):
        """设置求解器参数"""
        self.solver = SolverConfig(
            steady_state=steady_state,
            time_step=time_step,
            total_time=total_time,
            initial_temperature=initial_temp
        )

    def info(self) -> str:
        """返回配置信息字符串"""
        info_str = "=== 热传导模拟配置 ===\n"
        info_str += f"材料: {self.material.name}\n"
        info_str += f"  热传导系数: {self.material.thermal_conductivity} W/(m·K)\n"
        info_str += f"  密度: {self.material.density} kg/m³\n"
        info_str += f"  比热容: {self.material.specific_heat} J/(kg·K)\n"
        info_str += f"  热扩散率: {self.material.thermal_diffusivity():.6e} m²/s\n"
        info_str += f"\n几何: {self.geometry.width}m x {self.geometry.height}m\n"
        info_str += f"  网格: {self.geometry.num_elements_x} x {self.geometry.num_elements_y}\n"
        info_str += f"\n边界条件:\n"
        for bc in self.boundary_conditions:
            if bc.bc_type == "dirichlet":
                info_str += f"  {bc.edge}: Dirichlet, T={bc.temperature}°C\n"
            elif bc.bc_type == "neumann":
                info_str += f"  {bc.edge}: Neumann, q={bc.heat_flux} W/m²\n"
            elif bc.bc_type == "convection":
                info_str += f"  {bc.edge}: Convection, h={bc.heat_transfer_coeff} W/(m²·K), T∞={bc.ambient_temperature}°C\n"
        info_str += f"\n热源: {len(self.heat_sources)} 个\n"
        info_str += f"\n求解器:\n"
        info_str += f"  类型: {'稳态' if self.solver.steady_state else '瞬态'}\n"
        if not self.solver.steady_state:
            info_str += f"  时间步长: {self.solver.time_step} s\n"
            info_str += f"  总时间: {self.solver.total_time} s\n"
        info_str += f"  初始温度: {self.solver.initial_temperature}°C\n"
        return info_str
