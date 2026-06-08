"""
可视化渲染模块 - 温度场与热流密度可视化
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.animation import FuncAnimation
from typing import Optional, List
from config import ThermalConfig
from mesh import MeshGenerator
from solver import FEMSolver


def _setup_chinese_font():
    """配置matplotlib中文字体"""
    font_candidates = [
        'Microsoft YaHei',
        'SimHei',
        'SimSun',
        'KaiTi',
        'Arial Unicode MS',
        'PingFang SC',
        'Noto Sans CJK SC',
        'WenQuanYi Micro Hei',
    ]
    import matplotlib.font_manager as fm
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    for font in font_candidates:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
            break
    plt.rcParams['axes.unicode_minus'] = False


_setup_chinese_font()


class ThermalVisualizer:
    """热传导结果可视化"""

    def __init__(self, config: ThermalConfig, mesh: MeshGenerator, solver: FEMSolver):
        self.config = config
        self.mesh = mesh
        self.solver = solver
        self.triangulation: Optional[tri.Triangulation] = None
        self._build_triangulation()

    def _build_triangulation(self):
        """构建matplotlib三角剖分对象"""
        x = self.mesh.nodes[:, 0]
        y = self.mesh.nodes[:, 1]
        self.triangulation = tri.Triangulation(x, y, self.mesh.elements)

    def plot_mesh(self, show_node_numbers: bool = False,
                  show_element_numbers: bool = False,
                  ax: Optional[plt.Axes] = None) -> plt.Axes:
        """绘制网格"""
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))

        ax.triplot(self.triangulation, 'bo-', markersize=3, linewidth=0.5)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('有限元网格')
        ax.set_aspect('equal')

        if show_node_numbers:
            for i, (x, y) in enumerate(self.mesh.nodes):
                ax.text(x, y, str(i), fontsize=6, ha='center', va='center')

        if show_element_numbers:
            centroids = self.mesh.get_element_centroids()
            for i, (x, y) in enumerate(centroids):
                ax.text(x, y, str(i), fontsize=6, ha='center', va='center', color='red')

        return ax

    def plot_temperature(self, temperature: Optional[np.ndarray] = None,
                         ax: Optional[plt.Axes] = None,
                         show_contours: bool = True,
                         num_contours: int = 15,
                         cmap: str = 'jet') -> plt.Axes:
        """绘制温度场云图"""
        if temperature is None:
            temperature = self.solver.temperature

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 8))

        tpc = ax.tripcolor(self.triangulation, temperature, cmap=cmap, shading='gouraud')

        if show_contours:
            ax.tricontour(self.triangulation, temperature, levels=num_contours,
                          colors='white', linewidths=0.5, alpha=0.7)

        cbar = plt.colorbar(tpc, ax=ax)
        cbar.set_label('温度 (°C)')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('温度场分布')
        ax.set_aspect('equal')

        return ax

    def plot_heat_flux_vectors(self, scale: float = 1.0,
                               arrow_density: int = 1,
                               ax: Optional[plt.Axes] = None) -> plt.Axes:
        """绘制热流密度矢量图"""
        if self.solver.heat_flux_elements is None:
            self.solver.compute_heat_flux()

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 8))

        centroids = self.mesh.get_element_centroids()
        qx = self.solver.heat_flux_elements[:, 0]
        qy = self.solver.heat_flux_elements[:, 1]

        idx = np.arange(0, len(centroids), arrow_density)

        q_mag = np.sqrt(qx ** 2 + qy ** 2)
        q_max = np.max(q_mag) if np.max(q_mag) > 0 else 1.0

        ax.quiver(centroids[idx, 0], centroids[idx, 1],
                  qx[idx], qy[idx],
                  q_mag[idx],
                  scale=q_max * scale,
                  cmap='hot',
                  width=0.002,
                  headwidth=4)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('热流密度矢量分布')
        ax.set_aspect('equal')

        return ax

    def plot_heat_flux_magnitude(self, ax: Optional[plt.Axes] = None,
                                 cmap: str = 'hot') -> plt.Axes:
        """绘制热流密度大小云图"""
        if self.solver.heat_flux_elements is None:
            self.solver.compute_heat_flux()

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 8))

        q_mag = np.sqrt(self.solver.heat_flux_elements[:, 0] ** 2 +
                        self.solver.heat_flux_elements[:, 1] ** 2)

        tpc = ax.tripcolor(self.triangulation, facecolors=q_mag, cmap=cmap)

        cbar = plt.colorbar(tpc, ax=ax)
        cbar.set_label('热流密度 (W/m²)')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('热流密度大小分布')
        ax.set_aspect('equal')

        return ax

    def plot_temperature_profile(self, y: float = None, x: float = None,
                                 ax: Optional[plt.Axes] = None) -> plt.Axes:
        """绘制指定位置的温度分布曲线"""
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        temperature = self.solver.temperature

        if y is not None:
            tolerance = self.config.geometry.height / self.config.geometry.num_elements_y
            node_indices = np.where(np.abs(self.mesh.nodes[:, 1] - y) < tolerance)[0]

            if len(node_indices) > 0:
                sorted_indices = node_indices[np.argsort(self.mesh.nodes[node_indices, 0])]
                ax.plot(self.mesh.nodes[sorted_indices, 0], temperature[sorted_indices], 'b-', linewidth=2)
                ax.set_xlabel('X (m)')
                ax.set_title(f'沿 Y = {y:.3f}m 的温度分布')

        elif x is not None:
            tolerance = self.config.geometry.width / self.config.geometry.num_elements_x
            node_indices = np.where(np.abs(self.mesh.nodes[:, 0] - x) < tolerance)[0]

            if len(node_indices) > 0:
                sorted_indices = node_indices[np.argsort(self.mesh.nodes[node_indices, 1])]
                ax.plot(temperature[sorted_indices], self.mesh.nodes[sorted_indices, 1], 'b-', linewidth=2)
                ax.set_xlabel('温度 (°C)')
                ax.set_title(f'沿 X = {x:.3f}m 的温度分布')
                ax.set_ylabel('Y (m)')

        ax.grid(True, alpha=0.3)

        return ax

    def plot_combined(self, show_flux_vectors: bool = True) -> None:
        """绘制组合视图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        self.plot_temperature(ax=axes[0, 0])

        self.plot_heat_flux_magnitude(ax=axes[0, 1])

        if show_flux_vectors:
            self.plot_heat_flux_vectors(ax=axes[1, 0])
        else:
            self.plot_mesh(ax=axes[1, 0])

        mid_y = self.config.geometry.height / 2.0
        self.plot_temperature_profile(y=mid_y, ax=axes[1, 1])

        plt.tight_layout()

    def animate_transient(self, time_points: np.ndarray,
                          temperature_history: np.ndarray,
                          interval: int = 50,
                          cmap: str = 'jet',
                          save_path: Optional[str] = None) -> FuncAnimation:
        """
        创建瞬态温度场动画
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        vmin = np.min(temperature_history)
        vmax = np.max(temperature_history)

        tpc = ax.tripcolor(self.triangulation, temperature_history[0],
                           cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)

        cbar = plt.colorbar(tpc, ax=ax)
        cbar.set_label('温度 (°C)')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_aspect('equal')

        title = ax.set_title(f'温度场 - t = {time_points[0]:.4f} s')

        def update(frame):
            tpc.set_array(temperature_history[frame])
            title.set_text(f'温度场 - t = {time_points[frame]:.4f} s')
            return tpc,

        anim = FuncAnimation(fig, update, frames=len(time_points),
                             interval=interval, blit=False)

        if save_path:
            anim.save(save_path, writer='pillow', fps=20)

        return anim

    def plot_temperature_history(self, time_points: np.ndarray,
                                 temperature_history: np.ndarray,
                                 probe_points: Optional[List[tuple]] = None,
                                 ax: Optional[plt.Axes] = None) -> plt.Axes:
        """绘制监测点的温度历史曲线"""
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        if probe_points is None:
            w = self.config.geometry.width
            h = self.config.geometry.height
            probe_points = [
                (0.0, 0.0),
                (w / 2, h / 2),
                (w, h),
            ]

        for (px, py) in probe_points:
            dists = np.sqrt(
                (self.mesh.nodes[:, 0] - px) ** 2 +
                (self.mesh.nodes[:, 1] - py) ** 2
            )
            nearest_idx = np.argmin(dists)
            actual_x, actual_y = self.mesh.nodes[nearest_idx]
            temps = temperature_history[:, nearest_idx]
            ax.plot(time_points, temps, linewidth=2,
                    label=f'({actual_x:.2f}, {actual_y:.2f})m')

        ax.set_xlabel('时间 (s)')
        ax.set_ylabel('温度 (°C)')
        ax.set_title('监测点温度随时间变化')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def plot_boundary_temperatures(self, edge: str = "left",
                                   ax: Optional[plt.Axes] = None) -> plt.Axes:
        """绘制边界温度分布"""
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        edge_nodes = self.config.get_edge_nodes(self.mesh.nodes, edge)

        if edge in ["left", "right"]:
            sorted_indices = np.argsort(self.mesh.nodes[edge_nodes, 1])
            y_vals = self.mesh.nodes[edge_nodes[sorted_indices], 1]
            t_vals = self.solver.temperature[edge_nodes[sorted_indices]]
            ax.plot(y_vals, t_vals, 'b-', linewidth=2)
            ax.set_xlabel('Y (m)')
        else:
            sorted_indices = np.argsort(self.mesh.nodes[edge_nodes, 0])
            x_vals = self.mesh.nodes[edge_nodes[sorted_indices], 0]
            t_vals = self.solver.temperature[edge_nodes[sorted_indices]]
            ax.plot(x_vals, t_vals, 'b-', linewidth=2)
            ax.set_xlabel('X (m)')

        ax.set_ylabel('温度 (°C)')
        ax.set_title(f'{edge}边界温度分布')
        ax.grid(True, alpha=0.3)

        return ax

    def show(self):
        """显示所有图形"""
        plt.show()

    def save_figure(self, filename: str, dpi: int = 150):
        """保存当前图形"""
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')


def create_visualization(config: ThermalConfig, mesh: MeshGenerator,
                         solver: FEMSolver) -> ThermalVisualizer:
    """
    便捷函数：创建可视化对象
    """
    return ThermalVisualizer(config, mesh, solver)
