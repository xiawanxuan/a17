"""
主程序 - 二维材料热传导有限元数值模拟系统
整合四个核心模块：参数配置、网格划分、方程求解、可视化渲染
"""

import numpy as np
import matplotlib.pyplot as plt
from config import ThermalConfig
from mesh import generate_mesh, MeshGenerator
from solver import solve_thermal_problem, FEMSolver
from visualization import create_visualization, ThermalVisualizer


def example_1_steady_state_dirichlet():
    """
    示例1: 稳态热传导 - 左右两边Dirichlet边界条件
    验证：温度沿x方向线性分布
    """
    print("=" * 60)
    print("示例1: 稳态热传导 - 左右Dirichlet边界")
    print("=" * 60)

    config = ThermalConfig()
    config.set_geometry(width=1.0, height=1.0, nx=30, ny=30)
    config.set_material(thermal_conductivity=100.0, density=7850.0,
                        specific_heat=500.0, name="steel")
    config.clear_boundary_conditions()
    config.add_dirichlet_bc(temperature=100.0, edge="left")
    config.add_dirichlet_bc(temperature=25.0, edge="right")

    print(config.info())

    mesh = generate_mesh(config, mesh_type="structured")
    print(mesh.info())

    solver = solve_thermal_problem(config, mesh)
    print(solver.info())

    visualizer = create_visualization(config, mesh, solver)

    fig = plt.figure(figsize=(16, 12))

    ax1 = plt.subplot(2, 2, 1)
    visualizer.plot_temperature(ax=ax1)
    ax1.set_title("温度场分布 (稳态 Dirichlet 边界)")

    ax2 = plt.subplot(2, 2, 2)
    visualizer.plot_heat_flux_vectors(ax=ax2, arrow_density=5, scale=0.8)
    ax2.set_title("热流密度矢量")

    ax3 = plt.subplot(2, 2, 3)
    visualizer.plot_temperature_profile(y=0.5, ax=ax3)
    ax3.set_title("沿 Y=0.5m 的温度分布")
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(2, 2, 4)
    visualizer.plot_heat_flux_magnitude(ax=ax4)
    ax4.set_title("热流密度大小")

    plt.suptitle("示例1: 稳态热传导 - 左右Dirichlet边界条件", fontsize=14)
    plt.tight_layout()

    return config, mesh, solver, visualizer


def example_2_neumann_boundary():
    """
    示例2: Neumann边界条件 - 顶部热流输入，底部散热
    """
    print("\n" + "=" * 60)
    print("示例2: Neumann边界 - 顶部热流输入")
    print("=" * 60)

    config = ThermalConfig()
    config.set_geometry(width=2.0, height=1.0, nx=40, ny=20)
    config.set_material(thermal_conductivity=200.0, density=2700.0,
                        specific_heat=900.0, name="aluminum")
    config.clear_boundary_conditions()
    config.add_dirichlet_bc(temperature=25.0, edge="bottom")
    config.add_neumann_bc(heat_flux=5000.0, edge="top")
    config.add_dirichlet_bc(temperature=25.0, edge="left")
    config.add_dirichlet_bc(temperature=25.0, edge="right")

    print(config.info())

    mesh = generate_mesh(config, mesh_type="structured")
    print(mesh.info())

    solver = solve_thermal_problem(config, mesh)
    print(solver.info())

    visualizer = create_visualization(config, mesh, solver)

    fig = plt.figure(figsize=(14, 10))

    ax1 = plt.subplot(2, 2, 1)
    visualizer.plot_temperature(ax=ax1, cmap='hot')
    ax1.set_title("温度场分布 (Neumann边界)")

    ax2 = plt.subplot(2, 2, 2)
    visualizer.plot_heat_flux_magnitude(ax=ax2, cmap='hot')
    ax2.set_title("热流密度大小")

    ax3 = plt.subplot(2, 2, 3)
    visualizer.plot_temperature_profile(x=1.0, ax=ax3)
    ax3.set_title("沿中心线(X=1.0m)的温度分布")

    ax4 = plt.subplot(2, 2, 4)
    visualizer.plot_boundary_temperatures(edge="top", ax=ax4)
    ax4.set_title("顶部边界温度分布")

    plt.suptitle("示例2: Neumann边界条件 - 顶部热流输入", fontsize=14)
    plt.tight_layout()

    return config, mesh, solver, visualizer


def example_3_internal_heat_source():
    """
    示例3: 内部点热源
    """
    print("\n" + "=" * 60)
    print("示例3: 内部点热源")
    print("=" * 60)

    config = ThermalConfig()
    config.set_geometry(width=2.0, height=2.0, nx=40, ny=40)
    config.set_material(thermal_conductivity=50.0, density=8900.0,
                        specific_heat=385.0, name="copper")
    config.clear_boundary_conditions()
    config.add_convection_bc(h=100.0, T_ambient=25.0, edge="left")
    config.add_convection_bc(h=100.0, T_ambient=25.0, edge="right")
    config.add_convection_bc(h=100.0, T_ambient=25.0, edge="top")
    config.add_convection_bc(h=100.0, T_ambient=25.0, edge="bottom")

    config.add_point_heat_source(magnitude=5000.0, x=0.8, y=1.2)
    config.add_point_heat_source(magnitude=3000.0, x=1.5, y=0.5)

    print(config.info())

    mesh = generate_mesh(config, mesh_type="structured")
    print(mesh.info())

    solver = solve_thermal_problem(config, mesh)
    print(solver.info())

    visualizer = create_visualization(config, mesh, solver)

    fig = plt.figure(figsize=(14, 10))

    ax1 = plt.subplot(2, 2, 1)
    visualizer.plot_temperature(ax=ax1, cmap='hot', num_contours=20)
    ax1.set_title("温度场分布 (双点热源)")

    ax2 = plt.subplot(2, 2, 2)
    visualizer.plot_heat_flux_vectors(ax=ax2, arrow_density=8, scale=0.5)
    ax2.set_title("热流密度矢量")

    ax3 = plt.subplot(2, 2, 3)
    visualizer.plot_heat_flux_magnitude(ax=ax3, cmap='hot')
    ax3.set_title("热流密度大小")

    ax4 = plt.subplot(2, 2, 4)
    visualizer.plot_temperature_profile(y=1.0, ax=ax4)
    ax4.set_title("沿 Y=1.0m 的温度分布")

    plt.suptitle("示例3: 内部点热源 - 对流边界", fontsize=14)
    plt.tight_layout()

    return config, mesh, solver, visualizer


def example_4_transient():
    """
    示例4: 瞬态热传导
    """
    print("\n" + "=" * 60)
    print("示例4: 瞬态热传导")
    print("=" * 60)

    config = ThermalConfig()
    config.set_geometry(width=1.0, height=1.0, nx=25, ny=25)
    config.set_material(thermal_conductivity=50.0, density=7850.0,
                        specific_heat=500.0, name="steel")
    config.clear_boundary_conditions()
    config.add_dirichlet_bc(temperature=100.0, edge="left")
    config.add_dirichlet_bc(temperature=25.0, edge="right")
    config.add_dirichlet_bc(temperature=25.0, edge="top")
    config.add_dirichlet_bc(temperature=25.0, edge="bottom")

    config.set_solver_params(
        steady_state=False,
        time_step=0.05,
        total_time=2.0,
        initial_temp=25.0
    )

    print(config.info())

    mesh = generate_mesh(config, mesh_type="structured")
    print(mesh.info())

    solver = FEMSolver(config, mesh)
    time_points, temperature_history = solver.solve_transient()
    solver.temperature = temperature_history[-1]
    solver.compute_heat_flux()
    print(f"瞬态求解完成，共 {len(time_points)} 个时间步")
    print(f"最终温度范围: {np.min(temperature_history[-1]):.2f} ~ {np.max(temperature_history[-1]):.2f} °C")

    visualizer = create_visualization(config, mesh, solver)

    fig = plt.figure(figsize=(16, 12))

    ax1 = plt.subplot(2, 3, 1)
    visualizer.plot_temperature(temperature_history[0], ax=ax1)
    ax1.set_title(f"t = {time_points[0]:.2f}s")

    ax2 = plt.subplot(2, 3, 2)
    mid_idx = len(time_points) // 4
    visualizer.plot_temperature(temperature_history[mid_idx], ax=ax2)
    ax2.set_title(f"t = {time_points[mid_idx]:.2f}s")

    ax3 = plt.subplot(2, 3, 3)
    mid_idx2 = len(time_points) // 2
    visualizer.plot_temperature(temperature_history[mid_idx2], ax=ax3)
    ax3.set_title(f"t = {time_points[mid_idx2]:.2f}s")

    ax4 = plt.subplot(2, 3, 4)
    visualizer.plot_temperature(temperature_history[-1], ax=ax4)
    ax4.set_title(f"t = {time_points[-1]:.2f}s (稳态)")

    ax5 = plt.subplot(2, 3, 5)
    visualizer.plot_temperature_history(
        time_points, temperature_history,
        probe_points=[(0.5, 0.5), (0.1, 0.5), (0.9, 0.5)],
        ax=ax5
    )

    ax6 = plt.subplot(2, 3, 6)
    visualizer.plot_heat_flux_magnitude(ax=ax6)
    ax6.set_title("最终热流密度")

    plt.suptitle("示例4: 瞬态热传导过程", fontsize=14)
    plt.tight_layout()

    return config, mesh, solver, visualizer, time_points, temperature_history


def example_5_mesh_convergence():
    """
    示例5: 网格收敛性验证
    """
    print("\n" + "=" * 60)
    print("示例5: 网格收敛性验证")
    print("=" * 60)

    config = ThermalConfig()
    config.set_geometry(width=1.0, height=1.0, nx=10, ny=10)
    config.set_material(thermal_conductivity=100.0, density=7850.0,
                        specific_heat=500.0, name="steel")
    config.clear_boundary_conditions()
    config.add_dirichlet_bc(temperature=100.0, edge="left")
    config.add_dirichlet_bc(temperature=25.0, edge="right")

    mesh_sizes = [5, 10, 20, 30, 50]
    center_temps = []
    num_nodes_list = []

    for nx in mesh_sizes:
        config.set_geometry(width=1.0, height=1.0, nx=nx, ny=nx)
        mesh = generate_mesh(config, mesh_type="structured")
        solver = solve_thermal_problem(config, mesh)

        center_idx = np.argmin(
            np.abs(mesh.nodes[:, 0] - 0.5) + np.abs(mesh.nodes[:, 1] - 0.5)
        )
        center_temp = solver.temperature[center_idx]
        center_temps.append(center_temp)
        num_nodes_list.append(mesh.num_nodes)

        print(f"网格 {nx}x{nx}: 节点数={mesh.num_nodes}, "
              f"中心温度={center_temp:.4f}°C")

    analytical_center_temp = 62.5
    print(f"\n解析解中心温度: {analytical_center_temp}°C")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(num_nodes_list, center_temps, 'bo-', linewidth=2, markersize=8,
             label='有限元解')
    ax1.axhline(y=analytical_center_temp, color='r', linestyle='--',
                linewidth=2, label='解析解')
    ax1.set_xlabel('节点数')
    ax1.set_ylabel('中心温度 (°C)')
    ax1.set_title('网格收敛性验证')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')

    errors = [abs(t - analytical_center_temp) / analytical_center_temp * 100
              for t in center_temps]
    ax2.plot(num_nodes_list, errors, 'rs-', linewidth=2, markersize=8)
    ax2.set_xlabel('节点数')
    ax2.set_ylabel('相对误差 (%)')
    ax2.set_title('收敛误差')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    plt.suptitle("示例5: 网格收敛性分析", fontsize=14)
    plt.tight_layout()

    return config, mesh_sizes, center_temps


def main():
    """主函数"""
    print("╔" + "=" * 58 + "╗")
    print("║     二维材料热传导有限元数值模拟系统                      ║")
    print("║     2D Thermal Conduction FEM Simulation System       ║")
    print("╚" + "=" * 58 + "╝")

    try:
        example_1_steady_state_dirichlet()
        example_2_neumann_boundary()
        example_3_internal_heat_source()
        example_4_transient()
        example_5_mesh_convergence()

        print("\n" + "=" * 60)
        print("所有示例计算完成！")
        print("显示图形窗口中...")
        print("=" * 60)

        plt.show()

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
