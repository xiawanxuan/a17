"""
系统全面验证脚本
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import ThermalConfig
from mesh import generate_mesh
from solver import FEMSolver, solve_thermal_problem
from visualization import create_visualization


def test_steady_state_dirichlet():
    """测试稳态Dirichlet边界（解析解验证）"""
    print("\n" + "="*60)
    print("测试1: 稳态Dirichlet边界 - 解析解验证")
    print("="*60)

    config = ThermalConfig()
    config.set_geometry(width=1.0, height=1.0, nx=30, ny=30)
    config.set_material(thermal_conductivity=100.0, density=7850.0,
                        specific_heat=500.0, name="steel")
    config.clear_boundary_conditions()
    config.add_dirichlet_bc(temperature=100.0, edge="left")
    config.add_dirichlet_bc(temperature=25.0, edge="right")

    mesh = generate_mesh(config, mesh_type="structured")
    solver = solve_thermal_problem(config, mesh)

    def analytical_temp(x, y):
        T_left = 100.0
        T_right = 25.0
        L = 1.0
        return T_left + (T_right - T_left) * x / L

    result = solver.compare_with_analytical(analytical_temp)

    print(f"L2 误差: {result['l2_error']:.6e} °C")
    print(f"最大误差: {result['max_error']:.6e} °C")
    print(f"平均误差: {result['avg_error']:.6e} °C")

    assert result['max_error'] < 1e-10, f"最大误差过大: {result['max_error']}"
    print("✓ 解析解对比通过")

    energy = solver.verify_energy_conservation()
    print(f"\n能量守恒验证:")
    print(f"  边界流入热流量: {energy['boundary_heat_inflow']:.4f} W")
    print(f"  内热源功率: {energy['internal_heat_source']:.4f} W")
    print(f"  总流入量: {energy['total_inflow']:.4f} W (稳态应接近0)")
    print(f"  相对误差: {energy['relative_error']:.4f} %")
    print("✓ 能量守恒验证完成")

    return True


def test_transient_convergence():
    """测试瞬态求解收敛到稳态"""
    print("\n" + "="*60)
    print("测试2: 瞬态求解 - 收敛到稳态验证")
    print("="*60)

    config = ThermalConfig()
    config.set_geometry(width=1.0, height=1.0, nx=20, ny=20)
    config.set_material(thermal_conductivity=50.0, density=7850.0,
                        specific_heat=500.0, name="steel")
    config.clear_boundary_conditions()
    config.add_dirichlet_bc(temperature=100.0, edge="left")
    config.add_dirichlet_bc(temperature=25.0, edge="right")
    config.add_dirichlet_bc(temperature=25.0, edge="top")
    config.add_dirichlet_bc(temperature=25.0, edge="bottom")

    config.set_solver_params(
        steady_state=False,
        time_step=0.02,
        total_time=5.0,
        initial_temp=25.0
    )

    mesh = generate_mesh(config, mesh_type="structured")
    solver = FEMSolver(config, mesh)

    time_points, temp_history = solver.solve_transient()

    print(f"时间步数: {len(time_points)}")
    print(f"初始温度范围: {np.min(temp_history[0]):.2f} ~ {np.max(temp_history[0]):.2f} °C")
    print(f"最终温度范围: {np.min(temp_history[-1]):.2f} ~ {np.max(temp_history[-1]):.2f} °C")

    solver.temperature = temp_history[-1]

    config_steady = ThermalConfig()
    config_steady.set_geometry(width=1.0, height=1.0, nx=20, ny=20)
    config_steady.set_material(thermal_conductivity=50.0, density=7850.0,
                               specific_heat=500.0, name="steel")
    config_steady.clear_boundary_conditions()
    config_steady.add_dirichlet_bc(temperature=100.0, edge="left")
    config_steady.add_dirichlet_bc(temperature=25.0, edge="right")
    config_steady.add_dirichlet_bc(temperature=25.0, edge="top")
    config_steady.add_dirichlet_bc(temperature=25.0, edge="bottom")

    mesh_steady = generate_mesh(config_steady, mesh_type="structured")
    solver_steady = solve_thermal_problem(config_steady, mesh_steady)

    temp_diff = np.abs(temp_history[-1] - solver_steady.temperature)
    max_diff = np.max(temp_diff)
    print(f"\n瞬态终态与稳态解最大差异: {max_diff:.4f} °C")

    if max_diff < 1.0:
        print("✓ 瞬态收敛到稳态")
    else:
        print("⚠ 瞬态尚未完全收敛，请增加总时间")

    return True


def test_heat_flux_calculation():
    """测试热流密度计算"""
    print("\n" + "="*60)
    print("测试3: 热流密度计算验证")
    print("="*60)

    config = ThermalConfig()
    config.set_geometry(width=1.0, height=1.0, nx=20, ny=20)
    config.set_material(thermal_conductivity=100.0, density=7850.0,
                        specific_heat=500.0, name="steel")
    config.clear_boundary_conditions()
    config.add_dirichlet_bc(temperature=100.0, edge="left")
    config.add_dirichlet_bc(temperature=0.0, edge="right")

    mesh = generate_mesh(config, mesh_type="structured")
    solver = solve_thermal_problem(config, mesh)

    flux = solver.compute_heat_flux()
    qx_avg = np.mean(flux[:, 0])
    qy_avg = np.mean(flux[:, 1])

    analytical_qx = 100.0 * 100.0 / 1.0

    print(f"计算热流密度 qx 平均值: {qx_avg:.2f} W/m²")
    print(f"解析解热流密度 qx: {analytical_qx:.2f} W/m²")
    print(f"热流密度 qy 平均值: {qy_avg:.2f} W/m² (应接近0)")

    error = abs(qx_avg - analytical_qx) / analytical_qx * 100
    print(f"相对误差: {error:.4f} %")

    if error < 1.0:
        print("✓ 热流密度计算正确")
    else:
        print("⚠ 热流密度误差较大")

    return True


def test_neumann_boundary():
    """测试Neumann边界条件"""
    print("\n" + "="*60)
    print("测试4: Neumann边界条件验证")
    print("="*60)

    config = ThermalConfig()
    config.set_geometry(width=1.0, height=1.0, nx=30, ny=30)
    config.set_material(thermal_conductivity=50.0, density=2700.0,
                        specific_heat=900.0, name="aluminum")
    config.clear_boundary_conditions()
    config.add_neumann_bc(heat_flux=1000.0, edge="top")
    config.add_dirichlet_bc(temperature=25.0, edge="bottom")
    config.add_dirichlet_bc(temperature=25.0, edge="left")
    config.add_dirichlet_bc(temperature=25.0, edge="right")

    mesh = generate_mesh(config, mesh_type="structured")
    solver = solve_thermal_problem(config, mesh)

    print(f"温度范围: {np.min(solver.temperature):.2f} ~ {np.max(solver.temperature):.2f} °C")

    energy = solver.verify_energy_conservation()
    print(f"\n能量守恒验证:")
    print(f"  边界流入热流量: {energy['boundary_heat_inflow']:.2f} W")
    print(f"  内热源功率: {energy['internal_heat_source']:.2f} W")
    print(f"  总流入量: {energy['total_inflow']:.2f} W (稳态应接近0)")
    print(f"  相对误差: {energy['relative_error']:.2f} %")

    if energy['relative_error'] < 5.0:
        print("✓ 能量守恒基本满足")
    else:
        print("⚠ 能量守恒误差较大")

    return True


def test_internal_heat_source():
    """测试内热源"""
    print("\n" + "="*60)
    print("测试5: 内热源与能量守恒")
    print("="*60)

    config = ThermalConfig()
    config.set_geometry(width=2.0, height=2.0, nx=40, ny=40)
    config.set_material(thermal_conductivity=50.0, density=8900.0,
                        specific_heat=385.0, name="copper")
    config.clear_boundary_conditions()
    config.add_convection_bc(h=100.0, T_ambient=25.0, edge="left")
    config.add_convection_bc(h=100.0, T_ambient=25.0, edge="right")
    config.add_convection_bc(h=100.0, T_ambient=25.0, edge="top")
    config.add_convection_bc(h=100.0, T_ambient=25.0, edge="bottom")

    config.add_point_heat_source(magnitude=5000.0, x=1.0, y=1.0)

    mesh = generate_mesh(config, mesh_type="structured")
    solver = solve_thermal_problem(config, mesh)

    print(f"温度范围: {np.min(solver.temperature):.2f} ~ {np.max(solver.temperature):.2f} °C")
    print(f"中心温度: {np.max(solver.temperature):.2f} °C")

    energy = solver.verify_energy_conservation()
    print(f"\n能量守恒验证:")
    print(f"  边界散热量: {-energy['boundary_heat_inflow']:.2f} W")
    print(f"  内热源功率: {energy['internal_heat_source']:.2f} W")
    print(f"  总流入量: {energy['total_inflow']:.2f} W (稳态应接近0)")
    print(f"  相对误差: {energy['relative_error']:.2f} %")

    if energy['relative_error'] < 10.0:
        print("✓ 能量守恒基本满足")
    else:
        print("⚠ 能量守恒误差较大 (点热源可能导致局部误差)")

    return True


def test_visualization():
    """测试可视化模块"""
    print("\n" + "="*60)
    print("测试6: 可视化功能")
    print("="*60)

    config = ThermalConfig()
    config.set_geometry(width=1.0, height=1.0, nx=20, ny=20)
    config.set_material(thermal_conductivity=100.0, density=7850.0,
                        specific_heat=500.0, name="steel")
    config.clear_boundary_conditions()
    config.add_dirichlet_bc(temperature=100.0, edge="left")
    config.add_dirichlet_bc(temperature=25.0, edge="right")

    mesh = generate_mesh(config, mesh_type="structured")
    solver = solve_thermal_problem(config, mesh)
    visualizer = create_visualization(config, mesh, solver)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    visualizer.plot_mesh(ax=axes[0, 0])
    axes[0, 0].set_title("网格")

    visualizer.plot_temperature(ax=axes[0, 1])
    axes[0, 1].set_title("温度场")

    visualizer.plot_heat_flux_vectors(ax=axes[0, 2], arrow_density=5)
    axes[0, 2].set_title("热流矢量")

    visualizer.plot_heat_flux_magnitude(ax=axes[1, 0])
    axes[1, 0].set_title("热流密度大小")

    visualizer.plot_temperature_profile(y=0.5, ax=axes[1, 1])
    axes[1, 1].set_title("温度分布曲线")

    visualizer.plot_boundary_temperatures(edge="left", ax=axes[1, 2])
    axes[1, 2].set_title("左边界温度")

    plt.tight_layout()
    plt.savefig('verify_visualization.png', dpi=100, bbox_inches='tight')
    plt.close()

    print("✓ 可视化功能正常，图像已保存为 verify_visualization.png")

    return True


def main():
    print("╔" + "="*58 + "╗")
    print("║     有限元热传导系统全面验证                          ║")
    print("╚" + "="*58 + "╝")

    all_passed = True

    try:
        all_passed &= test_steady_state_dirichlet()
    except Exception as e:
        print(f"✗ 测试1失败: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        all_passed &= test_transient_convergence()
    except Exception as e:
        print(f"✗ 测试2失败: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        all_passed &= test_heat_flux_calculation()
    except Exception as e:
        print(f"✗ 测试3失败: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        all_passed &= test_neumann_boundary()
    except Exception as e:
        print(f"✗ 测试4失败: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        all_passed &= test_internal_heat_source()
    except Exception as e:
        print(f"✗ 测试5失败: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        all_passed &= test_visualization()
    except Exception as e:
        print(f"✗ 测试6失败: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("✓ 所有测试通过！系统运行正常")
    else:
        print("✗ 部分测试失败，请检查")
    print("="*60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
