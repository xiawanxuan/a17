"""
快速测试脚本 - 验证有限元热传导模拟系统
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, '.')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import ThermalConfig
from mesh import generate_mesh
from solver import solve_thermal_problem
from visualization import create_visualization

print("=" * 50)
print("开始测试有限元热传导模拟系统")
print("=" * 50)

try:
    print("\n[1/5] 测试参数配置模块...")
    config = ThermalConfig()
    config.set_geometry(width=1.0, height=1.0, nx=10, ny=10)
    config.set_material(thermal_conductivity=100.0, density=7850.0,
                        specific_heat=500.0, name="steel")
    config.clear_boundary_conditions()
    config.add_dirichlet_bc(temperature=100.0, edge="left")
    config.add_dirichlet_bc(temperature=25.0, edge="right")
    print("  OK 参数配置模块正常")
    print(f"    热扩散率: {config.material.thermal_diffusivity():.6e} m²/s")

    print("\n[2/5] 测试网格划分模块...")
    mesh = generate_mesh(config, mesh_type="structured")
    print(f"  OK 网格生成成功")
    print(f"    节点数: {mesh.num_nodes}")
    print(f"    单元数: {mesh.num_elements}")
    
    areas = mesh.get_element_areas()
    print(f"    平均单元面积: {np.mean(areas):.6f} m²")
    
    quality = mesh.get_mesh_quality()
    print(f"    平均网格质量: {quality['avg_quality']:.4f}")

    print("\n[3/5] 测试方程求解模块...")
    solver = solve_thermal_problem(config, mesh)
    print(f"  OK 稳态求解成功")
    print(f"    温度范围: {np.min(solver.temperature):.2f} ~ {np.max(solver.temperature):.2f} °C")
    
    center_idx = np.argmin(np.abs(mesh.nodes[:, 0] - 0.5) + np.abs(mesh.nodes[:, 1] - 0.5))
    center_temp = solver.temperature[center_idx]
    print(f"    中心温度: {center_temp:.2f} °C")
    print(f"    理论值: 62.5 °C")
    print(f"    误差: {abs(center_temp - 62.5) / 62.5 * 100:.3f} %")

    print("\n[4/5] 测试热流密度计算...")
    flux = solver.compute_heat_flux()
    flux_mag = np.sqrt(flux[:, 0] ** 2 + flux[:, 1] ** 2)
    print(f"  OK 热流密度计算成功")
    print(f"    平均热流密度: {np.mean(flux_mag):.2f} W/m²")

    print("\n[5/5] 测试可视化模块...")
    visualizer = create_visualization(config, mesh, solver)
    
    fig, ax = plt.subplots()
    visualizer.plot_temperature(ax=ax)
    plt.savefig('test_temperature.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  OK 温度场云图生成成功: test_temperature.png")
    
    fig, ax = plt.subplots()
    visualizer.plot_heat_flux_vectors(ax=ax)
    plt.savefig('test_heat_flux.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  OK 热流密度矢量图生成成功: test_heat_flux.png")

    print("\n" + "=" * 50)
    print("所有测试通过！系统运行正常")
    print("=" * 50)

except Exception as e:
    print(f"\n测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
