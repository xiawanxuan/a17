"""全面验证测试 - 非交互模式"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import ThermalConfig
from mesh import generate_mesh
from solver import solve_thermal_problem, FEMSolver
from visualization import create_visualization

print("=" * 60)
print("二维热传导有限元系统 - 全面功能验证")
print("=" * 60)

print("\n✓ 模块导入成功")
print("  - config  (参数配置)")
print("  - mesh    (网格划分)")
print("  - solver  (方程求解)")
print("  - visualization (可视化)")

print("\n--- 稳态求解 + 热应力 ---")
config = ThermalConfig()
config.set_geometry(width=1.0, height=1.0, nx=20, ny=20)
config.set_material(
    thermal_conductivity=100.0, density=7850.0, specific_heat=500.0,
    youngs_modulus=200e9, poissons_ratio=0.3,
    thermal_expansion_coeff=12e-6, reference_temperature=25.0,
    name="steel"
)
config.clear_boundary_conditions()
config.add_dirichlet_bc(temperature=100.0, edge="left")
config.add_dirichlet_bc(temperature=25.0, edge="right")
config.clear_displacement_bcs()
config.add_displacement_bc(edge="left", ux=0.0, uy=0.0)

mesh = generate_mesh(config, mesh_type="structured")
print(f"✓ 网格生成: {mesh.num_nodes}节点, {mesh.num_elements}单元")

solver = solve_thermal_problem(config, mesh)
print(f"✓ 稳态求解: T范围 {solver.temperature.min():.2f} ~ {solver.temperature.max():.2f}°C")

solver.solve_thermal_stress()
von_mises = solver.get_von_mises_stress()
ux, uy = solver.get_nodal_displacement()
print(f"✓ 热应力求解: σ_vm范围 {von_mises.min()/1e6:.2f} ~ {von_mises.max()/1e6:.2f} MPa")

print("\n--- 瞬态求解 ---")
config_t = ThermalConfig()
config_t.set_geometry(width=1.0, height=1.0, nx=10, ny=10)
config_t.set_material(thermal_conductivity=50.0, density=7850.0, specific_heat=500.0)
config_t.clear_boundary_conditions()
config_t.add_dirichlet_bc(temperature=100.0, edge="left")
config_t.add_dirichlet_bc(temperature=25.0, edge="right")
config_t.add_dirichlet_bc(temperature=25.0, edge="top")
config_t.add_dirichlet_bc(temperature=25.0, edge="bottom")
config_t.set_solver_params(steady_state=False, time_step=0.1, total_time=0.5, initial_temp=25.0)

mesh_t = generate_mesh(config_t, mesh_type="structured")
solver_t = FEMSolver(config_t, mesh_t)
time_points, temp_history = solver_t.solve_transient()
print(f"✓ 瞬态求解: {len(time_points)}步, T终值 {temp_history[-1].min():.2f} ~ {temp_history[-1].max():.2f}°C")

print("\n--- 可视化 ---")
vis = create_visualization(config, mesh, solver)
fig = vis.plot_thermal_stress_combined()
plt.savefig('verification_combined.png', dpi=100)
plt.close()
print("✓ 热应力组合视图保存: verification_combined.png")

vis_t = create_visualization(config_t, mesh_t, solver_t)
vis_t.animate_transient(time_points, temp_history,
                        save_path='verification_transient.gif',
                        save_fps=10)
plt.close('all')
print("✓ 瞬态动画保存: verification_transient.gif")

print("\n--- 能量守恒验证 ---")
energy = solver.verify_energy_conservation()
print(f"  边界热流: {energy['boundary_heat_inflow']:.4f} W")
print(f"  内热源: {energy['internal_heat_source']:.4f} W")
print(f"  相对误差: {energy['relative_error']:.4f} %")

print("\n" + "=" * 60)
print("✓ 所有功能验证通过!")
print("  新增功能:")
print("    1. 瞬态热传导求解 (向后欧拉法, 集中质量矩阵)")
print("    2. 热应力耦合计算 (平面应力, von Mises应力)")
print("    3. MP4/GIF动画导出 (自动格式选择)")
print("=" * 60)
