import xarray as xr
import numpy as np
import random
import csv
import math

# 读取 NetCDF 文件并筛选有效点
filename = "20160505000000-GLOBCURRENT-L4-CUReul_15m-ALT_MED_SUM-v03.0-fv01.0.nc"
ds = xr.open_dataset(filename)
u = ds['eastward_eulerian_current_velocity'][0, :, :]
v = ds['northward_eulerian_current_velocity'][0, :, :]
mask = (~np.isnan(u)) & (~np.isnan(v))

lat_vals = ds['lat'].values
lon_vals = ds['lon'].values
lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals, indexing='ij')

valid_lats = lat_grid[mask]
valid_lons = lon_grid[mask]

# 限制在边界内
lat_min, lat_max = 30, 45
lon_min, lon_max = 15, 36
inside_mask = (valid_lats >= lat_min) & (valid_lats <= lat_max) & \
              (valid_lons >= lon_min) & (valid_lons <= lon_max)

valid_lats = valid_lats[inside_mask]
valid_lons = valid_lons[inside_mask]

max_points = 10000
if len(valid_lons) > max_points:
    idx = np.random.choice(len(valid_lons), size=max_points, replace=False)
    valid_lons = valid_lons[idx]
    valid_lats = valid_lats[idx]

# 构建FieldSet
from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4
from datetime import timedelta

filenames = {'U': filename, 'V': filename}
variables = {'U': 'eastward_eulerian_current_velocity', 'V': 'northward_eulerian_current_velocity'}
dimensions = {'lon': 'lon', 'lat': 'lat', 'time': 'time'}
fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, allow_time_extrapolation=True)

# 粒子追踪
pset = ParticleSet(fieldset=fieldset, pclass=JITParticle, lon=valid_lons, lat=valid_lats)
pset.execute(AdvectionRK4, runtime=timedelta(days=100), dt=timedelta(minutes=30),
             output_file=pset.ParticleFile(name="trajectory_part.zarr", outputdt=timedelta(hours=1)))

# 从粒子追踪结果读取
ds_traj = xr.open_zarr("trajectory_part.zarr")
lon = ds_traj['lon'].values
lat = ds_traj['lat'].values

num_particles, num_times = lon.shape

# 随机选区域
region_half_size = 0.05
valid_points = [(lon[i, t], lat[i, t]) for i in range(num_particles) for t in range(num_times)
                if not np.isnan(lon[i, t]) and not np.isnan(lat[i, t])]
valid_points = np.array(valid_points)

min_points_in_region = 800
max_points_in_region = 1500
min_trajectories_in_region = 60

def points_in_region(center_lon, center_lat, lon_arr, lat_arr, half_size):
    return ((lon_arr >= center_lon - half_size) & (lon_arr <= center_lon + half_size) &
            (lat_arr >= center_lat - half_size) & (lat_arr <= center_lat + half_size))

for attempt in range(1000):
    center_idx = random.randint(0, len(valid_points) - 1)
    center_lon, center_lat = valid_points[center_idx]

    trajectory_count = 0
    point_count = 0
    for i in range(num_particles):
        mask = points_in_region(center_lon, center_lat, lon[i, :], lat[i, :], region_half_size)
        if np.sum(mask) > 0:
            trajectory_count += 1
            point_count += np.sum(mask)

    if (min_points_in_region <= point_count <= max_points_in_region) and trajectory_count >= min_trajectories_in_region:
        break

lon_min, lon_max = center_lon - region_half_size, center_lon + region_half_size
lat_min, lat_max = center_lat - region_half_size, center_lat + region_half_size

def map_to_rect(lon_val, lat_val):
    target_min = -1000
    target_max = 10260
    scale = target_max - target_min
    x_mapped = target_min + scale * (lon_val - lon_min) / (lon_max - lon_min)
    y_mapped = target_min + scale * (lat_val - lat_min) / (lat_max - lat_min)
    return x_mapped, y_mapped

# 边界处理函数
BOUNDARY_MIN = 0.0
BOUNDARY_MAX = 9260.0

def compute_boundary_intersection(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x1 - x2
    dy = y1 - y2

    if abs(dx) < 1e-8 and abs(dy) < 1e-8:
        return None

    intersections = []
    if dx != 0:
        y_at_left = y2 + (BOUNDARY_MIN - x2) * dy / dx
        if BOUNDARY_MIN <= y_at_left <= BOUNDARY_MAX:
            intersections.append((BOUNDARY_MIN, y_at_left))

        y_at_right = y2 + (BOUNDARY_MAX - x2) * dy / dx
        if BOUNDARY_MIN <= y_at_right <= BOUNDARY_MAX:
            intersections.append((BOUNDARY_MAX, y_at_right))

    if dy != 0:
        x_at_bottom = x2 + (BOUNDARY_MIN - y2) * dx / dy
        if BOUNDARY_MIN <= x_at_bottom <= BOUNDARY_MAX:
            intersections.append((x_at_bottom, BOUNDARY_MIN))

        x_at_top = x2 + (BOUNDARY_MAX - y2) * dx / dy
        if BOUNDARY_MIN <= x_at_top <= BOUNDARY_MAX:
            intersections.append((x_at_top, BOUNDARY_MAX))

    if not intersections:
        return None

    closest_point = min(intersections, key=lambda pt: np.hypot(pt[0] - x2, pt[1] - y2))
    return closest_point

# 主处理流程
processed_trajectories = []

for i in range(num_particles):
    lon_track = lon[i, :]
    lat_track = lat[i, :]

    in_region = points_in_region(center_lon, center_lat, lon_track, lat_track, region_half_size)

    if np.sum(in_region) == 0:
        continue

    x_points, y_points = [], []
    for x_orig, y_orig in zip(lon_track[in_region], lat_track[in_region]):
        if not np.isnan(x_orig) and not np.isnan(y_orig):
            x_mapped, y_mapped = map_to_rect(x_orig, y_orig)
            x_points.append(x_mapped)
            y_points.append(y_mapped)

    N = len(x_points)
    if N < 2:
        continue

    all_in_region = np.all((BOUNDARY_MIN <= np.array(x_points)) & (np.array(x_points) <= BOUNDARY_MAX) &
                           (BOUNDARY_MIN <= np.array(y_points)) & (np.array(y_points) <= BOUNDARY_MAX))
    if all_in_region:
        continue

    first_enter_idx = None
    for idx in range(1, N):
        xi, yi = x_points[idx], y_points[idx]
        if (BOUNDARY_MIN <= xi <= BOUNDARY_MAX) and (BOUNDARY_MIN <= yi <= BOUNDARY_MAX):
            first_enter_idx = idx
            break

    if first_enter_idx is None:
        continue

    p1 = (x_points[first_enter_idx - 1], y_points[first_enter_idx - 1])
    p2 = (x_points[first_enter_idx], y_points[first_enter_idx])

    new_start = compute_boundary_intersection(p2, p1)
    if new_start is None:
        continue

    clipped_traj = [new_start, p2]
    for j in range(first_enter_idx + 1, N):
        clipped_traj.append((x_points[j], y_points[j]))

    if len(clipped_traj) >= 2:
        processed_trajectories.append(clipped_traj)

# 保存最终结果
with open('Vehicle/region_trajectories_start_modified.csv', 'w', newline='') as f_out:
    writer = csv.writer(f_out)
    for traj in processed_trajectories:
        row = []
        for x, y in traj:
            row.extend([x, y])
        writer.writerow(row)

print(f"✅ 已生成 {len(processed_trajectories)} 条轨迹，保存为 region_trajectories_start_modified.csv")
