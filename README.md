#### 一、启动步骤

1、运行 `init.py`，随机初始化洋流轨迹

```bash
python3 init.py
```

2、运行 `main.py`，启动仿真环境

```bash
python3 main.py
```

---

#### 二、项目结构

```bash
SH14
├── init.py		# 初始化程序
├── main.py		# 主程序
├── README.md
│
├── Controller	# 控制库 TODO
│   └── control.py		    # UAV、USV 控制模块 TODO
│
├── Module		# 基本物理库
│   ├── Capture_Usv.py		# USV 捕获模块
│   ├── Collision_Uav.py	# UAV 碰撞模块
│   ├── Collision_Usv.py	# USV 碰撞模块
│   ├── Sensor_Uav.py		# UAV 探测模块
│   └── Sensor_Usv.py		# UAV 探测模块
│
├── Utils
│   ├── Manager.py			# 总控制模块
│   ├── Points.py			# Target 轨迹生成模块
│   ├── Refresher.py		# Target 刷新模块
│   ├── Scorer.py			# 计分模块
│   └── Visualizer.py		# 可视化模块
│
└── Vehicle		
    ├── BaseVehicle.py		# UAV/USV 父类
    ├── Target.py			# Target 控制模块 
    ├── Uav.py				# UAV 初始化模块
    └── Usv.py				# USV 初始化模块
```


---

#### 三、函数与变量说明

##### main.py

- 指定 `UAV` 探测列表 `manager.get_detected('uav', uid) -> List[Target]`
- 指定 `USV` 探测列表 `manager.get_detected('usv', uid) -> List[Target]`
- 指定 `USV` 捕获列表 `manager.get_captured('usv', uid) -> List[Target]`
- 总探测列表（去重）`manager.get_detected_all() -> List[Target.id]`
- 总捕获列表（去重）`manager.get_captured_all() -> List[Target.id]`
- `USV` 总探测列表（去重）`manager.get_detected_usv() -> List[Target.id]`
- 指定载具位置与朝向 `manager.get_state(typ, id) -> [pos, heading]`
- 目标出现步次 `init_step`
- 目标动力模型 `mode`：`'powerless'`-无动力，`'powerful'`-有动力

##### Utils/Refresher.py

- 目标刷新步长 `refresh_interval`
- 目标下次刷新的步次 `self.next_refresh_step`，注意本参数值应当与 `refresh_interval` 始终相等
- 场上无目标时就刷新，保证总共出现八个，在任务结束后直接终止程序
---

#### 四、开发说明

- 针对 `UAV` 和 `USV` 的控制文件原则上应写成类，全部放在 `Controller` 文件夹中，并在 `main.py` 中实例化控制类，每个 `step` 输出一个形如 `controls` 的控制矩阵供调用

```python
controls = [
        ["uav", "1", 0, 0], # 类型、编号、线速度、角速度
        ["uav", "2", 0, 0],
        ["usv", "1", 0, 0],
        ["usv", "2", 0, 0],
        ["usv", "3", 0, 0],
        ["usv", "4", 0, 0],
    ]
```
- 当前仿真为 `10` 倍速，每 `step` 对应现实时间约 `0.05s`，对应仿真时间 `0.5s`，仿真界面左上角所示为仿真时间
- 当前仿真目标刷新起始时刻（步次）为 `t1=600s (init_step=1200)`，刷新时间（步长）为 `t2=600s (refresh_interval=1200)`
- 仿真 `n` 倍速修改方法
    - 修改 `Vechicle/Uav.py` 中 `dt=0.05 * n`
    - 修改 `Vechicle/Usv.py` 中 `dt=0.05 * n`
    - 修改 `Vechicle/Target.py` 中 `dt=0.05 * n`
    - 修改 `Utils/Refresher.py` 中 `refresh_interval` 和 `self.next_refresh_step` 为 `t2 / dt`
- 可选可视化：修改 `Manager.py`
    - `UAV` 轨迹：`init_uavs()` 中 `self.trajectory_visible.append(True)`
    - `USV` 轨迹：`init_usvs()` 中 `self.trajectory_visible.append(True)`
    - `UAV` 历史探测区域：`__init__()` 中 `self.enable_swept_area = True`  `self.swept_area_start_time = 合适值`
- 算法鲁棒性验证：
    - 运行 `init.py` 以修改目标刷新位置（四条边随机）；
    - 修改 `main.py` 中的 `mode` 参数，对不同动力模型的目标做测试