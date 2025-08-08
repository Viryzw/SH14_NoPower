import os
from Utils.Manager import Manager
from Utils.Refresher import TargetRefresher
from Controller.control import UAVController
from Controller.control import USVController
from Utils.Scorer import score1, score2

# --- 系统初始化 --- #
manager = Manager()
refresher = TargetRefresher()
uav_controller = UAVController() # UAV 控制器
usv_controller = USVController() # USV 控制器
init_step = 1 # 自定义：Target 首次出现时刻
# mode = "powerless"
mode = "powerful"


# --- UAV USV 初始化 --- #
uavs = [
    ["1", [0, 5630], 0],
    ["2", [0, 3630], 0]
]
usvs = [
    ["1", [0, 6130], 0],
    ["2", [0, 5130], 0],
    ["3", [0, 4130], 0],
    ["4", [0, 3130], 0]
]

manager.init_objects(uavs, usvs)

# --- UAV-USV 控制器连接 --- #
uav_controller.set_usv_states_ref(usv_controller.usv_states)  # 🔥 关键：建立USV状态引用

# --- 仿真主循环 --- #
max_step = 14400

for step in range(max_step):
    
    # --- Target 初始化 --- #
    if step == init_step:
        manager.add_targets(['1', '2'], t=step)
    
    # --- 控制矩阵 example --- #
    # controls = [
    #     ["uav", "1", 0, 0], # 类型、编号、线速度、角速度
    #     ["uav", "2", 0, 0],
    #     ["usv", "1", 0, 0],
    #     ["usv", "2", 0, 0],
    #     ["usv", "3", 0, 0],
    #     ["usv", "4", 0, 0],
    # ]
    
    # --- 控制矩阵 --- #
    controls = [] # 总控制矩阵
    uav_controls = uav_controller.update(manager)  # UAV控制矩阵
    usv_controls = usv_controller.update(manager)  # USV控制矩阵
    
    for ctrl in uav_controls:
        if ctrl[0] == 'usv':
            usv_id = ctrl[1]
            if usv_id in usv_controls:
                # 拷贝原始控制命令并更新坐标信息
                updated_ctrl = ['usv', usv_id, usv_controls[usv_id][0], usv_controls[usv_id][1]]
                controls.append(updated_ctrl)
            else:
                controls.append(ctrl)
        else:
            controls.append(ctrl)
    
    for tid in list(manager.targets.keys()):
        controls.append(["target", tid, 0, 0])
        
    # --- 更新状态 --- #
    manager.update(controls, t=step, mode=mode)
    
    # --- 刷新目标 --- #
    new_target_list = refresher.refresh(step, manager)
    manager.add_targets(new_target_list, t=step)

    # --- 打印系统状态 --- #
    print(f"\n=== Step {step} 状态 ===")
    
    for uid in ['1', '2']:
        print(f"UAV {uid} 探测到目标: {manager.get_detected('uav', uid)}")
        
    for uid in ['1', '2', '3', '4']:
        print(f"USV {uid} 探测到目标: {manager.get_detected('usv', uid)}")
        print(f"USV {uid} 捕获的目标: {manager.get_captured('usv', uid)}")
        
    print(f"实时总探测目标列表: {manager.get_detected_all_timely()}")
    print(f"累积总探测目标列表: {manager.get_detected_all()}")
    print(f"累积总捕获目标列表: {manager.get_captured_all()}")
    
    # --- 得分情况 --- #
    P = len(manager.time1) / (refresher.current_id - 1)
    S1 = score1(manager.time1)
    S2 = score2(manager.time2)
    print(f"P = {P}, S1 = {S1}, S2 = {S2}, Total = {P * (S1 + S2)}")

    if refresher.current_id == 9 and manager.targets == {}:
        os._exit(1)