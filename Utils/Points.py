import csv
import random

def read_random_line_as_coordinates(csv_file_path):
    # 读取所有行
    with open(csv_file_path, newline='') as csvfile:
        reader = list(csv.reader(csvfile))
        if not reader:
            return []  # 文件为空
        # 随机选择一行
        row = random.choice(reader)
        # 转换为浮点数
        row_values = [float(value) for value in row]
        # 每两个值组成一个 (x, y) 元组
        coordinates = [
            [row_values[i], row_values[i + 1]]
            for i in range(0, len(row_values), 2)
        ]
        return coordinates
