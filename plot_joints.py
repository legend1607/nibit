import pandas as pd
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False  # 修复负号显示问题
# CSV 文件路径
csv_path = "traj_001.csv"   # 换成你的实际路径

# 读取 CSV
df = pd.read_csv(csv_path)

# 横轴：index
x = df["index"]

# 关节列名（按你导出的名字来）
joint_cols = ["q0", "q1", "q2", "q3"]

# 曲线标签
labels = ["jointbody", "jointboom", "jointarm", "jointbucket"]

plt.figure(figsize=(8, 5))

for col, label in zip(joint_cols, labels):
    if col not in df.columns:
        print(f"Warning: column '{col}' not found in CSV, skip.")
        continue
    plt.plot(x, df[col], label=label)

plt.xlabel("序号")
plt.ylabel("关节角度 (rad)")
# plt.title("Joint Trajectory")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
