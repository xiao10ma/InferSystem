# Example

硬件示例脚本，按机器人平台组织。

## 目录

```
Example/
└── flexiv/                         # Flexiv Rizon 系列机械臂
    ├── infer_client.py             # 配置驱动的推理控制循环 (核心)
    ├── go_home.py                  # 回 Home 位置
    ├── gripper.py                  # 夹爪开合控制
    ├── probe.py                    # 连接探针 — 轮询状态
    ├── replay_parquet.py           # 从 parquet 文件回放动作
    └── realsense_visualize.py      # RealSense 相机可视化
```

## 快速开始

```bash
# 推理控制循环 (核心用法)
python Example/flexiv/infer_client.py Config/rizon4_example.yaml
python Example/flexiv/infer_client.py Config/rizon4_example.yaml --dry-run

# 基础操作
python Example/flexiv/probe.py Rizon4-063609 --enable
python Example/flexiv/go_home.py Rizon4-063609
python Example/flexiv/gripper.py Rizon4-063609 open close

# 数据回放
python Example/flexiv/replay_parquet.py Rizon4-063609 data.parquet

# 相机调试
python Example/flexiv/realsense_visualize.py
```
