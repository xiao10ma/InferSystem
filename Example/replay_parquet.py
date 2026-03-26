"""从 parquet 文件读取 action 数据并在 Flexiv 机械臂上 replay。

数据格式 (来自 DataCollectionSystem/lerobot):
  action: List[float], 长度 8 = [joint_1, ..., joint_7, gripper] (rad)
    - joint_1..7: 关节角度 (rad)
    - gripper: 归一化宽度 [0, 1]，0=全闭, 1=全开
  observation.state: List[float], 长度 22 = [q(7), dq(7), tau(7), gripper]

控制方式:
  关节: NRT_JOINT_POSITION 模式下的 SendJointPosition
  夹爪: Gripper.Move 连续宽度控制

用法:
    python replay_parquet.py <robot_sn> <parquet_path> [--fps 30] [--speed 1.0]
    python replay_parquet.py Rizon4-063609 /path/to/file-000.parquet
    python replay_parquet.py Rizon4-063609 /path/to/file-000.parquet --fps 30 --speed 0.5
    python replay_parquet.py Rizon4-063609 /path/to/file-000.parquet --dry-run  # 仅打印，不控制
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_actions(parquet_path: str) -> tuple[list[list[float]], list[list[float]]]:
    """读取 parquet 文件，返回 (actions, observations)。"""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("需要 pyarrow: pip install pyarrow")

    pf = pq.ParquetFile(parquet_path)
    table = pf.read()

    actions = table["action"].to_pylist()
    observations = table["observation.state"].to_pylist()

    print(f"加载完成: {len(actions)} 帧, action 维度={len(actions[0])}")
    print(f"  首帧 action (deg): {[round(np.degrees(v), 1) for v in actions[0][:7]]} + gripper={actions[0][7]:.2f}")
    print(f"  末帧 action (deg): {[round(np.degrees(v), 1) for v in actions[-1][:7]]} + gripper={actions[-1][7]:.2f}")

    # 夹爪统计
    grip_vals = [a[7] for a in actions]
    print(f"  夹爪范围: [{min(grip_vals):.3f}, {max(grip_vals):.3f}]")

    return actions, observations


def replay(
    robot_sn: str,
    parquet_path: str,
    fps: float = 30.0,
    speed: float = 1.0,
    max_vel: list[float] | None = None,
    max_acc: list[float] | None = None,
    gripper_name: str = "Flexiv-GN01",
    gripper_force: float = 15.0,
    gripper_speed: float = 0.2,
    dry_run: bool = False,
) -> None:
    """在 Flexiv 上 replay 动作序列。"""

    actions, observations = load_actions(parquet_path)
    n_frames = len(actions)
    dt = 1.0 / (fps * speed)

    if dry_run:
        print(f"\n[DRY RUN] 共 {n_frames} 帧, fps={fps}, speed={speed}x, dt={dt:.4f}s")
        print("前 5 帧 action (rad):")
        for i, act in enumerate(actions[:5]):
            joints = act[:7]
            grip = act[7]
            print(f"  [{i:4d}] joints={[round(v, 4) for v in joints]} gripper={grip:.3f}")
        print(f"...共 {n_frames} 帧")
        return

    import flexivrdk

    # ── 1. 连接机器人 ──
    print(f"[1] 连接机器人 {robot_sn} ...")
    robot = flexivrdk.Robot(robot_sn)

    if robot.fault():
        print("[!] 检测到故障，正在清除 ...")
        robot.ClearFault()
        time.sleep(2.0)
        if robot.fault():
            raise RuntimeError("无法清除机器人故障")

    robot.Enable()

    timeout = 30
    t0 = time.time()
    while not robot.operational():
        if time.time() - t0 > timeout:
            raise RuntimeError("机器人未能在超时时间内变为 operational")
        time.sleep(0.1)
    print("[1] 机器人已 operational")

    # ── 2. 初始化夹爪 ──
    print(f"[2] 初始化夹爪 '{gripper_name}' ...")
    gripper_ok = False
    gripper_max_width = 0.09  # 默认值
    try:
        gripper = flexivrdk.Gripper(robot)
        gripper.Enable(gripper_name)
        tool = flexivrdk.Tool(robot)
        tool.Switch(gripper_name)
        gripper.Init()
        time.sleep(3.0)
        gripper_params = gripper.params()
        gripper_max_width = gripper_params.max_width
        gripper_ok = True
        print(f"[2] 夹爪就绪 (max_width={gripper_max_width:.4f}m, "
              f"max_vel={gripper_params.max_vel:.4f}m/s, max_force={gripper_params.max_force:.1f}N)")
    except Exception as e:
        print(f"[2] 夹爪初始化失败 ({e})，跳过夹爪控制")

    # 设置首帧夹爪位置
    first_gripper = actions[0][7]
    if gripper_ok:
        target_width = first_gripper * gripper_max_width
        gripper.Move(target_width, gripper_speed, gripper_force)
        print(f"[2] 夹爪移至首帧位置: {first_gripper:.3f} ({target_width*1000:.1f}mm)")
        time.sleep(1.0)

    # ── 3. MoveJ 到第一帧位置 ──
    first_joints_rad = actions[0][:7]
    first_joints_deg = [float(np.degrees(v)) for v in first_joints_rad]
    print(f"[3] MoveJ 到起始位置 (deg): {[round(v, 1) for v in first_joints_deg]}")

    robot.SwitchMode(flexivrdk.Mode.NRT_PRIMITIVE_EXECUTION)
    start_jpos = flexivrdk.JPos(first_joints_deg)
    robot.ExecutePrimitive("MoveJ", {"target": start_jpos, "jntVelScale": 30})

    # 等待到位
    while not robot.primitive_states().get("reachedTarget", 0):
        time.sleep(0.1)
    print("[3] 已到达起始位置")

    # ── 4. 切换到 NRT_JOINT_POSITION 模式 ──
    print(f"[4] 切换到 NRT_JOINT_POSITION 模式, fps={fps}, speed={speed}x")
    robot.SwitchMode(flexivrdk.Mode.NRT_JOINT_POSITION)
    time.sleep(0.5)

    dof = 7
    if max_vel is None:
        max_vel = [2.0] * dof
    if max_acc is None:
        max_acc = [3.0] * dof
    zero_vel = [0.0] * dof

    # ── 5. Replay 主循环 ──
    print(f"[5] 开始 replay: {n_frames} 帧, 预计 {n_frames * dt:.1f}s")
    print("    按 Ctrl+C 可安全停止")

    frame_count = 0

    try:
        for i, act in enumerate(actions):
            t_loop = time.perf_counter()

            joint_targets = list(act[:7])
            gripper_val = act[7]

            # 发送关节位置
            try:
                robot.SendJointPosition(joint_targets, zero_vel, max_vel, max_acc)
            except Exception as e:
                print(f"\n[!] 帧 {i}: SendJointPosition 失败: {e}")
                if robot.fault():
                    print("[!] 检测到故障，尝试清除...")
                    robot.ClearFault()
                    time.sleep(1.0)
                    if robot.fault():
                        print("[!] 无法清除故障，停止 replay")
                        break
                    robot.SwitchMode(flexivrdk.Mode.NRT_JOINT_POSITION)
                    time.sleep(0.5)
                continue

            # 夹爪连续宽度控制 (每帧都发，Gripper.Move 是非阻塞的)
            if gripper_ok:
                try:
                    target_width = gripper_val * gripper_max_width
                    gripper.Move(target_width, gripper_speed, gripper_force)
                except Exception as e:
                    if frame_count % 100 == 0:
                        print(f"  [{i:4d}] 夹爪控制失败: {e}")

            frame_count += 1

            # 进度显示 (~1Hz)
            if i % int(fps) == 0:
                cur_q = list(robot.states().q)
                err = [abs(joint_targets[j] - cur_q[j]) for j in range(dof)]
                max_err_deg = max(err) * 180.0 / np.pi
                grip_info = ""
                if gripper_ok:
                    grip_info = f" gripper_cmd={gripper_val:.3f} actual={gripper.states().width/gripper_max_width:.3f}"
                print(
                    f"  [{i:4d}/{n_frames}] "
                    f"t={i/fps:.1f}s "
                    f"max_err={max_err_deg:.2f}deg"
                    f"{grip_info}",
                    flush=True,
                )

            # 精确定时
            elapsed = time.perf_counter() - t_loop
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print(f"\n[!] 用户中断 (已执行 {frame_count}/{n_frames} 帧)")

    # ── 6. 停止 ──
    print("[6] Replay 完成，停止机器人")
    try:
        robot.Stop()
    except Exception:
        pass

    final_q = list(robot.states().q)
    print(f"    最终关节 (deg): {[round(np.degrees(v), 1) for v in final_q]}")
    if gripper_ok:
        print(f"    最终夹爪宽度: {gripper.states().width:.4f}m")
    print(f"    实际执行帧数: {frame_count}/{n_frames}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 parquet 文件 replay 录制的动作到 Flexiv 机械臂"
    )
    parser.add_argument("robot_sn", help="机器人序列号，例如 Rizon4-063609")
    parser.add_argument("parquet_path", help="parquet 数据文件路径")
    parser.add_argument("--fps", type=float, default=30.0, help="录制时的帧率 (default: 30)")
    parser.add_argument("--speed", type=float, default=1.0, help="回放速度倍率 (default: 1.0)")
    parser.add_argument("--gripper-name", type=str, default="Flexiv-GN01", help="夹爪名称")
    parser.add_argument("--gripper-force", type=float, default=15.0, help="夹爪力 (N)")
    parser.add_argument("--gripper-speed", type=float, default=0.2, help="夹爪速度 (m/s)")
    parser.add_argument("--dry-run", action="store_true", help="仅读取打印数据，不连接机器人")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    replay(
        robot_sn=args.robot_sn,
        parquet_path=args.parquet_path,
        fps=args.fps,
        speed=args.speed,
        gripper_name=args.gripper_name,
        gripper_force=args.gripper_force,
        gripper_speed=args.gripper_speed,
        dry_run=args.dry_run,
    )
