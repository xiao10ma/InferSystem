"""从 parquet 或 LeRobot v3.0 数据集读取动作并在 ARX5 上 replay（默认双臂）。

支持 action 维度:
  - 14 维: [lq0..lq5, lgrip, rq0..rq5, rgrip]（双臂）
  - 7 维: [q0..q5, gripper]（单臂）
  - 12 / 6 维: 仅关节，夹爪保持当前

用法:
    # 直接指定 parquet 文件
    python Example/arx5/replay_parquet.py Config/arx5_bimanual_example.yaml data.parquet

    # LeRobot v3.0 数据集目录（自动读取 info.json 中的 fps）
    python Example/arx5/replay_parquet.py Config/arx5_bimanual_example.yaml ~/ARX/005
    python Example/arx5/replay_parquet.py Config/arx5_bimanual_example.yaml ~/ARX/005 --episode 0
    python Example/arx5/replay_parquet.py Config/arx5_bimanual_example.yaml ~/ARX/005 --episode 2 --fps 30 --speed 0.5

    # 同步播放数据集中的视频
    python Example/arx5/replay_parquet.py Config/arx5_bimanual_example.yaml ~/ARX/005 --episode 0 --play-video
    python Example/arx5/replay_parquet.py Config/arx5_bimanual_example.yaml ~/ARX/005 --episode 0 --play-video --cameras top left_wrist

    # 其他选项
    python Example/arx5/replay_parquet.py Config/arx5_bimanual_example.yaml data.parquet --no-home-before
    python Example/arx5/replay_parquet.py Config/arx5_bimanual_example.yaml data.parquet --home-after
    python Example/arx5/replay_parquet.py Config/arx5_bimanual_example.yaml data.parquet --column action --start 100 --end 500
    python Example/arx5/replay_parquet.py Config/arx5_bimanual_example.yaml data.parquet --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Core import Action, ActionSpace


from Robot import BaseRobot


def log(msg: str) -> None:
    print(msg, flush=True)


def _find_video_keys(info: dict) -> list[str]:
    """从 info.json 的 features 中提取所有 video 类型的 key。"""
    keys = []
    for k, v in info.get("features", {}).items():
        if v.get("dtype") == "video":
            keys.append(k)
    return sorted(keys)


def _resolve_video_path(
    dataset_dir: Path,
    info: dict,
    video_key: str,
    chunk_index: int,
    file_index: int,
) -> Path:
    """根据 info.json 的 video_path 模板解析实际视频文件路径。"""
    tmpl = info.get("video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    rel = tmpl.format(video_key=video_key, chunk_index=chunk_index, file_index=file_index)
    return dataset_dir / rel


class VideoPlayer:
    """同步播放多路视频，每次 show_frame(i) 显示第 i 帧。"""

    def __init__(self, video_paths: dict[str, Path]) -> None:
        import cv2
        self._cv2 = cv2
        self._caps: dict[str, cv2.VideoCapture] = {}
        for name, path in video_paths.items():
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                log(f"[WARN] 无法打开视频: {path}")
                continue
            self._caps[name] = cap
        if self._caps:
            names = ", ".join(self._caps.keys())
            log(f"视频播放器已打开: {names}")

    def show_frame(self, frame_idx: int) -> bool:
        """显示第 frame_idx 帧，返回 False 表示用户按 q 退出。"""
        cv2 = self._cv2
        for name, cap in self._caps.items():
            # 如果当前位置不是目标帧，seek 过去
            cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if cur != frame_idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                cv2.imshow(name, frame)
        key = cv2.waitKey(1) & 0xFF
        return key != ord("q")

    def release(self) -> None:
        for cap in self._caps.values():
            cap.release()
        if self._caps:
            self._cv2.destroyAllWindows()

    @property
    def active(self) -> bool:
        return len(self._caps) > 0


def _is_lerobot_dataset(path: Path) -> bool:
    """判断路径是否为 LeRobot v3.0 数据集目录。"""
    return path.is_dir() and (path / "meta" / "info.json").exists()


def _load_lerobot_info(dataset_dir: Path) -> dict:
    """读取 LeRobot 数据集的 info.json。"""
    with open(dataset_dir / "meta" / "info.json") as f:
        return json.load(f)


def _load_lerobot_episode_meta(dataset_dir: Path) -> list[dict]:
    """从 meta/episodes/ 读取所有 episode 元信息，按 episode_index 排序返回。"""
    import pyarrow.parquet as pq

    episodes_dir = dataset_dir / "meta" / "episodes"
    episodes = []
    for chunk_dir in sorted(episodes_dir.iterdir()):
        if not chunk_dir.is_dir():
            continue
        for pf in sorted(chunk_dir.glob("*.parquet")):
            table = pq.read_table(pf, columns=["episode_index", "length", "data/chunk_index", "data/file_index"])
            for i in range(table.num_rows):
                episodes.append({
                    "episode_index": table["episode_index"][i].as_py(),
                    "length": table["length"][i].as_py(),
                    "chunk_index": table["data/chunk_index"][i].as_py(),
                    "file_index": table["data/file_index"][i].as_py(),
                })
    episodes.sort(key=lambda e: e["episode_index"])
    return episodes


def load_actions_from_lerobot(
    dataset_dir: Path,
    *,
    episode: int | None = None,
    column: str = "action",
    start: int | None = None,
    end: int | None = None,
) -> list[list[float]]:
    """从 LeRobot v3.0 数据集目录加载动作序列。

    如果指定 episode，只加载该 episode 的帧；否则加载所有帧。
    """
    import pyarrow.parquet as pq

    episodes_meta = _load_lerobot_episode_meta(dataset_dir)
    if not episodes_meta:
        raise ValueError(f"数据集 {dataset_dir} 中无 episode 元信息")

    available_eps = [e["episode_index"] for e in episodes_meta]
    log(f"数据集共 {len(episodes_meta)} 个 episode: {available_eps}")

    if episode is not None:
        targets = [e for e in episodes_meta if e["episode_index"] == episode]
        if not targets:
            raise ValueError(f"episode {episode} 不存在，可用: {available_eps}")
    else:
        targets = episodes_meta

    # 按 (chunk_index, file_index) 分组，避免重复读同一个文件
    file_groups: dict[tuple[int, int], list[int]] = {}
    for ep in targets:
        key = (ep["chunk_index"], ep["file_index"])
        file_groups.setdefault(key, []).append(ep["episode_index"])

    all_actions: list[list[float]] = []
    for (chunk_idx, file_idx), ep_indices in sorted(file_groups.items()):
        parquet_path = dataset_dir / "data" / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {parquet_path}")
        table = pq.read_table(parquet_path, columns=[column, "episode_index", "frame_index"])

        for ep_idx in sorted(ep_indices):
            # 筛选该 episode 的行并按 frame_index 排序
            mask = [table["episode_index"][i].as_py() == ep_idx for i in range(table.num_rows)]
            indices = [i for i, m in enumerate(mask) if m]
            indices.sort(key=lambda i: table["frame_index"][i].as_py())
            for i in indices:
                all_actions.append(table[column][i].as_py())

    all_actions = all_actions[start:end]
    log(f"加载完成: {len(all_actions)} 帧, action_dim={len(all_actions[0]) if all_actions else '?'}")
    return all_actions


def load_actions(
    parquet_path: str,
    column: str = "action",
    start: int | None = None,
    end: int | None = None,
) -> list[list[float]]:
    """从单个 parquet 文件加载动作序列。

    Args:
        parquet_path: parquet 文件路径。
        column: 动作列名（默认 "action"）。
        start: 起始帧索引（含），None 表示从头。
        end: 结束帧索引（不含），None 表示到尾。
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("需要 pyarrow: pip install pyarrow") from exc

    table = pq.read_table(parquet_path, columns=[column])
    actions = table[column].to_pylist()
    if not actions:
        raise ValueError(f"parquet 列 '{column}' 为空")

    actions = actions[start:end]
    log(f"加载完成: {len(actions)} 帧, action_dim={len(actions[0])}")
    return actions


def replay(
    config_path: str,
    data_path: str,
    *,
    fps: float | None = None,
    speed: float = 1.0,
    column: str = "action",
    episode: int | None = None,
    start: int | None = None,
    end: int | None = None,
    home_before: bool = True,
    home_after: bool = False,
    dry_run: bool = False,
    play_video: bool = False,
    cameras: list[str] | None = None,
) -> None:
    """执行 replay 主流程。"""
    data_p = Path(data_path).expanduser().resolve()
    info: dict | None = None
    episode_meta: dict | None = None

    if _is_lerobot_dataset(data_p):
        info = _load_lerobot_info(data_p)
        if fps is None:
            fps = float(info.get("fps", 30.0))
            log(f"从 info.json 读取 fps={fps}")
        actions = load_actions_from_lerobot(data_p, episode=episode, column=column, start=start, end=end)
        # 获取目标 episode 的 chunk/file 信息（用于定位视频文件）
        if play_video and episode is not None:
            all_eps = _load_lerobot_episode_meta(data_p)
            episode_meta = next((e for e in all_eps if e["episode_index"] == episode), None)
    else:
        if fps is None:
            fps = 30.0
        if play_video:
            log("[WARN] --play-video 仅支持 LeRobot v3.0 数据集目录，忽略")
            play_video = False
        actions = load_actions(str(data_p), column=column, start=start, end=end)

    n_frames = len(actions)
    dt = 1.0 / (fps * speed)

    if dry_run:
        log(f"[DRY RUN] n_frames={n_frames}, fps={fps}, speed={speed}x, dt={dt:.4f}s")
        for i, act in enumerate(actions[:5]):
            log(f"  [{i:4d}] dim={len(act)} values={[round(float(v), 4) for v in act[:8]]}")
        if play_video and info is not None:
            all_video_keys = _find_video_keys(info)
            log(f"  数据集视频: {all_video_keys}")
        return

    # 准备视频播放器
    video_player: VideoPlayer | None = None
    if play_video and info is not None:
        all_video_keys = _find_video_keys(info)
        # 用户指定 cameras 时做短名匹配（如 "top" 匹配 "observation.images.top"）
        if cameras:
            selected = []
            for cam in cameras:
                matched = [k for k in all_video_keys if cam in k]
                if matched:
                    selected.extend(matched)
                else:
                    log(f"[WARN] 未找到匹配 '{cam}' 的视频，可用: {all_video_keys}")
            video_keys = list(dict.fromkeys(selected))  # 去重保序
        else:
            video_keys = all_video_keys

        if video_keys:
            chunk_idx = episode_meta["chunk_index"] if episode_meta else 0
            file_idx = episode_meta["file_index"] if episode_meta else 0
            video_paths: dict[str, Path] = {}
            for vk in video_keys:
                vp = _resolve_video_path(data_p, info, vk, chunk_idx, file_idx)
                if vp.exists():
                    # 窗口名用短名（去掉 observation.images. 前缀）
                    short = vk.replace("observation.images.", "")
                    video_paths[short] = vp
                else:
                    log(f"[WARN] 视频文件不存在: {vp}")
            if video_paths:
                video_player = VideoPlayer(video_paths)

    try:
        log("[1] 从配置创建机器人...")
        with BaseRobot.from_config(config_path) as robot:
            if robot.is_fault():
                log("[!] 检测到故障，clear_fault...")
                robot.clear_fault()
                time.sleep(1.5)
            robot.enable()
            if not robot.wait_until_operational(timeout_s=20.0):
                raise RuntimeError("机器人未能在超时时间内变为 operational")
            log(f"[2] 机器人 '{robot.name}' 已 operational (dof={robot.dof})")

            # replay 前先回 Home，从已知安全位置出发
            if home_before:
                log("[3] 回 Home（避免从任意姿态直接跳到第一帧）...")
                robot.go_home()

            log(f"[4] 开始 replay: {n_frames} 帧, fps={fps}, speed={speed}x")
            frame_count = 0
            log_interval = max(int(fps), 1)
            # start 偏移量，用于视频帧对齐
            frame_offset = start or 0

            for i, act in enumerate(actions):
                t_loop = time.perf_counter()
                values = [float(v) for v in act]
                try:
                    robot.act(Action(ActionSpace.JOINT_POSITION, values))
                except Exception as exc:
                    log(f"[!] 帧 {i} 执行失败: {exc}")
                    break
                frame_count += 1

                if video_player and video_player.active:
                    if not video_player.show_frame(frame_offset + i):
                        log("[!] 用户按 q 退出视频播放")
                        break

                if i % log_interval == 0:
                    s = robot.observe()
                    q = s.joint_positions
                    if len(q) >= 14:
                        log(f"  [{i:4d}/{n_frames}] lj0={q[0]:.4f} rj0={q[7]:.4f} lg={q[6]:.4f} rg={q[13]:.4f}")
                    else:
                        log(f"  [{i:4d}/{n_frames}] j0={q[0]:.4f}")

                elapsed = time.perf_counter() - t_loop
                if elapsed < dt:
                    time.sleep(dt - elapsed)

            log(f"[5] Replay 完成: {frame_count}/{n_frames} 帧")

            if home_after:
                log("[6] 回 Home...")
                robot.go_home()
                log("[7] Home 完成")
    finally:
        if video_player is not None:
            video_player.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 parquet / LeRobot v3.0 数据集 replay ARX5 动作")
    parser.add_argument("config", help="YAML 配置文件路径")
    parser.add_argument("data_path", help="parquet 文件路径 或 LeRobot v3.0 数据集目录")
    parser.add_argument("--fps", type=float, default=None, help="回放帧率 (LeRobot 数据集默认从 info.json 读取，否则 30)")
    parser.add_argument("--speed", type=float, default=1.0, help="回放速度倍率 (默认 1.0)")
    parser.add_argument("--column", type=str, default="action", help="parquet 中的动作列名 (默认 action)")
    parser.add_argument("--episode", type=int, default=None, help="LeRobot 数据集: 指定 episode 索引 (默认全部)")
    parser.add_argument("--start", type=int, default=None, help="起始帧索引（含）")
    parser.add_argument("--end", type=int, default=None, help="结束帧索引（不含）")
    parser.add_argument("--no-home-before", action="store_true", help="跳过 replay 前的 go_home")
    parser.add_argument("--home-after", action="store_true", help="replay 结束后回 Home")
    parser.add_argument("--dry-run", action="store_true", help="只打印动作信息，不连接机器人")
    parser.add_argument("--play-video", action="store_true", help="同步播放数据集中的视频 (仅 LeRobot 数据集)")
    parser.add_argument("--cameras", nargs="*", default=None, help="指定播放的相机名 (如 top left_wrist)，默认播放全部")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    replay(
        config_path=args.config,
        data_path=args.data_path,
        fps=args.fps,
        speed=args.speed,
        column=args.column,
        episode=args.episode,
        start=args.start,
        end=args.end,
        home_before=not args.no_home_before,
        home_after=args.home_after,
        dry_run=args.dry_run,
        play_video=args.play_video,
        cameras=args.cameras,
    )
