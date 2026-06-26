# -- coding: UTF-8
"""
#!/usr/bin/python3 
"""
import argparse
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import sys


import cv2
import numpy as np
import rospy
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from openpi_client import image_tools, websocket_client_policy
from piper_msgs.msg import PosCmd
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header
import signal
import sys
import os
import threading
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


INFER_SYSTEM_ROOT = Path('/home/ubuntu/InferSystem')
if str(INFER_SYSTEM_ROOT) not in sys.path:
    sys.path.insert(0, str(INFER_SYSTEM_ROOT))

from Core import load_yaml
from Sensor.manager import SensorManager


DEG_TO_RAD = np.pi / 180.0
RAD_TO_DEG = 180.0 / np.pi
CAMERA_NAME_MAP = {
    "top_head": "observation.image.top_head",
    "hand_right": "observation.image.hand_right",
    "hand_left": "observation.image.hand_left",
}

try:
    from piper_sdk import C_PiperInterface_V2 as PiperInterface
except Exception:
    try:
        from piper_sdk import C_PiperInterface as PiperInterface
    except Exception:
        PiperInterface = None


@dataclass
class ArmJointSnapshot:
    position: list


# 内部统一使用 3 路相机的逻辑名，后面再映射到 InferSystem 的真实相机名。
CAMERA_NAMES = ["top_head", "hand_right", "hand_left"]

# 推理线程产出的 action chunk 会被写入这里，主控制线程只从这里逐步取下一帧动作。
stream_buffer = None   # type: StreamActionBuffer

# 仅保留最近两帧观测。这里不是做长时序建模，只是给推理线程一个最新快照。
observation_window = None

# lang_embeddings = "fold the cloth"
lang_embeddings = "fold the towels"

RIGHT_OFFSET = 0.003
published_actions_history = []  # list[np.ndarray(shape=(14,))]
observed_qpos_history = []      # list[np.ndarray(shape=(14,))]
publish_step_global = 0     
inferred_chunks = []            # list[dict(start_step:int, chunk:np.ndarray[chunk,14])]
inferred_chunks_lock = threading.Lock()
shutdown_event = threading.Event()

def inference_fn_non_blocking_fast(args, config, policy, ros_operator):
    """
    后台推理线程。

    这个线程只做两件事：
    1. 不断读取“当前最新观测”；
    2. 向 server 请求新的 action chunk，并把 chunk 交给 stream_buffer 做融合。

    它不直接控制机械臂。真正的控制频率由主线程里的 publish_rate 决定。
    因此这里的 inference_rate 变快，通常意味着 chunk 更新更及时，而不是机械臂必然更快。
    """
    global stream_buffer
    # assert stream_buffer is not None, "[inference_fn_non_blocking_fast] stream_buffer not initialized"

    global observation_window

    global lang_embeddings

    # global action_lock
    rate = rospy.Rate(getattr(args, "inference_rate", 4))
    while not rospy.is_shutdown():
        try:
            time1 = time.time()
            # 1) Get latest observation (non-blocking)
            update_observation_window(args, config, ros_operator)

            print("Get Observation Time", time.time() - time1, "s")
            time1 = time.time()

            latest_obs = observation_window[-1]
            imgs = [
                latest_obs["images"][config["camera_names"][0]],
                latest_obs["images"][config["camera_names"][1]],
                latest_obs["images"][config["camera_names"][2]],
            ]
            # BGR->RGB and pad/resize to model input size
            imgs = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in imgs]
            imgs = image_tools.resize_with_pad(np.array(imgs), 224, 224)
            proprio = latest_obs["qpos"]

            # 2) Build payload (field names match AgilexInputs on the server)
            payload = {
                "state": proprio,
                "images": {
                    "top_head": imgs[0].transpose(2, 0, 1),
                    "hand_right": imgs[1].transpose(2, 0, 1),
                    "hand_left": imgs[2].transpose(2, 0, 1),
                },
                "prompt": lang_embeddings,
            }


            # 3) Infer (block until current chunk is received, then start next immediately)
            # Expected return: {"actions": np.ndarray [chunk, state_dim]}
            # out = policy.infer(payload)
            # actions = out.get("actions", None)
            
            actions = policy.infer(payload)["actions"]
            
            # actions = binarize_gripper(actions)
            
            actions = scale_gripper(actions, scale=1.0)
            print("Inference Time", time.time() - time1, "s")
            time1 = time.time()

            # 4) Push to parallel buffer (consumed by main control loop / temporal smoothing)
            if actions is not None and len(actions) > 0:
                max_k = int(getattr(args, "latency_k", 0))
                min_m = int(getattr(args, "min_smooth_steps", 8))
                stream_buffer.integrate_new_chunk(actions, max_k=max_k, min_m=min_m)
            elif actions is None:
                print("actions is None")
            elif len(actions) == 0:
                print("len(actions) == 0")

            # 5) Continue to next round immediately (no sleep, no rate limit)
            print("Append Buffer Time", time.time() - time1, "s")
            time1 = time.time()
            # Throttle inference thread
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass

        except Exception as e:
            # Log and continue to next round so thread does not die; minimal yield to avoid saturating one core
            import traceback
            print(f"[inference_fn_non_blocking_fast] {type(e).__name__}: {e}")
            traceback.print_exc()
            rospy.logwarn(f"[inference_fn_non_blocking_fast] {e}")
            try:
                rate.sleep()
            except Exception:
                try:
                    time.sleep(0.001)
                except Exception:
                    pass
            continue


class StreamActionBuffer:
    """
    这个类是整个 temporal smoothing 的核心。

    可把它理解成“当前正在执行的动作序列 cur_chunk”。
    后台推理线程拿到新的 chunk 后，不会粗暴替换当前动作，而是调用 integrate_new_chunk():
    - 先根据已经发布的步数 k 做一点点前裁剪，补偿推理延迟；
    - 再把“旧 chunk 的前段”和“新 chunk 的前段”按线性权重做重叠融合；
    - 最终得到一个新的 cur_chunk，主线程继续按 publish_rate 一步步 pop 出去。

    这样做的效果是：
    - 新 chunk 到来时不会突然跳变；
    - 模型持续重规划时，动作过渡更连续；
    - 平滑主要发生在 chunk 的重叠区域，而不是对整段动作统一低通。
    """
    def __init__(self, max_chunks=10, decay_alpha=0.25, state_dim=14, smooth_method="temporal"):
        self.chunks = deque()                 # Kept for backward compatibility
        self.max_chunks = max_chunks
        self.lock = threading.Lock()
        self.decay_alpha = float(decay_alpha)  # Smoothing strength (exponential weight)
        self.state_dim = state_dim
        self.smooth_method = smooth_method
        self.cur_chunk = deque()              # Current sequence to publish (after smoothing)
        self.k = 0                            # Published step count (for latency trimming)
        self.last_action = None               # Last successfully popped action

    def push_chunk(self, actions_chunk: np.ndarray):
        """Legacy interface (no longer used)."""
        with self.lock:
            if actions_chunk is None or len(actions_chunk) == 0:
                return
            dq = deque([a.copy() for a in actions_chunk], maxlen=None)
            self.chunks.append(dq)
            while len(self.chunks) > self.max_chunks:
                self.chunks.popleft()

    def integrate_new_chunk(self, actions_chunk: np.ndarray, max_k: int, min_m: int = 8):
        """
        把新 chunk 融合进当前执行序列。

        参数说明：
        - max_k: 最多允许按“已发布步数 k”裁掉新 chunk 的多少步，用来补偿推理延迟。
          max_k 越大，新 chunk 越容易追上当前执行进度；过大时可能显得更激进。
        - min_m: 当旧 chunk 剩余太短时，至少扩展到多少步再做重叠融合。
          min_m 越大，旧动作对新动作的“拖拽”越明显，过渡更软，但也更容易显慢。

        融合方式不是指数平滑，而是重叠区线性插值：
        - 重叠区起点：100% 旧动作
        - 重叠区终点：100% 新动作
        - 中间逐步线性过渡
        """
        with self.lock:
            if actions_chunk is None or len(actions_chunk) == 0:
                return
            max_k = max(0, int(max_k))
            min_m = max(1, int(min_m))
            drop_n = min(self.k, max_k)
            if drop_n >= len(actions_chunk):
                # Entire chunk trimmed; skip this update
                return
            new_chunk = [a.copy() for a in actions_chunk[drop_n:]]
            # Build old sequence: if empty but last_action exists, extend with last_action to min_m steps;
            # if non-empty and len < m, pad tail to min_m; if both empty, take new sequence as-is
            if len(self.cur_chunk) == 0 and self.last_action is not None:
                old_list = [np.asarray(self.last_action, dtype=float).copy() for _ in range(min_m)]
                self.last_action = None
            else:
                old_list = list(self.cur_chunk)
                if len(old_list) > 0 and len(old_list) < min_m:
                    tail = np.asarray(old_list[-1], dtype=float).copy()
                    old_list.extend([tail.copy() for _ in range(min_m - len(old_list))])
                elif len(old_list) == 0:
                    self.cur_chunk = deque(new_chunk, maxlen=None)
                    self.k = 0
                    return
            new_list = list(new_chunk)

            # Overlap length = min of remaining old length and new length
            overlap_len = min(len(old_list), len(new_list))
            if overlap_len <= 0:
                # No overlap; use new sequence as-is
                self.cur_chunk = deque(new_list, maxlen=None)
                self.k = 0
                return

            # If old sequence is longer than new, trim old tail
            if len(old_list) > len(new_list):
                old_list = old_list[:len(new_list)]
                overlap_len = len(new_list)

            # Linear weights: first element 100% old, last element 0% old
            if overlap_len == 1:
                w_old = np.array([1.0], dtype=float)
            else:
                w_old = np.linspace(1.0, 0.0, overlap_len, dtype=float)
            w_new = 1.0 - w_old

            # Smooth the overlap region
            smoothed = [
                (w_old[i] * np.asarray(old_list[i], dtype=float) +
                 w_new[i] * np.asarray(new_list[i], dtype=float))
                for i in range(overlap_len)
            ]
            # Append the extra tail from the new sequence
            combined = smoothed + new_list[overlap_len:]
            self.cur_chunk = deque([a.copy() for a in combined], maxlen=None)
            self.k = 0

    

    def pop_left_step_from_all(self):
        """Legacy interface (no longer used)."""
        with self.lock:
            if len(self.cur_chunk) > 0:
                self.cur_chunk.popleft()

    def has_any(self):
        with self.lock:
            return len(self.cur_chunk) > 0

    def pop_next_action(self) -> np.ndarray | None:
        """Pop and return the next action to publish; k += 1."""
        with self.lock:
            if len(self.cur_chunk) == 0:
                return None
            # If about to pop the last element, save it as last_action
            if len(self.cur_chunk) == 1:
                self.last_action = np.asarray(self.cur_chunk[0], dtype=float).copy()
            act = np.asarray(self.cur_chunk.popleft(), dtype=float)
            self.k += 1
            return act

    def temporal_smooth_action_at_index(self, idx_from_left: int) -> np.ndarray | None:
        """Legacy interface (no longer used)."""
        with self.lock:
            if len(self.cur_chunk) == 0:
                return None
            return np.asarray(self.cur_chunk[0], dtype=float)

    def delta_eef_smooth_action_at_index(self, idx_from_left: int) -> np.ndarray | None:
        # TODO: implement this
        pass


# Start inference thread
def start_inference_thread(args, config, policy, ros_operator):
    inference_thread = threading.Thread(target=inference_fn_non_blocking_fast, args=(args, config, policy, ros_operator))
    inference_thread.daemon = True
    inference_thread.start()


def _on_sigint(signum, frame):
    # Set shutdown event so main loop can flush and save
    try:
        shutdown_event.set()
    except Exception:
        pass
    # Notify ROS to shut down
    try:
        rospy.signal_shutdown("SIGINT")
    except Exception:
        pass
    # Do not exit immediately in signal handler; let main flow clean up


def binarize_gripper(actions: np.ndarray, threshold: float = 0.005, max_value: float = 0.1) -> np.ndarray:
    """Hard-threshold gripper channels (indices 6 and 13) in an action chunk.

    Values strictly below `threshold` are clamped to 0; values >= threshold are
    snapped to `max_value`. Operates in-place and returns the same array.
    """
    if actions is None or len(actions) == 0:
        return actions
    actions = np.array(actions, dtype=float, copy=True)
    for idx in (6, 13):
        col = actions[:, idx]
        actions[:, idx] = np.where(col < threshold, 0.0, max_value)
    return actions
 
def scale_gripper(actions: np.ndarray, scale: float = 10.0):
    if actions is None or len(actions) == 0:
        return actions
    actions = np.array(actions, dtype=float, copy=True)
    for idx in (6, 13):
        col = actions[:, idx]
        actions[:, idx] *= scale
    return actions


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


# 这是“单步大跳变时的线性补点”工具函数。
# 当前 direct_can 版本的主路径主要依赖 chunk overlap smoothing，
# 这个函数更像一个备用的局部插值器，而不是主平滑机制。
def interpolate_action(args, prev_action, cur_action):
    steps = np.concatenate((np.array(args.arm_steps_length), np.array(args.arm_steps_length)), axis=0)
    diff = np.abs(cur_action - prev_action)
    step = np.ceil(diff / steps).astype(int)
    step = np.max(step)
    if step <= 1:
        return cur_action[np.newaxis, :]
    new_actions = np.linspace(prev_action, cur_action, step + 1)
    return new_actions[1:]


def get_config(args):
    config = {
        "episode_len": args.max_publish_step,
        "state_dim": 14,
        "chunk_size": args.chunk_size,
        "camera_names": CAMERA_NAMES,
    }
    return config


# Get the observation from the ROS topic
def get_ros_observation(args, ros_operator):
    rate = rospy.Rate(args.publish_rate)
    print_flag = True
    time3 = time.time()

    while True and not rospy.is_shutdown():
        result = ros_operator.get_frame()
        if time.time() - time3 > 0.01:
            print("Get Frame Time is too long", time.time() - time3, "s")
        if not result:
            if print_flag:
                print("syn fail when get_ros_observation")
                print_flag = False
            rate.sleep()
            continue
        print_flag = True
        (
            img_front,
            img_left,
            img_right,
            img_front_depth,
            img_left_depth,
            img_right_depth,
            puppet_arm_left,
            puppet_arm_right,
            robot_base,
        ) = result
        # print(f"sync success when get_ros_observation")
        return (img_front, img_left, img_right, puppet_arm_left, puppet_arm_right)


# 更新 observation_window。
# 这里会把相机图像做一次 PNG encode/decode，保持无损编码路径。
def update_observation_window(args, config, ros_operator):
    def png_mapping(img):
        img = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])[1].tobytes()
        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        return img

    global observation_window
    if observation_window is None:
        observation_window = deque(maxlen=2)

        # Append the first dummy image
        observation_window.append(
            {
                "qpos": None,
                "images": {
                    config["camera_names"][0]: None,
                    config["camera_names"][1]: None,
                    config["camera_names"][2]: None,
                },
            }
        )

    # time2 = time.time()
    img_front, img_left, img_right, puppet_arm_left, puppet_arm_right = get_ros_observation(args, ros_operator)
    # print("Get Observation Time", time.time() - time2, "s")
    img_front = png_mapping(img_front)
    img_left = png_mapping(img_left)
    img_right = png_mapping(img_right)

    qpos = np.concatenate(
        (np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)),
        axis=0,
    )

    observation_window.append(
        {
            "qpos": qpos,
            "images": {
                config["camera_names"][0]: img_front,
                config["camera_names"][1]: img_right,
                config["camera_names"][2]: img_left,
            },
        }
    )


def inference_fn(args, config, policy):
    global observation_window
    global lang_embeddings

    # print(f"Start inference_thread_fn: t={t}")
    while True and not rospy.is_shutdown():
        # time1 = time.time()

        # fetch images in sequence [front, right, left]
        image_arrs = [
            observation_window[-1]["images"][config["camera_names"][0]],
            observation_window[-1]["images"][config["camera_names"][1]],
            observation_window[-1]["images"][config["camera_names"][2]],
        ]
        # convert bgr ro rgb
        image_arrs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in image_arrs]
        image_arrs = image_tools.resize_with_pad(np.array(image_arrs), 224, 224)

        # get last qpos in shape [14, ]
        proprio = observation_window[-1]["qpos"]

        payload = {
            "state": proprio,
            "images": {
                "top_head": image_arrs[0].transpose(2, 0, 1),
                "hand_right": image_arrs[1].transpose(2, 0, 1),
                "hand_left": image_arrs[2].transpose(2, 0, 1),
            },
            "prompt": lang_embeddings,
        }

        time1 = time.time()

        # actions shaped as [64, 14] in format [left, right]
        actions = policy.infer(payload)["actions"]


        print(f"Model inference time: {(time.time() - time1)*1000:.3f} ms")

        return actions


# 主控制循环。
# 这个函数里有两条并行链路：
# 1. 后台推理线程持续产出新的 action chunk；
# 2. 当前线程按 publish_rate 稳定地从 stream_buffer 里取下一帧动作并下发。
#
# smooth 的本质不是“把速度降下来”，而是把 chunk 和 chunk 之间的衔接变连续。
# 真正决定机械臂体感速度的主要还是 publish_rate、arm_speed_percent 和模型动作幅度。
def model_inference(args, config, ros_operator):
    global lang_embeddings

    global stream_buffer

    # Load client
    print("host: ", args.host, " port: ", args.port)
    
    policy = websocket_client_policy.WebsocketClientPolicy(
        args.host,
        args.port,
    )
    print(f"Server metadata: {policy.get_server_metadata()}")

    max_publish_step = config["episode_len"]
    chunk_size = config["chunk_size"]

    # Initialize position of the puppet arm
    left0 = [0, 0.32, -0.36, 0, 0.24, 0, 0.07]
    # left0 = [0, 0.32, -0.36, 0, 0.24, 0, 0.0]
    right0 = [0, 0.32, -0.36, 0, 0.24, 0, 0.07]
    # right0 = [0.0042737800000000005, -0.020549032000000002, 0.005773964, 0.020392036000000002, 0.413108808, 0.08352187200000001, 0.0975]
    # right0 = [0, 0.32, -0.36, 0, 0.24, 0, 0.0]

    ros_operator.puppet_arm_publish_continuous(left0, right0)
    input("Press enter to continue")
    ros_operator.puppet_arm_publish_continuous(left0, right0)
    # Warmup: run one blocking inference (result discarded)
    try:
        update_observation_window(args, config, ros_operator)
        latest_obs = observation_window[-1]
        image_arrs = [
            latest_obs["images"][config["camera_names"][0]],
            latest_obs["images"][config["camera_names"][1]],
            latest_obs["images"][config["camera_names"][2]],
        ]
        image_arrs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in image_arrs]
        image_arrs = image_tools.resize_with_pad(np.array(image_arrs), 224, 224)
        proprio = latest_obs["qpos"]
        payload = {
            "state": proprio,
            "images": {
                "top_head": image_arrs[0].transpose(2, 0, 1),
                "hand_right": image_arrs[1].transpose(2, 0, 1),
                "hand_left": image_arrs[2].transpose(2, 0, 1),
            },
            "prompt": lang_embeddings,
        }
        try:
            _ = policy.infer(payload)
        except Exception as e:
            rospy.logwarn(f"[startup_warmup_infer] {e}")
    except Exception as e:
        rospy.logwarn(f"[startup_warmup_prep] {e}")
    # Initialize the previous action to be the initial robot state
    pre_action = np.zeros(config["state_dim"])
    action = None
    # Persistent accumulated joint correction offsets (first 6 joints)
    corr_left_q6 = np.zeros(6, dtype=float)
    corr_right_q6 = np.zeros(6, dtype=float)
    # Inference loop
    with torch.inference_mode():
        while True and not rospy.is_shutdown():
            # The current time step
            t = 0
            rate = rospy.Rate(args.publish_rate)

            action_buffer = np.zeros([chunk_size, config["state_dim"]])

            while t < max_publish_step and not rospy.is_shutdown() and not shutdown_event.is_set():             
                if shutdown_event.is_set():
                    break
                if args.use_temporal_smoothing:

                    if stream_buffer is None:
                        stream_buffer = StreamActionBuffer(
                            max_chunks=args.buffer_max_chunks,
                            decay_alpha=args.exp_decay_alpha,
                            state_dim=config["state_dim"],
                            smooth_method="temporal",
                        )
                        start_inference_thread(args, config, policy, ros_operator)
                    act = stream_buffer.pop_next_action()
                    if act is not None:
                        if args.ctrl_type == "joint":
                            left_action = act[:7].copy()
                            right_action = act[7:14].copy()
                            left_action[6] = max(0.0, left_action[6]-RIGHT_OFFSET)
                            right_action[6] = max(0.0, right_action[6]-RIGHT_OFFSET)
                            ros_operator.puppet_arm_publish(left_action, right_action)
                            published_actions_history.append(np.concatenate([left_action, right_action], axis=0).astype(float))

                        else:
                            print("Make sure ctrl_type is joint")
                    else:
                        print("act is None")
                        time.sleep(0.001)
                        continue
                    print("Published Step", t)
                    try:
                        publish_step_global = len(published_actions_history)
                    except Exception:
                        pass

                    rate.sleep()
                    t += 1
                else:
                    print("Make sure touse temporal smoothing")

                if shutdown_event.is_set():
                    break


# 这里虽然名字还叫 RosOperator，但 direct_can 版本实际上是一个“混合适配器”：
# - 机械臂：通过 piper_sdk 直连 can0/can1；
# - 相机：通过 InferSystem 的 SensorManager 直接读取；
# - ROS：这里只剩下 node 生命周期和可选的底盘 topic。
class RosOperator:
    def __init__(self, args):
        self.communication_thread = None
        self.communication_flag = False
        self.lock = threading.Lock()
        self.robot_base_deque = None
        self.bridge = None
        self.robot_base_publisher = None
        self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_lock = None
        self.args = args
        self.left_arm = None
        self.right_arm = None
        self.sensors = None
        self.enabled_camera_names = []
        self.init()
        self._init_direct_can()
        self._init_sensors()
        self.init_ros()

    def init(self):
        self.bridge = CvBridge()
        self.robot_base_deque = deque()
        self.puppet_arm_publish_lock = threading.Lock()
        self.puppet_arm_publish_lock.acquire()

    def _init_direct_can(self):
        # 直连 CAN 后，机械臂状态读取和动作发送都不再依赖 /puppet/joint_left 之类的 ROS topic。
        if PiperInterface is None:
            raise ImportError('piper_sdk is required for direct CAN smoothing control')
        self.left_arm = PiperInterface(self.args.can_interface_left)
        self.right_arm = PiperInterface(self.args.can_interface_right)
        self.left_arm.ConnectPort()
        self.right_arm.ConnectPort()
        self._enable_arm(self.left_arm, 'left', self.args.can_interface_left)
        self._enable_arm(self.right_arm, 'right', self.args.can_interface_right)
        for arm in (self.left_arm, self.right_arm):
            motion_ctrl = getattr(arm, 'MotionCtrl_2', None)
            if callable(motion_ctrl):
                motion_ctrl(0x01, 0x01, int(self.args.arm_speed_percent), 0x00)

    def _init_sensors(self):
        # 直接复用 InferSystem 的传感器配置，避免再走 ROS 图像同步带来的等待和时间戳对齐问题。
        sensor_config = load_yaml(self.args.sensor_config)
        self.enabled_camera_names = sensor_config.get('inference', {}).get('enabled_cameras') or [
            'observation.image.left_wrist_view',
            'observation.image.right_wrist_view',
            'observation.image.third_view',
        ]
        self.sensors = SensorManager.from_config(sensor_config)
        self.sensors.open_all()

    def close(self):
        if self.sensors is not None:
            try:
                self.sensors.close_all()
            except Exception:
                pass
            self.sensors = None
        for arm in (self.left_arm, self.right_arm):
            if arm is None:
                continue
            disconnect = getattr(arm, 'DisconnectPort', None)
            if callable(disconnect):
                try:
                    disconnect()
                except Exception:
                    pass

    def _enable_arm(self, arm, arm_name, can_name):
        deadline = time.monotonic() + float(self.args.arm_enable_timeout)
        enable_fn = getattr(arm, 'EnablePiper', None)
        if not callable(enable_fn):
            return
        while True:
            try:
                if enable_fn():
                    return
            except Exception:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError(f'EnablePiper timed out for {arm_name} arm on {can_name}')
            time.sleep(0.01)

    def _read_single_arm(self, arm):
        joint_state = arm.GetArmJointMsgs().joint_state
        joints_rad = [
            float(joint_state.joint_1) / 1000.0 * DEG_TO_RAD,
            float(joint_state.joint_2) / 1000.0 * DEG_TO_RAD,
            float(joint_state.joint_3) / 1000.0 * DEG_TO_RAD,
            float(joint_state.joint_4) / 1000.0 * DEG_TO_RAD,
            float(joint_state.joint_5) / 1000.0 * DEG_TO_RAD,
            float(joint_state.joint_6) / 1000.0 * DEG_TO_RAD,
        ]
        gripper_m = 0.0
        try:
            gripper = arm.GetArmGripperMsgs()
            gripper_m = float(gripper.gripper_state.grippers_angle) / 10000000.0
        except Exception:
            pass
        return joints_rad + [gripper_m]

    def get_joint_snapshots(self):
        left = self._read_single_arm(self.left_arm)
        right = self._read_single_arm(self.right_arm)
        return ArmJointSnapshot(left), ArmJointSnapshot(right)

    def puppet_arm_publish(self, left, right):
        left_joints_sdk = [int(round(float(v) * RAD_TO_DEG * 1000.0)) for v in left[:6]]
        right_joints_sdk = [int(round(float(v) * RAD_TO_DEG * 1000.0)) for v in right[:6]]
        left_gripper_sdk = int(round(float(left[6]) * 10000000.0))
        right_gripper_sdk = int(round(float(right[6]) * 10000000.0))
        self.left_arm.JointCtrl(*left_joints_sdk)
        self.right_arm.JointCtrl(*right_joints_sdk)
        self.left_arm.GripperCtrl(abs(left_gripper_sdk), 1000, 0x01, 0)
        self.right_arm.GripperCtrl(abs(right_gripper_sdk), 1000, 0x01, 0)

    def endpose_publish(self, left, right):
        raise NotImplementedError('Direct CAN mode only supports joint control')

    def robot_base_publish(self, vel):
        vel_msg = Twist()
        vel_msg.linear.x = vel[0]
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = vel[1]
        self.robot_base_publisher.publish(vel_msg)

    def puppet_arm_publish_continuous(self, left, right):
        rate = rospy.Rate(self.args.publish_rate)
        left_arm = None
        right_arm = None
        while True and not rospy.is_shutdown():
            try:
                left_snap, right_snap = self.get_joint_snapshots()
                left_arm = list(left_snap.position)
                right_arm = list(right_snap.position)
                break
            except Exception as e:
                rospy.logwarn_throttle(2.0, f'Waiting for direct CAN arm states: {e}')
                rate.sleep()

        if rospy.is_shutdown():
            rospy.loginfo('ROS shutdown while waiting for arm states; skip initial arm motion.')
            return

        left_symbol = [1 if left[i] - left_arm[i] > 0 else -1 for i in range(len(left))]
        right_symbol = [1 if right[i] - right_arm[i] > 0 else -1 for i in range(len(right))]
        flag = True
        step = 0
        while flag and not rospy.is_shutdown():
            if self.puppet_arm_publish_lock.acquire(False):
                return
            left_diff = [abs(left[i] - left_arm[i]) for i in range(len(left))]
            right_diff = [abs(right[i] - right_arm[i]) for i in range(len(right))]
            flag = False
            for i in range(len(left)):
                if left_diff[i] < self.args.arm_steps_length[i]:
                    left_arm[i] = left[i]
                else:
                    left_arm[i] += left_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            for i in range(len(right)):
                if right_diff[i] < self.args.arm_steps_length[i]:
                    right_arm[i] = right[i]
                else:
                    right_arm[i] += right_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            self.puppet_arm_publish(left_arm, right_arm)
            step += 1
            print('puppet_arm_publish_continuous:', step)
            rate.sleep()

    def puppet_arm_publish_linear(self, left, right):
        num_step = 100
        rate = rospy.Rate(200)
        left_snap, right_snap = self.get_joint_snapshots()
        left_arm = list(left_snap.position)
        right_arm = list(right_snap.position)
        traj_left_list = np.linspace(left_arm, left, num_step)
        traj_right_list = np.linspace(right_arm, right, num_step)
        for i in range(len(traj_left_list)):
            traj_left = traj_left_list[i]
            traj_right = traj_right_list[i]
            traj_left[-1] = left[-1]
            traj_right[-1] = right[-1]
            self.puppet_arm_publish(traj_left, traj_right)
            rate.sleep()

    def puppet_arm_publish_continuous_thread(self, left, right):
        if self.puppet_arm_publish_thread is not None:
            self.puppet_arm_publish_lock.release()
            self.puppet_arm_publish_thread.join()
            self.puppet_arm_publish_lock.acquire(False)
            self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_thread = threading.Thread(target=self.puppet_arm_publish_continuous, args=(left, right))
        self.puppet_arm_publish_thread.start()

    def get_frame(self):
        # 这里不做 ROS 严格同步，而是直接拿 InferSystem 当前读到的最新三路图像。
        # 对 smooth 来说，更重要的是“尽快拿到最新观测”，而不是为了时间戳完全一致去等待。
        if self.sensors is None:
            return False
        images = self.sensors.read_images(self.enabled_camera_names)
        required = set(self.enabled_camera_names)
        if not required.issubset(images.keys()):
            missing = sorted(required.difference(images.keys()))
            rospy.logwarn_throttle(2.0, f'Missing camera frames from InferSystem sensors: {missing}')
            return False
        left_snap, right_snap = self.get_joint_snapshots()
        img_front = images[CAMERA_NAME_MAP['top_head']]
        img_right = images[CAMERA_NAME_MAP['hand_right']]
        img_left = images[CAMERA_NAME_MAP['hand_left']]
        return (img_front, img_left, img_right, None, None, None, left_snap, right_snap, None)

    def robot_base_callback(self, msg):
        if len(self.robot_base_deque) >= 2000:
            self.robot_base_deque.popleft()
        self.robot_base_deque.append(msg)

    def init_ros(self):
        rospy.init_node('joint_state_publisher', anonymous=True)
        rospy.Subscriber(self.args.robot_base_topic, Odometry, self.robot_base_callback, queue_size=1000, tcp_nodelay=True)
        self.robot_base_publisher = rospy.Publisher(self.args.robot_base_cmd_topic, Twist, queue_size=10)


def get_arguments():
    parser = argparse.ArgumentParser()

    # 下面这些超参里，和 smooth 体感最相关的是：
    # - publish_rate: 主线程发布动作的频率，越高通常越丝滑；
    # - inference_rate: 后台更新 chunk 的频率，越高通常越跟手；
    # - latency_k: 新 chunk 到来时允许跳过的前几步，用来补偿推理延迟；
    # - min_smooth_steps: 新旧 chunk 至少重叠多少步后再做融合，越大越软；
    # - arm_speed_percent: Piper 底层运动速度百分比，过低会显慢。
    parser.add_argument(
        "--max_publish_step",
        action="store",
        type=int,
        help="Maximum number of action publishing steps",
        default=10000,
        required=False,
    )
    parser.add_argument(
        "--seed",
        action="store",
        type=int,
        help="Random seed",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--img_front_topic",
        action="store",
        type=str,
        help="img_front_topic",
        default="/camera_f/color/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_left_topic",
        action="store",
        type=str,
        help="img_left_topic",
        default="/camera_l/color/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_right_topic",
        action="store",
        type=str,
        help="img_right_topic",
        default="/camera_r/color/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_front_depth_topic",
        action="store",
        type=str,
        help="img_front_depth_topic",
        default="/camera_f/depth/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_left_depth_topic",
        action="store",
        type=str,
        help="img_left_depth_topic",
        default="/camera_l/depth/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_right_depth_topic",
        action="store",
        type=str,
        help="img_right_depth_topic",
        default="/camera_r/depth/image_raw",
        required=False,
    )
    parser.add_argument(
        "--puppet_arm_left_cmd_topic",
        action="store",
        type=str,
        help="puppet_arm_left_cmd_topic",
        default="/master/joint_left",
        required=False,
    )
    parser.add_argument(
        "--puppet_arm_right_cmd_topic",
        action="store",
        type=str,
        help="puppet_arm_right_cmd_topic",
        default="/master/joint_right",
        required=False,
    )
    parser.add_argument(
        "--puppet_arm_left_topic",
        action="store",
        type=str,
        help="puppet_arm_left_topic",
        default="/puppet/joint_left",
        required=False,
    )
    parser.add_argument(
        "--puppet_arm_right_topic",
        action="store",
        type=str,
        help="puppet_arm_right_topic",
        default="/puppet/joint_right",
        required=False,
    )
    parser.add_argument(
        "--endpose_left_cmd_topic",
        action="store",
        type=str,
        help="endpose_left_cmd_topic",
        default="/pos_cmd_left",
        required=False,
    )
    parser.add_argument(
        "--endpose_right_cmd_topic",
        action="store",
        type=str,
        help="endpose_right_cmd_topic",
        default="/pos_cmd_right",
        required=False,
    )

    parser.add_argument(
        "--robot_base_topic",
        action="store",
        type=str,
        help="robot_base_topic",
        default="/odom_raw",
        required=False,
    )
    parser.add_argument(
        "--robot_base_cmd_topic",
        action="store",
        type=str,
        help="robot_base_topic",
        default="/cmd_vel",
        required=False,
    )
    parser.add_argument(
        "--use_robot_base",
        action="store_true",
        help="Whether to use the robot base to move around",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--publish_rate",
        action="store",
        type=int,
        help="The rate at which to publish the actions",
        default=30,
        required=False,
    )
    parser.add_argument(
        "--chunk_size",
        action="store",
        type=int,
        help="Action chunk size",
        default=50,
        required=False,
    )
    parser.add_argument(
        "--arm_steps_length",
        action="store",
        type=float,
        help="The maximum change allowed for each joint per timestep",
        default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2],
        required=False,
    )
    parser.add_argument(
        "--use_actions_interpolation",
        action="store_true",
        help="Whether to interpolate the actions if the difference is too large",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--use_depth_image",
        action="store_true",
        help="Whether to use depth images",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--sensor_config",
        action="store",
        type=str,
        help="InferSystem sensor YAML config path",
        default="/home/ubuntu/InferSystem/Config/aloha_dj_kai0.yaml",
        required=False,
    )
    parser.add_argument(
        "--can_interface_left",
        action="store",
        type=str,
        help="Direct CAN interface for left arm",
        default="can0",
        required=False,
    )
    parser.add_argument(
        "--can_interface_right",
        action="store",
        type=str,
        help="Direct CAN interface for right arm",
        default="can1",
        required=False,
    )
    parser.add_argument(
        "--arm_enable_timeout",
        action="store",
        type=float,
        help="Timeout seconds when enabling each Piper arm",
        default=5.0,
        required=False,
    )
    parser.add_argument(
        "--arm_speed_percent",
        action="store",
        type=int,
        help="MotionCtrl_2 speed percent for each Piper arm",
        default=80,
        required=False,
    )

    parser.add_argument(
        "--host",
        action="store",
        type=str,
        help="Websocket server host",
        default="localhost",
        required=False,
    )
    parser.add_argument(
        "--port",
        action="store",
        type=int,
        help="Websocket server port",
        default=8000,
        required=False,
    )

    parser.add_argument(
        "--ctrl_type",
        type=str,
        choices=["joint", "eef"],
        help="Control type for the robot arm",
        default="joint",
    )

    parser.add_argument(
        "--use_temporal_smoothing",
        action="store_true",
        help="Enable non-blocking communication and control execution",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--latency_k",
        type=int,
        help="Max Latency in steps",
        default=8,
        required=False,
    )
    parser.add_argument(
        "--inference_rate",
        type=float,
        help="Inference loop rate (Hz)",
        default=3.0,
        required=False,
    )
    parser.add_argument(
        "--min_smooth_steps",
        type=int,
        help="Minimum smoothing steps m",
        default=8,
        required=False,
    )
    parser.add_argument(
        "--buffer_max_chunks",
        type=int,
        help="Maximum number of chunks in the buffer",
        default=10,
        required=False,
    )
    parser.add_argument(
        "--exp_decay_alpha",
        type=float,
        help="Exponential decay alpha",
        default=0.25,
        required=False,
    )
    
    parser.add_argument(
        "--use_delta_eef_smoothing",
        action="store_true",
        help="Whether to use delta eef smoothing",
        default=False,
        required=False,
    )

    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    ros_operator = RosOperator(args)
    if args.seed is not None:
        set_seed(args.seed)
    config = get_config(args)
    signal.signal(signal.SIGINT, _on_sigint)
    try:
        model_inference(args, config, ros_operator)
    except KeyboardInterrupt:
        sys.exit(1)
    finally:
        try:
            ros_operator.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
