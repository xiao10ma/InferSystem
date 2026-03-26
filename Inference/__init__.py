from .dispatch import ActionDispatcher, build_state_vector

__all__ = [
    "ActionDispatcher",
    "build_state_vector",
]

# client/server 需要 zmq + msgpack，按需导入
try:
    from .client import InferenceClient
    from .server import InferenceServer

    __all__ += ["InferenceClient", "InferenceServer"]
except ImportError:
    pass
