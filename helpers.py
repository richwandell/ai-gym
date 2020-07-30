from tensorflow.core.protobuf.config_pb2 import ConfigProto, GPUOptions
from tensorflow.python.client.session import Session
from tensorflow.python.keras.backend import set_session


def fix_mem_issue():
    config = ConfigProto(
        gpu_options=GPUOptions(per_process_gpu_memory_fraction=0.8)
    )
    config.gpu_options.allow_growth = True
    session = Session(config=config)
    set_session(session)

