from utils.config import Config
from utils.registry_class import INFER_ENGINE

if __name__ == '__main__':
    cfg_update = Config(load=True)
    INFER_ENGINE.build(dict(type=cfg_update.TASK_TYPE), cfg_update=cfg_update.cfg_dict)