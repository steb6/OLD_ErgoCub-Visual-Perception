import os
from pathlib import Path

from utils.confort import BaseConfig, to_class
from configs.action_rec_config import ActionRec
import configs.grasping_config as Grasping


class Config(BaseConfig):
    class Manager:
        run_process = True
        docker = False

        file = 'scripts/manager.py'

        class Params:
            ip = 'localhost'
            port = 50000
            nodes = ['source', 'sink', 'grasping', 'action_rec']

    class Source:
        run_process = True
        docker = False

        file = 'scripts/source.py'

        class Parameters:
            ip = 'localhost'
            port = 50000

    Grasping = to_class(Grasping)
    class Grasping(Grasping): pass

    class ActionRec(ActionRec): pass

    class Sink:
        run_process = True
        docker = False

        file = 'scripts/sink.py'