import os
from pathlib import Path

from configs.action_rec_config import ActionRec
from configs.grasping_config import Grasping
from utils.confort import BaseConfig


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

    class Grasping(Grasping): pass
    class ActionRec(ActionRec): pass

    class Sink:
        run_process = True
        docker = False

        file = 'scripts/sink.py'