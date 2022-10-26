import os
from pathlib import Path

from utils.confort import BaseConfig, to_class
from configs.action_rec_config import ActionRec

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

    class Grasping:
        run_process = True
        docker = True
        file = 'scripts/grasping_pipeline.py'

        class Docker:
            image = 'ecub'
            name = 'ecub-grasping'
            options = ['-it', '--rm', '--gpus=all']
            volumes = [f'{Path(os.getcwd()).as_posix()}:/home/ecub']

    class ActionRec(ActionRec): pass

    class Sink:
        run_process = True
        docker = False

        file = 'scripts/sink.py'