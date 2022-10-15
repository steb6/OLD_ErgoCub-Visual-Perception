import os
from pathlib import Path


class ActionRec:
    run_process = False
    docker = True

    class Docker:
        image = 'ecub-env'
        name = 'ecub-human'
        options = ['-it', '--rm', '--gpus=all']
        volumes = [f'{Path(os.getcwd()).as_posix()}:/home/ecub']

    file = 'human/human_process.py'
