import os
from pathlib import Path


class Grasping:
    run_process = True
    docker = True

    class Docker:
        image = 'ecub-env'
        name = 'ecub-grasping'
        options = ['-it', '--rm', '--gpus=all']
        volumes = [f'{Path(os.getcwd()).as_posix()}:/home/ecub']

    file = 'scripts/grasping_process.py'

    class Logging:
        log = True

    class Network:
        ip = 'locahost'
        port = 50000

    class Segmentation:
        engine = ''

    class ShapeCompletion:
        engine = ''

    class Grasping:
        engine = ''

