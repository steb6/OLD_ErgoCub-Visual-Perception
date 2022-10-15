import os
from pathlib import Path
from subprocess import Popen

from utils.confort import BaseConfig, init


class Config(BaseConfig):
    class Docker:
        process = False

        docker_env = 'ecub-env'
        docker_options = ['-it', '--rm', '--gpus=all']
        volumes = [f'{Path(os.getcwd()).as_posix()}:/home/ecub']

        docker_run = f'docker run {" ".join(docker_options)} -v {" -v ".join(volumes)} {docker_env}'
        run = lambda x, cmd=docker_run: Popen(cmd.split(' ') + ['python', x])

    class Manager:
        process = True
        docker = False

        file = 'manager.py'

    class Source:
        process = True
        docker = False

        file = 'source.py'

    class Grasping:
        process = True
        docker = True

        file = 'grasping/grasping_process.py'

    class Action:
        process = True
        docker = True

        file = 'human/human_process.py'

    class Sink:
        process = True
        docker = False

        file = 'sink.py'


class Shell:
    def __init__(self, type='cmd'):
        self.type = type
        self.cmds = []

    def add_pane(self, cmd):
        self.cmds += [cmd]

    def start(self):
        cmd = ''
        for i in range((len(self.cmds) -1) // 4 + 1):
            cmd += 'wt -M' if i == 0 else ';'

            if len(self.cmds) >= (i * 4) + 1:
                cmd += f' --title ecub-visual-pipeline --suppressApplicationTitle -d {os.getcwd()} --colorScheme "Solarized Dark" {self.cmds[(i * 4) + 0]}'
            if len(self.cmds) >= (i * 4) + 2:
                cmd += f' ;split-pane -V -d {os.getcwd()} --colorScheme "Solarized Dark" {self.cmds[(i * 4) + 1]}' \
                       f' ;move-focus left'
            if len(self.cmds) >= (i * 4) + 3:
                cmd += f' ;split-pane -H -d {os.getcwd()} --colorScheme "Solarized Dark" {self.cmds[(i * 4) + 2]}' \
                       f' ;move-focus right'
            if len(self.cmds) >= (i * 4) + 4:
                cmd += f' ;split-pane -H -d {os.getcwd()} --colorScheme "Solarized Dark" {self.cmds[(i * 4) + 3]}'

        os.system(cmd)

if __name__ == '__main__':
    shell = Shell('cmd')
    for c in Config:
        if c.process:
            cmd = f'python {c.file}'
            if c.docker:
                cmd = f'{Config.Docker.docker_run} {cmd}'
            shell.add_pane(cmd)
    shell.start()