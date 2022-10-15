import os
import signal
from subprocess import Popen

from configs.config import Config


class Shell:
    def __init__(self, type='cmd'):
        self.process = None
        self.type = type
        self.cmds = []

    def add_pane(self, cmd):
        self.cmds += [cmd]

    def start(self):
        cmd = ''
        for i in range((len(self.cmds) - 1) // 4 + 1):
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

    python_procs = [p for p in Config if p.run_process and not p.docker]
    docker_procs = [p for p in Config if p.run_process and p.docker]

    for pr in python_procs:
        cmd = f'python {pr.file}'
        shell.add_pane(cmd)

    for pr in docker_procs:
        docker = pr.Docker
        cmd = f'docker run --name {docker.name}' \
              f' {" ".join(docker.options)}' \
              f' -v {" -v ".join(docker.volumes)}' \
              f' {docker.image} python {pr.file}'
        shell.add_pane(cmd)

    shell.start()

    print('Press any key to kill the containers')
    input()

    Popen(f'docker kill'.split(' ') + [pr.Docker.name for pr in docker_procs])
