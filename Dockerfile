FROM andrewr96/ecub-env:latest

USER root
WORKDIR /
# Install essentials
RUN apt update && \
    apt install --no-install-recommends -y -qq apt-utils build-essential ca-certificates cmake cmake-curses-gui curl emacs-nox git git-lfs glmark2 gnupg2 gpg htop iputils-ping jq libssl-dev libusb-1.0-0-dev libz-dev lsb-release mesa-utils nano psmisc sudo unzip vim wget zip && \
    rm -rf /var/lib/apt/lists/*

# Github CLI
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg && \
    chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null && \
    apt update && \
    apt install --no-install-recommends -y -qq gh && \
    rm -rf /var/lib/apt/lists/*

# Install additional dependencies
RUN apt update && \
    apt install --no-install-recommends -y -qq eog libassimp-dev libconfig++-dev libglfw3-dev libglew-dev libgtk2.0-dev libglm-dev libeigen3-dev libpython3-dev libqt5svg5 libtclap-dev libvtk7-dev && \
    git clone --progress https://github.com/robotology/robotology-superbuild && cd robotology-superbuild && bash scripts/install_apt_dependencies.sh && \
    rm -rf /var/lib/apt/lists/*


# Build robotology-superbuild
RUN git config --global user.name "user" && \
    git config --global user.email "user@email.com" && \
    cd robotology-superbuild && \
    git checkout v2022.09.0 && \
    mkdir build && cd build && \
    cmake -DROBOTOLOGY_ENABLE_CORE=ON -DROBOTOLOGY_USES_GAZEBO=OFF -DROBOTOLOGY_USES_PYTHON=ON -DROBOTOLOGY_USES_LUA=ON -DYCM_EP_ADDITIONAL_CMAKE_ARGS:STRING="-DICUB_COMPILE_BINDINGS:BOOL=ON -DCREATE_PYTHON:BOOL=ON -DENABLE_yarpmod_RGBDSensorWrapper:BOOL=ON -DENABLE_yarpmod_RGBDSensorClient:BOOL=ON -DENABLE_yarpcar_mjpeg:BOOL=ON -DENABLE_yarppm_depthimage_to_rgb:BOOL=ON -DENABLE_yarppm_depthimage_compression_zlib:BOOL=ON" ../ && \
    make -j1
#    echo "source /robotology-superbuild/build/install/share/robotology-superbuild/setup.sh" >> /home/ecub/.bashrc
RUN echo "source /robotology-superbuild/build/install/share/robotology-superbuild/setup.sh" >> /home/ecub/.bashrc
# Create user with passwordless sudo
RUN sudo usermod -a -G sudo ecub && \
    sed -i.bkp -e 's/%sudo\s\+ALL=(ALL\(:ALL\)\?)\s\+ALL/%sudo ALL=NOPASSWD:ALL/g' /etc/sudoers

# Switch to user
USER ecub
WORKDIR /home/ecub
CMD ["bash"]
