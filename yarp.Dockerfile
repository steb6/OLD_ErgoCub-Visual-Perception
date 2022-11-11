FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
ARG NJOBS=1
LABEL org.opencontainers.image.title=""
LABEL org.opencontainers.image.source=""
LABEL org.opencontainers.image.authors="Nicola A. Piga <nicola.piga@iit.it>"

# Use /bin/bash instead of /bin/sh
SHELL ["/bin/bash", "-c"]

# Non-interactive installation mode
ENV DEBIAN_FRONTEND=noninteractive

# Set the locale
RUN apt update && \
    apt install -y -qq locales && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8

# Install essentials
RUN apt update && \
    apt install --no-install-recommends -y -qq apt-utils build-essential ca-certificates cmake cmake-curses-gui curl emacs-nox git git-lfs glmark2 gnupg2 gpg htop iputils-ping jq libssl-dev libusb-1.0-0-dev libz-dev lsb-release mesa-utils nano psmisc python3-pip python3-virtualenv sudo unzip vim wget zip && \
    rm -rf /var/lib/apt/lists/*

# Install GitHub cli
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
    make -j$NJOBS

# Build RobotsIO
RUN source /robotology-superbuild/build/install/share/robotology-superbuild/setup.sh && \
    git clone --progress https://github.com/xenvre/robots-io && \
    cd robots-io && mkdir build && cd build && \
    cmake -DUSE_YARP:BOOL=ON -DUSE_ICUB:BOOL=ON .. && \
    make -j$NJOBS install && \
    rm -rf /robots-io

# Build RobotsViz
RUN source /robotology-superbuild/build/install/share/robotology-superbuild/setup.sh && \
    git clone --progress https://github.com/xenvre/robots-viz && \
    cd robots-viz && mkdir build && cd build && \
    cmake -DUSE_YARP:BOOL=ON .. && \
    make -j$NJOBS install && \
    rm -rf /robots-viz

# Create user with passwordless sudo
RUN useradd -l -G sudo -md /home/user -s /bin/bash -p user user && \
    sed -i.bkp -e 's/%sudo\s\+ALL=(ALL\(:ALL\)\?)\s\+ALL/%sudo ALL=NOPASSWD:ALL/g' /etc/sudoers

# Switch to user
USER user

# Configure emacs
RUN echo "(setq-default indent-tabs-mode nil)" >> /home/user/.emacs.el && \
    echo "(setq-default tab-width 4)" >> /home/user/.emacs.el && \
    echo "(setq make-backup-files nil)" >> /home/user/.emacs.el && \
    echo "(setq auto-save-default nil)" >> /home/user/.emacs.el && \
    echo "(setq c-default-style \"linux\"" >> /home/user/.emacs.el && \
    echo "      c-basic-offset 4)" >> /home/user/.emacs.el && \
    echo "(global-subword-mode 1)" >> /home/user/.emacs.el && \
    echo "(add-hook 'before-save-hook 'delete-trailing-whitespace)" >> /home/user/.emacs.el && \
    echo "(custom-set-variables '(custom-enabled-themes '(tango-dark)))" >> /home/user/.emacs.el && \
    echo "(custom-set-faces)" >> /home/user/.emacs.elx

# Setup hyperpcr
WORKDIR /home/user
RUN source /robotology-superbuild/build/install/share/robotology-superbuild/setup.sh && \
    git clone --progress https://github.com/xenvre/hyperpcr && \
    cd ./hyperpcr && mkdir build && cd build && \
    cmake ../ && make -j$NJOBS && \
    cd /home/user/hyperpcr && virtualenv env && \
    . env/bin/activate && \
    pip install --upgrade pip && \
    pip install . && \
    # install cuDF and cuML
     pip uninstall --yes jupyter-client && \
     pip install cudf-cu11 dask-cudf-cu11 --extra-index-url=https://pypi.ngc.nvidia.com && \
     pip install cuml-cu11 --extra-index-url=https://pypi.ngc.nvidia.com && \
     pip uninstall --yes cupy-cuda115 && pip install cupy-cuda111 && \
    # \
     pip install --upgrade nvidia-tensorrt pycuda colored polygraphy --extra-index-url https://pypi.ngc.nvidia.com && \
    deactivate

# Configure robotology-superbuild
RUN echo "source /robotology-superbuild/build/install/share/robotology-superbuild/setup.sh" >> /home/user/.bashrc

# Several configurations
RUN mkdir -p /home/user/.local/share/yarp/contexts && \
    mkdir -p /home/user/.local/share/yarp/applications && \
    mkdir /home/user/.runtime && \
#
# exports
    echo "export YARP_RUNTIME_DIR=/home/user/.runtime" >> /home/user/.bashrc
#
# hyperpcr-viewer xml and config
#    cp /home/user/hyperpcr/yarp_sample/app/hyperpcr.xml /home/user/.local/share/yarp/applications/hyperpcr.xml && \
#    mkdir /home/user/.local/share/yarp/contexts/hyperpcr-viewer && \
#    cp /home/user/hyperpcr/src/viewer/config/config.ini.template /home/user/.local/share/yarp/contexts/hyperpcr-viewer/config.ini

# Launch bash from /home/user
WORKDIR /home/user
CMD ["bash"]