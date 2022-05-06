FROM nvidia/cuda:11.4.2-runtime-ubuntu20.04

ENV CONDA_VERSION "py38_4.11.0"
ENV CONDA_MD5 718259965f234088d785cad1fbd7de03

# Tell Python not to recreate the bytecode files. Since this is a docker image,
# these will be recreated every time, writing them just uses unnecessary disk
# space.
ENV PYTHONDONTWRITEBYTECODE=true

RUN apt-get update && apt-get install -y --no-install-recommends wget bzip2 \
    && addgroup grasping \
    && useradd -ms /bin/bash grasping -g grasping \
    && wget --quiet https://repo.continuum.io/miniconda/Miniconda3-$CONDA_VERSION-Linux-x86_64.sh \
    && mv Miniconda3-$CONDA_VERSION-Linux-x86_64.sh miniconda.sh \
    && sh ./miniconda.sh -b -p /opt/conda \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> /home/grasping/.bashrc \
    && echo "conda activate base" >> /home/grasping/.bashrc \
    && find /opt/conda/ -follow -type f -name '*.a' -delete \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
    && /opt/conda/bin/conda clean -afy \
    && chown -R grasping:grasping /opt/conda \
    && apt-get --purge -y autoremove wget bzip2

RUN /opt/conda/bin/conda install --yes --freeze-installed -c rapidsai -c nvidia -c pytorch -c conda-forge\
        numpy \
        cuml=22.02 \
        pytorch torchvision \
        cudatoolkit=11.3 \
    && /opt/conda/bin/conda clean -afy \
    && find /opt/conda/ -follow -type f -name '*.a' -delete \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
    && find /opt/conda/ -follow -type f -name '*.js.map' -delete \
    && find /opt/conda/lib/python*/site-packages/bokeh/server/static -follow -type f -name '*.js' ! -name '*.min.js' -delete

# TO merge with the previous layer
RUN /opt/conda/bin/conda install --yes --freeze-installed -c conda-forge\
        fastapi \
        uvicorn \
        gunicorn \
        python-multipart \
    && /opt/conda/bin/conda clean -afy \
    && find /opt/conda/ -follow -type f -name '*.a' -delete \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
    && find /opt/conda/ -follow -type f -name '*.js.map' -delete \
    && find /opt/conda/lib/python*/site-packages/bokeh/server/static -follow -type f -name '*.js' ! -name '*.min.js' -delete

ENV PATH="/opt/conda/bin:/opt/conda/condabin:${PATH}"
USER grasping:grasping
WORKDIR /home/grasping

COPY ./ /home/grasping/denoising

CMD ["uvicorn", "denoising.src.main:app", "--host", "0.0.0.0", "--port", "8000"]