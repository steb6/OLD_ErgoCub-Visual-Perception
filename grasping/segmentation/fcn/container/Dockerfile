FROM nvcr.io/nvidia/tensorrt:22.03-py3

ENV CONDA_VERSION "py38_4.11.0"
ENV CONDA_MD5 718259965f234088d785cad1fbd7de03


ENV PYTHONDONTWRITEBYTECODE=true

# Install cv2 dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg libsm6 libxext6

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    numpy \
#TODO Remove
    opencv-python \
    open3d

RUN addgroup grasping \
    && useradd -ms /bin/bash grasping -g grasping

USER grasping:grasping

COPY ./ /home/grasping/segmentation
WORKDIR /home/grasping/segmentation

ENV PYTHONPATH /home/grasping/segmentation

CMD ["python", "src/main.py"]