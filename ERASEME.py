from utils.input import RealSense
import pyrealsense2 as rs
import tqdm


camera = RealSense(color_format=rs.format.rgb8, fps=60)

for _ in tqdm.tqdm(range(10000)):
    ret, img = camera.read()

