from multiprocessing import Queue, Process

from vispy import app, scene, visuals
from vispy.scene.visuals import Text, Image, Markers
import numpy as np
import math


class VISPYVisualizer:


    @staticmethod
    def create_visualizer(qi):
        _ = VISPYVisualizer(qi)
        app.run()

    def __init__(self, input_queue):

        self.input_queue = input_queue
        self.show = True

        self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.input_text = '>'

        self.canvas = scene.SceneCanvas(keys='interactive')
        self.canvas.size = 1200, 600
        self.canvas.show()

        # This is the top-level widget that will hold three ViewBoxes, which will
        # be automatically resized whenever the grid is resized.
        grid = self.canvas.central_widget.add_grid()

        # Plot
        # Image
        b1 = grid.add_view(row=0, col=0)
        b1.camera = scene.PanZoomCamera(rect=(0, 0, 640, 480))
        b1.camera.interactive = False
        b1.border_color = (0.5, 0.5, 0.5, 1)
        self.image1 = Image()
        b1.add(self.image1)

        # Image
        b2 = grid.add_view(row=1, col=0)
        b2.camera = scene.PanZoomCamera(rect=(0, 0, 640, 480))
        b2.camera.interactive = False
        b2.border_color = (0.5, 0.5, 0.5, 1)
        self.image2 = Image()
        b2.add(self.image2)

        b3 = grid.add_view(row=0, col=1)
        b3.camera = 'turntable'
        b3.border_color = (0.5, 0.5, 0.5, 1)
        self.scatter1 = Markers()
        b3.add(self.scatter1)

        b4 = grid.add_view(row=1, col=1)
        b4.camera = 'turntable'
        b4.border_color = (0.5, 0.5, 0.5, 1)
        self.scatter2 = Markers()
        b4.add(self.scatter2)


    def on_timer(self, _):

        data = self.input_queue.get()


        # IMAGE
        self.image1.set_data(data['res1'])
        self.image2.set_data(data['res2'])
        self.scatter1.set_data(data['pc1'], edge_color=None, face_color=(1, 1, 1, .5), size=5)
        self.scatter2.set_data(data['pc2'], edge_color=None, face_color=(1, 1, 1, .5), size=5)

    def on_draw(self, event):
        pass

#
# if __name__ == '__main__':
#
#
#     input_queue = Queue(1)
#     output_queue = Queue(1)
#     output_proc = Process(target=VISPYVisualizer.create_visualizer,
#                                args=(output_queue,))
#     output_proc.start()
#
#     while True:
#         res1 = np.random.randint(0, 255, [480, 640, 3], dtype=np.uint8)
#         res2 = np.random.randint(0, 255, [480, 640, 3], dtype=np.uint8)
#
#         pc1 = np.random.rand(1000, 3)
#         pc2 = np.random.rand(1000, 3)
#         output_queue.put({'res1': res1, 'res2': res2, 'pc1': pc1, 'pc2': pc2})