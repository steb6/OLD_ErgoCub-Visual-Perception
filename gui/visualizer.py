from multiprocessing import Queue, Process
from multiprocessing.managers import BaseManager

import cv2
from vispy import app, scene, visuals
from vispy.scene.visuals import Text, Image, Markers
import numpy as np
import math
import sys
from loguru import logger
from scipy.spatial.transform import Rotation

from grasping.modules.utils.misc import draw_mask

logger.remove()
logger.add(sys.stdout,
           format="<fg #b28774>{time:YYYY-MM-DD HH:mm:ss:SSS ZZ}</> <yellow>|</>"
                  " <lvl>{level: <8}</> "
                  "<yellow>|</> <blue>{process.name: ^12} {file} {line}</> <yellow>-</> <lvl>{message}</>",
           diagnose=True)

logger.level('INFO', color='<fg #fef5ed>')
logger.level('SUCCESS', color='<fg #79d70f>')
logger.level('WARNING', color='<fg #fd811e>')
logger.level('ERROR', color='<fg #ed254e>')


class Visualizer(Process):

    def __init__(self):
        super().__init__()
        self.name = 'Visualizer'
        logger.info('Connecting to manager...')

        BaseManager.register('get_queue')
        manager = BaseManager(address=('localhost', 50000), authkey=b'abracadabra')
        manager.connect()

        logger.success('Connected to manager.')

        self.human_in = manager.get_queue('vis_in_human')
        self.human_out = manager.get_queue('vis_out_human')
        self.grasping_in = manager.get_queue('vis_in_grasping')

        self.widgets = []

    def run(self):
        self.build_gui()
        app.run()

    def remove_all_widgets(self):
        for wg in self.widgets:
            self.grid.remove_widget(wg)

    def add_all_widgets(self):
        cols = [0] * 4 + [3] * 4
        rows = list(range(4)) + list(range(4))
        for wg, r, c in zip(self.widgets, rows, cols):
            self.grid.add_widget(wg, row=r, col=c)

    def highlight(self, event):
        if event.type == 'mouse_press' and event.button == 2:

            self.remove_all_widgets()
            self.add_all_widgets()

            self.grid.add_widget(event.visual, row=0, col=1, row_span=4, col_span=2)

            # self.center.camera = event.visual.camera
            # self.center = event.visual


    def build_gui(self):
        logger.debug('Started gui building')
        self.show = True

        self._timer = app.Timer('auto', connect=self.on_timer, start=True)

        canvas = scene.SceneCanvas(keys='interactive')
        canvas.size = 1200, 600
        canvas.show()

        # This is the top-level widget that will hold three ViewBoxes, which will
        # be automatically resized whenever the grid is resized.
        self.grid = canvas.central_widget.add_grid()
        self.input_text = '>'

        ######################
        ##### View Boxes #####
        ######################
                ######################
                ###### Grasping ######
                ######################
        # Mask Visualization 1
        b1 = self.grid.add_view(row=0, col=0)
        b1.camera = scene.PanZoomCamera(rect=(0, 0, 640, 480))
        b1.camera.interactive = False
        b1.border_color = (0.5, 0.5, 0.5, 1)
        self.image1 = Image()
        b1.add(self.image1)
        b1.events.mouse_press.connect(self.highlight)
        self.widgets.append(b1)

        # Mask Visualization 2
        b2 = self.grid.add_view(row=1, col=0)
        b2.camera = scene.PanZoomCamera(rect=(0, 0, 640, 480))
        b2.camera.interactive = False
        b2.border_color = (0.5, 0.5, 0.5, 1)
        self.image2 = Image()
        b2.add(self.image2)
        b2.events.mouse_press.connect(self.highlight)
        self.widgets.append(b2)

        # Point Cloud 1
        b3 = self.grid.add_view(row=2, col=0)
        b3.camera = scene.TurntableCamera(elevation=0, azimuth=0, distance=1)
        b3.border_color = (0.5, 0.5, 0.5, 1)
        self.scatter1 = Markers(parent=b3.scene)
        self.scatter2 = Markers(parent=b3.scene)
        b3.events.mouse_press.connect(self.highlight)
        self.widgets.append(b3)

        # Point Cloud 2
        b4 = self.grid.add_view(row=3, col=0)
        b4.camera = scene.TurntableCamera(elevation=0, azimuth=0, distance=0.5)
        b4.border_color = (0.5, 0.5, 0.5, 1)
        self.scatter3 = Markers(parent=b4.scene)
        axis = scene.XYZAxis(parent=b4.scene)
        b4.events.mouse_press.connect(self.highlight)
        self.widgets.append(b4)
                ######################
                ####### Human ########
                ######################

        b5 = self.grid.add_view(row=0, col=3)
        b5.border_color = (0.5, 0.5, 0.5, 1)
        b5.camera = scene.TurntableCamera(45, elevation=30, azimuth=0, distance=2)
        self.lines = []
        Plot3D = scene.visuals.create_visual_node(visuals.LinePlotVisual)
        for _ in range(30):
            self.lines.append(Plot3D(
                [],
                width=3.0,
                color="purple",
                edge_color="w",
                symbol="o",
                face_color=(0.2, 0.2, 1, 0.8),
                marker_size=1,
            ))
            b5.add(self.lines[_])
        coords = scene.visuals.GridLines(parent=b5.scene)
        b5.events.mouse_press.connect(self.highlight)
        self.widgets.append(b5)

        # Info
        self.b6 = self.grid.add_view(row=1, col=3)
        self.b6.camera = scene.PanZoomCamera(rect=(0, 0, 1, 1))
        self.b6.camera.interactive = False
        self.b6.border_color = (0.5, 0.5, 0.5, 1)
        self.distance = Text('', color='white', rotation=0, anchor_x="center", anchor_y="bottom",
                             font_size=12, pos=(0.25, 0.9))
        self.b6.add(self.distance)
        self.focus = Text('', color='green', rotation=0, anchor_x="center", anchor_y="bottom",
                          font_size=12, pos=(0.5, 0.9))
        self.b6.add(self.focus)
        self.fps = Text('', color='white', rotation=0, anchor_x="center", anchor_y="bottom",
                        font_size=12, pos=(0.75, 0.9))
        self.b6.add(self.fps)
        self.actions = {}
        self.b6.events.mouse_press.connect(self.highlight)
        self.widgets.append(self.b6)

        # Image
        b7 = self.grid.add_view(row=2, col=3)
        b7.camera = scene.PanZoomCamera(rect=(0, 0, 640, 480))
        b7.camera.interactive = False
        b7.border_color = (0.5, 0.5, 0.5, 1)
        self.image = Image()
        b7.add(self.image)
        b7.events.mouse_press.connect(self.highlight)
        self.widgets.append(b7)

        # Commands
        b8 = self.grid.add_view(row=3, col=3)
        b8.camera = scene.PanZoomCamera(rect=(0, 0, 1, 1))
        b8.camera.interactive = False
        b8.border_color = (0.5, 0.5, 0.5, 1)
        self.desc_add = Text('ADD ACTION: add action_name [-focus][-box/nobox]', color='white', rotation=0,
                             anchor_x="left",
                             anchor_y="bottom",
                             font_size=12, pos=(0.1, 0.9))
        self.desc_remove = Text('REMOVE ACTION: remove action_name', color='white', rotation=0, anchor_x="left",
                                anchor_y="bottom",
                                font_size=12, pos=(0.1, 0.7))
        self.input_string = Text(self.input_text, color='purple', rotation=0, anchor_x="left", anchor_y="bottom",
                                 font_size=12, pos=(0.1, 0.5))
        self.log = Text('', color='orange', rotation=0, anchor_x="left", anchor_y="bottom",
                        font_size=12, pos=(0.1, 0.3))
        b8.add(self.desc_add)
        b8.add(self.desc_remove)
        b8.add(self.input_string)
        b8.add(self.log)
        b8.events.mouse_press.connect(self.highlight)
        self.widgets.append(b8)

        self.center = self.grid.add_view(row=0, col=1, row_span=4, col_span=2)
        logger.debug('Gui built successfully')

    def printer(self, x):
        if x.text == '\b':
            if len(self.input_text) > 1:
                self.input_text = self.input_text[:-1]
            self.log.text = ''
        elif x.text == '\r':
            self.human_out.put(self.input_text[1:])  # Do not send '<'
            self.input_text = '>'
            self.log.text = ''
        elif x.text == '\\':
            self.show = not self.show
        else:
            self.input_text += x.text
        self.input_string.text = self.input_text

    def on_timer(self, _):

        if not self.show:
            return

        theta = 90
        R = np.array([[1, 0, 0],
                       [0, math.cos(theta), -math.sin(theta)],
                       [0, math.sin(theta), math.cos(theta)]])

        ##################
        #### Grasping ####
        ##################
        if self.grasping_in.empty():
            return
        data = self.grasping_in.get()

        res1, res2 = draw_mask(data['rgb'], data['mask'])

        font = cv2.FONT_ITALIC
        bottomLeftCornerOfText = (10, 30)
        fontScale = 1
        fontColor = (255, 255, 255)
        thickness = 1
        lineType = 2

        cv2.putText(res2, 'Distance: {:.2f}'.format(data["distance"] / 1000),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)

        self.image1.set_data(res1[::-1, ..., ::-1])
        self.image2.set_data(res2[::-1, ..., ::-1])
        self.scatter1.set_data((data['partial'] @ Rotation.from_euler('xyz', [270, 0, 0], degrees=True).as_matrix()) * np.array([1, -1, 1]), edge_color='orange', face_color='orange', size=5)
        self.scatter2.set_data((data['reconstruction'] @ Rotation.from_euler('xyz', [270, 0, 0], degrees=True).as_matrix()) * np.array([1, -1, 1]), edge_color='blue', face_color='blue', size=5)
        self.scatter3.set_data(((data['scene'][..., :3] - data['mean']) @ Rotation.from_euler('xyz', [270, 0, 0], degrees=True).as_matrix()), edge_color=data['scene'][..., 3:], face_color=data['scene'][..., 3:])


        ##################
        ##### Human ######
        ##################
        if self.human_in.empty():
            return
        data = self.human_in.get()

        if not data:
            return

        if "log" in data:
            self.log.text = data["log"]
        else:
            edges = data["edges"]
            pose = data["pose"]
            img = data["img"]
            focus = data["focus"]
            fps = data["fps"]
            results = data["actions"]
            distance = data["distance"]
            box = data["box"]

            # POSE
            if pose is not None:
                R = Rotation.from_euler('z', 90, degrees=True).as_matrix()
                pose = pose @ R
                for i, edge in enumerate(edges):
                    self.lines[i].set_data((pose[[edge[0], edge[1]]]))

            # IMAGE
            if img is not None:
                if box is not None:
                    x1, x2, y1, y2 = box
                    img = cv2.rectangle(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                        (x1, x2), (y1, y2), (255, 0, 0), 1).astype(np.uint8)
                self.image.set_data(img)

            # INFO
            if focus is not None:
                if focus:
                    self.focus.text = "FOCUS"
                    self.focus.color = "green"
                else:
                    self.focus.text = "NOT FOCUS"
                    self.focus.color = "red"

            # FPS
            if fps is not None:
                self.fps.text = "FPS: {:.2f}".format(fps)

            # Distance
            if distance is not None:
                self.distance.text = "DIST: {:.2f}m".format(distance)

            # Actions
            m = max([_[0] for _ in results.values()]) if len(results) > 0 else 0
            for i, r in enumerate(results.keys()):
                score, requires_focus, requires_box = results[r]
                # Check if conditions are satisfied
                if score == m:
                    c1 = True if not requires_focus else focus
                    c2 = True if (requires_box is None) else (box == requires_box)
                    if c1 and c2:
                        color = "green"
                    else:
                        color = "orange"
                else:
                    color = "red"
                if r in self.actions.keys():
                    text = "{}: {:.2f}".format(r, score)
                    if requires_focus:
                        text += ' (0_0)'
                    if requires_box:
                        text += ' [ ]'
                    if requires_box is not None and not requires_box:
                        text += ' [X]'
                    self.actions[r].text = text
                else:
                    self.actions[r] = Text('', rotation=0, anchor_x="center", anchor_y="bottom", font_size=12)
                    self.b6.add(self.actions[r])
                self.actions[r].pos = 0.5, 0.7 - (0.1 * i)
                self.actions[r].color = color

            # Remove erased action (if any)
            to_remove = []
            for key in self.actions.keys():
                if key not in results.keys():
                    to_remove.append(key)
            for key in to_remove:
                self.actions[key].parent = None
                self.actions.pop(key)

    def on_draw(self, event):
        pass


def grasping():
    BaseManager.register('get_queue')
    manager = BaseManager(address=('localhost', 50000), authkey=b'abracadabra')
    manager.connect()

    grasping_in = manager.get_queue('vis_in_grasping')

    while True:
        res1 = np.random.randint(0, 255, [480, 640, 3], dtype=np.uint8)
        res2 = np.random.randint(0, 255, [480, 640, 3], dtype=np.uint8)

        pc1 = np.random.rand(1000, 3)
        pc2 = np.random.rand(1000, 3)
        grasping_in.put({'res1': res1, 'res2': res2, 'pc1': pc1, 'pc2': pc2})


def human():
    BaseManager.register('get_queue')
    manager = BaseManager(address=('localhost', 50000), authkey=b'abracadabra')
    manager.connect()

    human_in = manager.get_queue('vis_in_human')
    human_out = manager.get_queue('vis_out_human')

    while True:
        elements = [{"img": np.random.random((640, 480, 3)),
                    "pose": np.random.random((30, 3)),
                    "edges": [(1, 2)],
                    "fps": 0,
                    "focus": False,
                    "actions": {},
                    "distance": 0,  # TODO fix
                    "box": [1, 2, 3, 4]
                    }]
        human_in.put(elements)

if __name__ == '__main__':

    viz = Visualizer()
    viz.run()