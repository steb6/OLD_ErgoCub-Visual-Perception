import time
from multiprocessing import Queue, Process
from multiprocessing.managers import BaseManager

import cv2
from vispy import app, scene, visuals
from vispy.scene import ViewBox
from vispy.scene.visuals import Text, Image, Markers
import numpy as np
import math
import sys
from loguru import logger
from scipy.spatial.transform import Rotation

from grasping.modules.utils.misc import draw_mask
from human.modules.focus import FocusDetector
from human.utils.params import RealSenseIntrinsics

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
        self.builders = {}
        self.cameras = {}
        self.last_widget = None
        self.fps_s = []
        self.last_time = 0

    def run(self):
        self.build_gui()
        app.run()

    def remove_all_widgets(self):
        for wg, _ in self.widgets:
            self.grid.remove_widget(wg)
            del wg

    def add_all_widgets(self, name='', args=None):
        for build in self.builders:
            if build == name:
                self.builders[build](**args)
            else:
                self.builders[build]()

    def highlight(self, event):
        if event.type == 'mouse_press' and event.button == 2:
            self.canvas.central_widget.children[0].parent = None
            self.grid = self.canvas.central_widget.add_grid()
            self.add_all_widgets(event.visual.name, {'row': 0, 'col': 1, 'row_span': 4, 'col_span': 2})

            # if self.last_widget is None:
            #     self.last_widget = event.visual
            # else:
            #     self.last_widget.camera = self.center.camera

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
        self.canvas = canvas
        self.input_text = '>'

        ######################
        ##### View Boxes #####
        ######################
        ######################
        ###### Grasping ######
        ######################
        # Mask Visualization 1
        def build_mask1(row=0, col=0, row_span=1, col_span=1):
            b1 = ViewBox(name='mask1')
            self.grid.add_widget(b1, row=row, col=col, row_span=row_span, col_span=col_span)

            if 'mask1' not in self.cameras:
                self.cameras['mask1'] = scene.PanZoomCamera(rect=(0, 0, 640, 480), name='mask1')

            b1.camera = self.cameras['mask1']
            b1.camera.interactive = False
            b1.border_color = (0.5, 0.5, 0.5, 1)
            self.image1 = Image()
            b1.add(self.image1)
            b1.events.mouse_press.connect(self.highlight)
            self.widgets.append([b1, {'row': 0, 'col': 0}])

        self.builders['mask1'] = build_mask1

        # Mask Visualization 2
        def build_mask2(row=1, col=0, row_span=1, col_span=1):
            b2 = ViewBox(name='mask2')
            self.grid.add_widget(b2, row=row, col=col, row_span=row_span, col_span=col_span)

            if 'mask2' not in self.cameras:
                self.cameras['mask2'] = scene.PanZoomCamera(rect=(0, 0, 640, 480), name='mask2')

            b2.camera = self.cameras['mask2']
            b2.camera.interactive = False
            b2.border_color = (0.5, 0.5, 0.5, 1)
            self.image2 = Image()
            b2.add(self.image2)
            b2.events.mouse_press.connect(self.highlight)
            self.widgets.append([b2, {'row': 1, 'col': 0}])

        self.builders['mask2'] = build_mask2

        # Point Cloud 1
        def build_pc1(row=2, col=0, row_span=1, col_span=1):
            b3 = ViewBox(name='pc1')
            b3 = self.grid.add_widget(b3, row=row, col=col, row_span=row_span, col_span=col_span)

            if 'pc1' not in self.cameras:
                self.cameras['pc1'] = scene.TurntableCamera(elevation=0, azimuth=0, distance=1, name='pc1')

            b3.camera = self.cameras['pc1']
            b3.border_color = (0.5, 0.5, 0.5, 1)
            self.scatter1 = Markers(parent=b3.scene)
            self.scatter2 = Markers(parent=b3.scene)
            b3.events.mouse_press.connect(self.highlight)
            self.widgets.append([b3, {'row': 2, 'col': 0}])

        self.builders['pc1'] = build_pc1

        # Point Cloud 2
        def build_pc2(row=3, col=0, row_span=1, col_span=1):
            b4 = ViewBox(name='pc2')
            b4 = self.grid.add_widget(b4, row=row, col=col, row_span=row_span, col_span=col_span)

            if 'pc2' not in self.cameras:
                self.cameras['pc2'] = scene.TurntableCamera(elevation=0, azimuth=0, distance=0.5, name='pc2')

            b4.camera = self.cameras['pc2']
            b4.border_color = (0.5, 0.5, 0.5, 1)
            self.scatter3 = Markers(parent=b4.scene)
            self.scatter4 = Markers(parent=b4.scene)
            self.r_hand = scene.XYZAxis(parent=b4.scene)
            self.l_hand = scene.XYZAxis(parent=b4.scene)
            axis = scene.XYZAxis(parent=b4.scene)
            b4.events.mouse_press.connect(self.highlight)
            self.widgets.append([b4, {'row': 3, 'col': 0}])
            self.test = b4

        self.builders['pc2'] = build_pc2

        ######################
        ####### Human ########
        ######################

        def human_1(row=0, col=3, row_span=1, col_span=1):
            b5 = ViewBox(name='human_1')
            self.grid.add_widget(b5, row=row, col=col, row_span=row_span, col_span=col_span)
            b5.border_color = (0.5, 0.5, 0.5, 1)
            b5.camera = scene.TurntableCamera(elevation=-90, azimuth=0, distance=0)
            self.lines = []
            Plot3D = scene.visuals.create_visual_node(visuals.LinePlotVisual)
            # Skeleton
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
            # Box  # TODO ADD BOX
            self.box = Markers(parent=b5.scene)
            # Rest
            # coords = scene.visuals.GridLines(parent=b5.scene)
            axis = scene.visuals.XYZAxis(parent=b5.scene)
            b5.events.mouse_press.connect(self.highlight)
            self.widgets.append([b5, {'row': 0, 'col': 3}])

        self.builders['human_1'] = human_1

        # Info
        def human_2(row=1, col=3, row_span=1, col_span=1):
            self.b6 = ViewBox(name='human_2')
            self.grid.add_widget(self.b6, row=row, col=col, row_span=row_span, col_span=col_span)
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
            self.action = Text('', rotation=0, anchor_x="center", anchor_y="center", font_size=12)
            self.b6.add(self.action)
            self.b6.events.mouse_press.connect(self.highlight)
            self.widgets.append([self.b6, {'row': 1, 'col': 3}])

        self.builders['human_2'] = human_2

        # Image
        def human_3(row=2, col=3, row_span=1, col_span=1):
            b7 = ViewBox(name='human_3')
            self.grid.add_widget(b7, row=row, col=col, row_span=row_span, col_span=col_span)
            b7.camera = scene.PanZoomCamera(rect=(0, 0, 640, 480))
            b7.camera.interactive = False
            b7.border_color = (0.5, 0.5, 0.5, 1)
            self.image = Image()
            b7.add(self.image)
            b7.events.mouse_press.connect(self.highlight)
            self.widgets.append([b7, {'row': 2, 'col': 3}])

        self.builders['human_3'] = human_3

        def speed(row=3, col=3, row_span=1, col_span=1):
            self.b8 = ViewBox(name='speed')
            self.grid.add_widget(self.b8, row=row, col=col, row_span=row_span, col_span=col_span)
            self.b8.camera = scene.PanZoomCamera(rect=(0, 0, 1, 1))
            self.b8.camera.interactive = False
            self.b8.border_color = (0.5, 0.5, 0.5, 1)

            self.avg_fps = Text('', color='white', rotation=0, anchor_x="center", anchor_y="bottom",
                                font_size=12, pos=(0.5, 0.95))

            self.b8.add(self.avg_fps)
            self.b8.events.mouse_press.connect(self.highlight)
            self.widgets.append([self.b8, {'row': 1, 'col': 3}])

        self.builders['speed'] = speed

        self.add_all_widgets()

        logger.debug('Gui built successfully')

    def on_timer(self, _):

        if not self.show:
            return

        ##################
        #### Grasping ####
        ##################
        if not self.grasping_in.empty() and False:
            data = self.grasping_in.get()

            rgb_mask = draw_mask(data['rgb'], data['mask'])

            depth_image = cv2.applyColorMap(cv2.convertScaleAbs(data['depth'], alpha=0.03), cv2.COLORMAP_JET)
            depth_mask = draw_mask(depth_image, data['mask'])

            font = cv2.FONT_ITALIC
            bottomLeftCornerOfText = (10, 30)
            fontScale = 1
            fontColor = (255, 255, 255)
            thickness = 1
            lineType = 2

            cv2.putText(depth_mask, 'Distance: {:.2f}'.format(data["distance"] / 1000),
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)

            R = Rotation.from_euler('xyz', [-90, 0, 0], degrees=True).as_matrix()
            self.image1.set_data(rgb_mask[::-1, ..., ::-1])
            self.image2.set_data(depth_mask[::-1, ...])
            self.scatter1.set_data((data['partial'] @ R) * np.array([1, -1, 1]), edge_color='orange',
                                   face_color='orange', size=5)
            self.scatter2.set_data((data['reconstruction'] @ R) * np.array([1, -1, 1]), edge_color='blue',
                                   face_color='blue', size=5)

            self.scatter3.set_data((data['scene'][..., :3]) @ R, edge_color=data['scene'][..., 3:],
                                   face_color=data['scene'][..., 3:])
            self.scatter4.set_data(
                ((data['reconstruction']) * (data['var'] * 2) + data['mean'] * np.array([1, 1, -1])) @ R * np.array(
                    [1, -1, 1]), edge_color='blue', face_color='blue', size=5)
            if data['poses'] is not None:
                p = data['poses']
                pos = np.tile(p[0], [6, 1])
                a = np.eye(3) @ p[1]
                pos[1] = a[0]
                pos[3] = a[1]
                pos[5] = a[2]
                self.r_hand.set_data(pos)

            text = '\n'.join([f'{key}: {value:.2f} fps' for (key, value) in data['fps'].items()])
            self.avg_fps.text = text

        ##################
        ##### Human ######
        ##################
        if not self.human_in.empty():
            data = self.human_in.get()

            if not data:
                return

            edges = data["edges"]
            pose = data["pose"]
            img = data["img"]
            focus = data["focus"]
            action = data["action"]
            distance = data["distance"]
            human_bbox = data["human_bbox"]
            face = data["face"]
            box_center_3d = data["box_center"]
            pose2d = data["pose2d"]

            # POSE
            if pose is not None:
                for i, edge in enumerate(edges):
                    self.lines[i].set_data((pose[[edge[0], edge[1]]]), color='purple')
            else:
                for i in list(range(len(self.lines))):
                    self.lines[i].set_data(color='grey')

            # BOX
            box_center_2d = None
            if box_center_3d is not None and np.any(box_center_3d):
                # Draw box with human
                self.box.set_data(box_center_3d, edge_color='orange', face_color='orange', size=20)
                # Draw projection of box
                box_center = box_center_3d
                box_center_2d = RealSenseIntrinsics().K @ box_center.T
                box_center_2d = box_center_2d[0:2] / box_center_2d[2, :]
                box_center_2d = np.round(box_center_2d, 0).astype(int).squeeze()
            else:
                self.box.set_data(np.array([0, 0, 0])[None, ...])

            # IMAGE
            if img is not None:
                if human_bbox is not None:
                    x1, x2, y1, y2 = human_bbox
                    img = cv2.rectangle(img,
                                        (x1, y1), (x2, y2), (0, 0, 255), 1).astype(np.uint8)
                if face is not None:
                    x1, y1, x2, y2 = face.bbox.reshape(-1)
                    img = cv2.rectangle(img,
                                        (x1, y1), (x2, y2), (255, 0, 0), 1).astype(np.uint8)
                if box_center_2d is not None:
                    img = cv2.circle(img, box_center_2d, 5, (0, 255, 0)).astype(np.uint8)
                if pose2d is not None:
                    for edge in edges:
                        c1 = 0 < pose2d[edge[0]][0] < 640 and 0 < pose2d[edge[0]][1] < 480
                        c2 = 0 < pose2d[edge[1]][0] < 640 and 0 < pose2d[edge[1]][1] < 480
                        if c1 and c2:
                            img = cv2.line(img, pose2d[edge[0]], pose2d[edge[1]], (255, 0, 255), 3, cv2.LINE_AA)
                img = cv2.flip(img, 0)
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
            self.fps_s.append(1 / (time.time() - self.last_time))
            self.last_time = time.time()
            fps_s = self.fps_s[-10:]
            fps = sum(fps_s) / len(fps_s)
            if fps is not None:
                self.fps.text = "FPS: {:.2f}".format(fps)

            # Distance
            if distance is not None:
                self.distance.text = "DIST: {:.2f}m".format(distance)

            # Actions
            self.action.text = str(action)
            self.action.pos = 0.5, 0.5
            self.action.color = "green"

    def on_draw(self, event):
        pass


if __name__ == '__main__':
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
                         "action": {},
                         "distance": 0,  # TODO fix
                         "box": [1, 2, 3, 4]
                         }]
            human_in.put(elements)


    viz = Visualizer()
    viz.run()

    # # Testing
    # BaseManager.register('get_queue')
    # manager = BaseManager(address=('localhost', 50000), authkey=b'abracadabra')
    # manager.connect()
    #
    # queue = manager.get_queue('vis_in_grasping')
    # while True:
    #     data = {'rgb': np.random.randint(0, 255, [480, 640, 3], dtype=np.uint8), 'depth': np.random.rand(480, 640, 3),
    #             'mask': np.random.randint(0, 1, [480, 640], dtype=np.uint8), 'distance': 1.5,
    #             'mean': np.array([0, 0, 0]), 'var': np.array([1, 1, 1]), 'partial': np.random.rand(2048, 3),
    #             'reconstruction': np.random.rand(2048, 3), 'scene': np.random.rand(2048, 6), 'poses': None, 'fps': {'test': 1}}
    #
    #     queue.put(data)
