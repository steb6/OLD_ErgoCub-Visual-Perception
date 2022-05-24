import numpy as np
from scipy.spatial.transform import Rotation
from vispy import scene, app
from vispy.scene import ViewBox, Markers, Grid
from vispy.visuals.transforms import MatrixTransform


def draw_geometries(geometries):
    canvas = scene.SceneCanvas(keys='interactive')
    canvas.size = 640, 480
    canvas.show()

    grid = canvas.central_widget.add_grid()
    vb = grid.add_view(row=0, col=0)

    vb.camera = scene.TurntableCamera(elevation=0, azimuth=0, roll=0, distance=0.5)
    # vb.camera.orbit(180, -90)

    vb.border_color = (0.5, 0.5, 0.5, 1)

    R = Rotation.from_euler('xyz', [-90, 0, 0], degrees=True).as_matrix()
    for gm in geometries:
        if isinstance(gm, Markers):
            gm._data['a_position'] = gm._data['a_position']
        if isinstance(gm, (scene.XYZAxis, scene.Line)):
            gm._pos = gm._pos

        vb.add(gm)

    app.run()



def draw_geometries_img(geometries):
    canvas = scene.SceneCanvas(keys='interactive')
    canvas.size = 640, 480
    canvas.show()

    grid = canvas.central_widget.add_grid()
    vb = grid.add_view(row=0, col=0)

    vb.camera = scene.TurntableCamera(elevation=0, azimuth=0, roll=0, distance=0.5)
    # vb.camera.orbit(180, -90)

    vb.border_color = (1, 1, 1, 0)
    vb.bgcolor = (1, 1, 1, 0)

    R = Rotation.from_euler('xyz', [-90, 0, 0], degrees=True).as_matrix()
    for gm in geometries:
        if isinstance(gm, Markers):
            gm._data['a_position'] = gm._data['a_position'] @ R
        if isinstance(gm, (scene.XYZAxis, scene.Line)):
            gm._pos = gm._pos @ R

        vb.add(gm)

    return canvas.render()

if __name__ == '__main__':
    scatter3 = Markers()
    scatter3.set_data(np.random.rand(1000, 3), face_color=(1, 0, 0, .5))

    scatter4 = Markers()
    scatter4.set_data(np.random.rand(1000, 3))

    # r_hand = scene.XYZAxis(width=10)
    # r_hand.set_data()
    #
    # l_hand = scene.XYZAxis()
    # l_hand.set_data()

    axis = scene.XYZAxis(width=10)
    axis.set_data()

    draw_geometries([scatter3, scatter4, axis])