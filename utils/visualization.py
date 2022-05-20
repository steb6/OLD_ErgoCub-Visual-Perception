import numpy as np
from vispy import scene, app
from vispy.scene import ViewBox, Markers, Grid


def draw_geometries(geometries):
    canvas = scene.SceneCanvas(keys='interactive')
    canvas.size = 1200, 600
    canvas.show()

    grid = canvas.central_widget.add_grid()
    vb = grid.add_view(row=0, col=0)

    vb.camera = scene.TurntableCamera(elevation=0, azimuth=0, distance=0.5)
    vb.border_color = (0.5, 0.5, 0.5, 1)

    for gm in geometries:
        vb.add(gm)

    app.run()

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