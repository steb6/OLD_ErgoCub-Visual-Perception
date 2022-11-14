from ISBFSAR.main import ISBFSAR
from ISBFSAR.utils.params import MainConfig

if __name__ == "__main__":
    master = ISBFSAR(MainConfig(), visualizer=True)
    master.load()
    master.run()
