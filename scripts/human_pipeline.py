from ISBFSAR.main import ISBFSAR
from ISBFSAR.utils.params import MainConfig

if __name__ == "__main__":
    m = ISBFSAR(MainConfig())
    m.run()
