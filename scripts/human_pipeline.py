from ISBFSAR.main import ISBFSAR
from configs.action_rec_config import MAIN

if __name__ == "__main__":
    m = ISBFSAR(**MAIN.Args.to_dict())
    m.run()
