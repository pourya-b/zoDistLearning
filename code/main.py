import os
from config import Parameter_config, set_seed
from Net_Trainer import Trainer


def main():
    path = os.path.dirname(os.path.abspath(__file__)) + '/'
    os.chdir(path)

    configer = Parameter_config()
    set_seed(configer.args)
    instance = Trainer(configer.args)
    instance.run(configer.args)

if __name__ == "__main__":
    main()
