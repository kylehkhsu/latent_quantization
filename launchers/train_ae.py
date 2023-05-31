import hydra
import ipdb
import omegaconf
import pprint
import os
import sys
import pathlib
sys.path.insert(1, pathlib.Path(__file__).parent.parent.absolute().as_posix())


@hydra.main(version_base=None, config_path='/iris/u/kylehsu/code/disentangle/configs', config_name='train_ae')
def main(config):
    from scripts import train_ae
    print(os.environ['LD_LIBRARY_PATH'])
    pprint.pprint(config)
    train_ae.main(config)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        main()  # data processing might error out due to multiple jobs doing the same thing
        print(e)

