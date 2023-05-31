import hydra
import ipdb
import omegaconf
import pprint
import os
import sys
import pathlib
sys.path.insert(1, pathlib.Path(__file__).parent.parent.absolute().as_posix())


@hydra.main(version_base=None, config_path='/iris/u/kylehsu/code/disentangle/configs', config_name='perturbations')
def main(config):
    from scripts import perturbations
    print(os.environ['LD_LIBRARY_PATH'])
    pprint.pprint(config)
    perturbations.main(config)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)

