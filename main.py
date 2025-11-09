import hydra
from src.config.config import Config
from src.experiment import execute_experiment
from src.util.test_gpu import test_gpu

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: Config) -> None:
    test_gpu()
    execute_experiment(config)

if __name__ == '__main__':
    main()