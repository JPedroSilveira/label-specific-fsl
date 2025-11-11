import hydra
from config.type import Config
from src.experiment import execute_experiment

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: Config) -> None:
    execute_experiment(config)

if __name__ == '__main__':
    main()