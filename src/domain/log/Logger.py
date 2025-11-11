import datetime

from config.type import Config


class Logger:
    _config: Config = None
    
    @staticmethod
    def setup(config: Config) -> None:
        Logger._config = config
    
    @staticmethod
    def execute(message: str, end="\n") -> None:
        now = datetime.datetime.now()
        time_str = now.strftime("%Y-%m-%d %H:%M:%S.%f")
        final_message = f'{time_str}: {message}'
        print(final_message, end=end)
        with open(Logger._config.output.execution_output.log_file, "a") as f:
            f.write(f"{final_message}{end}")