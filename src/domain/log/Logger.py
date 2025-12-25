import datetime

from config.type import Config


class Logger:
    _config: Config = None
    
    @classmethod
    def setup(cls, config: Config) -> None:
        cls._config = config
    
    @classmethod
    def execute(cls, message: str, end="\n") -> None:
        now = datetime.datetime.now()
        time_str = now.strftime("%Y-%m-%d %H:%M:%S.%f")
        final_message = f'{time_str}: {message}'
        print(final_message, end=end)
        with open(cls._config.output.execution_output.log_file, "a") as f:
            f.write(f"{final_message}{end}")