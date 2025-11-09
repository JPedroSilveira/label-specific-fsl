import datetime
from src.util.string_util import to_string_with_2_digits

def print_with_time(message: str, end="\n"):
    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    print(f'{time_str}: {message}', end=end)

def print_load_bar(progress, total, bar_length=50, info: str=None):
    percent = (progress / total) * 100
    filled_length = int(bar_length * percent / 100)
    bar = "=" * filled_length + " " * (bar_length - filled_length)
    percent_str = f"{percent:.2f}".ljust(6)
    text = f"\rProgress: {percent_str}% [{bar}]"
    if info is not None:
        text += f" - {info}"
    print(text, end="")
    if int(percent) == 100:
        print("\r")