from src.domain.data.reader.Reader import Reader


class WeightReader(Reader):
    def __init__(self) -> None:
        super().__init__()
        
    @staticmethod
    def _get_file_suffix() -> str:
        return "_raw.csv"