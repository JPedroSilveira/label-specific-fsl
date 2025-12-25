from src.domain.data.reader.Reader import Reader


class RankingReader(Reader):        
    @staticmethod
    def _get_file_suffix() -> str:
        return "_sorted.csv"