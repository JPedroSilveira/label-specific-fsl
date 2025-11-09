from config.type import Config


class SelectorTypeCreator():
    @staticmethod
    def execute(config: Config) -> list:
        selectors_class = []
        for selector in config.selectors:
            print(selector.model)
            module = __import__('src.domain.selector.types.' + selector.model, fromlist=[selector.model])
            selector_class = getattr(module, selector.model)
            selectors_class.append(selector_class)
        return selectors_class