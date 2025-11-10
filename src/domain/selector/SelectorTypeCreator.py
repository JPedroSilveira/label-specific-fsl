from config.type import Config


class SelectorTypeCreator():
    @staticmethod
    def execute(config: Config) -> list:
        selectors = set([selector.model for selector in config.selectors])
        selectors_class = []
        for selector in selectors:
            module = __import__('src.domain.selector.types.' + selector, fromlist=[selector])
            selector_class = getattr(module, selector)
            selectors_class.append(selector_class)
        return selectors_class