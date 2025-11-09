from typing import List


def add_on_dict_list(dict: dict[any, List[any]], key: any, value: any):
    if key in dict:
        dict[key].append(value)
    else:
        dict[key] = [value]