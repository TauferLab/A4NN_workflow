from collections.abc import Iterable

def flatten_list(arr):
    """
    Return a flatten list from nested lists recursively
    """
    result = []

    if isinstance(arr, Iterable):
        for item in arr:
            result.extend(flatten_list(item))
    else:
        result.append(arr)

    return result