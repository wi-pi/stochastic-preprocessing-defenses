import inspect


class Registry(type):
    entries: dict

    def __getitem__(self, item):
        return self.entries[item]

    def __call__(cls, entry):
        cls.entries[entry.__name__] = entry

    def __repr__(cls):
        format_string = cls.__name__ + ': {\n'
        for k, v in cls.entries.items():
            format_string += f"    '{k}': {v.__module__}.{v.__name__}\n"
        format_string += '}'
        return format_string

    def keys(cls):
        return cls.entries.keys()

    def values(cls):
        return cls.entries.values()

    def items(cls):
        return cls.entries.items()

    def __iter__(cls):
        return iter(cls.entries)


def get_params(start: int = 1):
    frame = inspect.currentframe().f_back
    args, _, _, values = inspect.getargvalues(frame)
    params = {k: values[k] for k in args[start:]}
    return params
