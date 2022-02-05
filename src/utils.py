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

    def __iter__(cls):
        return iter(cls.entries)
