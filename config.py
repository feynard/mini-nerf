from types import SimpleNamespace

import yaml


class Config(SimpleNamespace):

    def __init__(self, *args, **kwargs):

        for k in kwargs:
            if isinstance(kwargs[k], dict):
                kwargs[k] = Config(**kwargs[k])

        super().__init__(*args, **kwargs)

    def __getitem__(self, k):
        return self.__dict__[k]

    def keys(self):
        return self.__dict__.keys()

    def __repr__(self):
        result = ""

        for i, (k, v) in enumerate(self.__dict__.items()):
            
            if isinstance(v, Config):
                
                result += k + ":" + "\n"
                lines = repr(v).split('\n')
                
                for j, line in enumerate(lines):
                    result += " " * 2 + line
                    result += "\n"
            else:
                result += k + ": " + str(v)
                if i != len(self.__dict__) - 1:
                    result += "\n"

        return result
    
    @classmethod
    def from_yaml(cls, yaml_file: str):
        with open(yaml_file, 'r') as f:
            return cls(**yaml.safe_load(f))