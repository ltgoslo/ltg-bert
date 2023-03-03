mport sys
import json
import math
import copy


# Configuration class to store the configuration of a `BertModel`.
class BertConfig:
    def __init__(self, config_json_file_or_dict):
        if isinstance(config_json_file_or_dict, dict):
            config_dict = config_json_file_or_dict
        else:
            with open(config_json_file_or_dict, "r", encoding='utf-8') as reader:
                config_dict = json.loads(reader.read())

        for key, value in config_dict.items():
            self.__dict__[key] = value

        self.initializer_range = math.sqrt(2.0 / (5.0 * self.hidden_size))

    # Constructs a `BertConfig` from a Python dictionary of parameters.
    @classmethod
    def from_dict(cls, json_object):
        return BertConfig(json_object)

    # Constructs a `BertConfig` from a json file of parameters.
    @classmethod
    def from_json_file(cls, json_file):
        return BertConfig(json_file)

    def __repr__(self):
        return str(self.to_json_string())

    # Serializes this instance to a Python dictionary.
    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    # Serializes this instance to a JSON string.
    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + '\n'

    def clone(self):
        return BertConfig.from_dict(self.to_dict())
