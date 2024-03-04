"""Config utilities for yml file."""
import collections
import functools
import os
import re
import yaml

class AttrDict(dict):
    pass

class Config(AttrDict):
    pass