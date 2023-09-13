from dataclasses import dataclass, field
import random
from typing import Dict
from enum import Enum


@dataclass
class Configuration:
    id: int
    conf: dict
    generator: Enum

@dataclass
class Parameter:
    name: str
    type: str
    bound: list
    default: int
    condition: list
    scale: str
    original_bound: list

@dataclass
class Tournament:
    id : int
    best_finisher: list
    worst_finisher: list
    configurations: list
    configuration_ids: list
    ray_object_store: dict
    instance_set: list
    instance_set_id : int

class ParamType(Enum):
    categorical = 1
    continuous = 2
    integer = 3

class TaskType(Enum):
    target_algorithm = 1
    monitor = 2

class Generator(Enum):
    default = 1
    random = 2
    var_graph = 3
    lhc = 4
    smac = 5
    cppl = 6

class Status(Enum):
    win = 1
    cap = 2
    timeout = 3
    stop = 4
    running = 5

class Surrogates(Enum):
    SMAC = 1
    GGA = 2
    CPPL = 3
