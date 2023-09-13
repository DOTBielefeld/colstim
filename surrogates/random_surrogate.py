import copy
import sys
import os
from sklearn.preprocessing import OneHotEncoder
import time


sys.path.append(os.getcwd())


import uuid
import numpy as np
from conf_generetors import check_no_goods, check_conditionals, default_point, random_point,reset_conditionals, variable_graph_structure, graph_crossover

np.set_printoptions(threshold=sys.maxsize)



class RANDOM_CHOOSER:
    def __init__(self, scenario, seed, pool_size = 15):

        self.scenario = scenario
        self.features = features
        self.seed = seed

        self.pool_size = pool_size
        self.pool = [random_point(scenario, uuid.uuid4()) for _ in range(self.pool_size -1 )] + [default_point(scenario, uuid.uuid4())]
        self.feature_store = {}


    def get_suggestions(self, scenario, n_to_select, next_instance_set, instance_features=None):

        suggest = [random_point(scenario, uuid.uuid4()) for _ in range(n_to_select)]

        return suggest, []








