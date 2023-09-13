"""This module contains functions for the CPPL surrogate."""

import copy
import sys
import os

from sklearn.preprocessing import OneHotEncoder
import uuid
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pool import Configuration, Generator, ParamType

from conf_generetors import check_no_goods, check_conditionals, default_point, random_point,reset_conditionals, variable_graph_structure, graph_crossover


np.set_printoptions(threshold=sys.maxsize)
sys.path.append(os.getcwd())


class ISS:

    def __init__(self, scenario, seed, pool_size=15, eta=3.5, random_prob=0.2, mutation_prob=0.8, dp=0.2):

        self.scenario = scenario
        self.seed = seed
        self.dp = dp

        self.pool_size = pool_size
        self.pool = [random_point(scenario, uuid.uuid4())
                     for _ in range(self.pool_size - 1)] + \
                    [default_point(scenario, uuid.uuid4())]
        self.feature_store = {}

        self.eta = eta
        self.w_store = {}
        self.f_store = {}
        self.t = 0

        self.random_prob = random_prob
        self.mutation_prob = mutation_prob
        self.number_new_confs = 4


    def update_feature_store(self, conf, instance):
        """
        For a conf/instance pair compute the features and store them in a feature store for latere
        """

        if conf.id not in self.w_store:
            self.w_store[conf.id] = 0
        if conf.id not in self.f_store:
            self.f_store[conf.id] = 0


    def update_model_single_observation(self, winner_id, tried_confs):
        for tc in tried_confs:
            self.w_store[winner_id] = self.w_store[winner_id] + self.eta * 1
            self.f_store[tc] = self.f_store[tc] + self.eta * 1


    def select_from_set(self, conf_set, instance_set, n_to_select):
        """
        For a set of configurations and instances select the most promising configurations
        """

        v_hat = np.zeros((len(conf_set), len(instance_set)))

        for c in range(len(conf_set)):
            conf = conf_set[c]
            for next_instance in range(len(instance_set)):
                beta_sample = np.random.beta(self.w_store[conf.id] + 1, self.f_store[conf.id] + 1)
                v_hat[c][next_instance] = beta_sample

        v_hat_s = v_hat.sum(axis=1)

        quality = v_hat_s

        selection = (-quality).argsort()[:n_to_select]
        print(f"choosing {selection} {(-quality).argsort()} {[conf_set[i].id for i in selection]}")
        print(f" {[quality[i] for i in (-quality).argsort()]}, {list(quality)}")
        return [conf_set[i] for i in selection], [[v_hat_s[i]] for i in (-quality).argsort() ]

    def delete_from_pool(self, instance_set):
        """
        Based on the feedback delete poor performing configurations from the pool
        """

        if self.t > 0:

            v_hat = np.zeros((len(self.pool), len(instance_set)))

            for c in range(len(self.pool)):
                conf = self.pool[c]
                for next_instance in range(len(instance_set)):
                    beta_sample = np.random.beta(self.w_store[conf.id] + 1, self.f_store[conf.id] + 1)
                    v_hat[c][next_instance] = beta_sample

            v_hat_s = v_hat.sum(axis=1)

            quality = v_hat_s

            discard_index = (-quality).argsort()[len(quality)-int(len(quality)*self.dp):]
            print(f"dicsarding {discard_index} {[self.pool[i].id for i in discard_index]}")

            dis = []
            for i in sorted(discard_index, reverse=True):
                dis.append(self.pool[i])
                del self.pool[i]


    def create_new_conf(self, parent_one, parent_two):
        """
        Create new configurations based on the genetic procedure described
        """
        no_good = True
        while no_good:
            rn = np.random.uniform()

            if rn < self.random_prob:
                new_conf = random_point(self.scenario, uuid.uuid4())
                new_conf.generator = Generator.cppl
            else:
                graph_structure = variable_graph_structure(self.scenario)

                new_conf = graph_crossover(graph_structure, parent_one, parent_two, self.scenario)

                possible_mutations = random_point(self.scenario, uuid.uuid4())
                for param, value in new_conf.items():
                    rn = np.random.uniform()
                    if rn < self.mutation_prob:
                        new_conf[param] = possible_mutations.conf[param]

                identity = uuid.uuid4()

                new_conf = Configuration(identity,
                                         new_conf,
                                         Generator.cppl)

            cond_vio = check_conditionals(self.scenario, new_conf.conf)
            if cond_vio:
                new_conf.conf = reset_conditionals(self.scenario, new_conf.conf, cond_vio)

            no_good = check_no_goods(self.scenario, new_conf.conf)
        return new_conf

    def add_to_pool(self, past_instances):
        """
        Add the most promising newly created configurations to the pool
        """
        number_to_create = self.pool_size - len(self.pool)
        new_promising_conf = []

        if number_to_create > 0:
            if len(self.pool) > 1:
                best_to_confs, _ = self.select_from_set(self.pool, past_instances, 2)
                conf_one, conf_two = best_to_confs[0], best_to_confs[1]
            elif len(self.pool) == 1:
                conf_one = self.pool[0]
                conf_two = random_point(self.scenario, uuid.uuid4())
                conf_two.generator = Generator.cppl
            else:
                conf_one = random_point(self.scenario, uuid.uuid4())
                conf_one.generator = Generator.cppl
                conf_two = random_point(self.scenario, uuid.uuid4())
                conf_two.generator = Generator.cppl
            new_promising_conf = []
            for nc in range(number_to_create):
                new_promising_conf = new_promising_conf + [self.create_new_conf(conf_one, conf_two)]

            for instance in past_instances:
                for c in new_promising_conf:
                    self.update_feature_store(c, instance)

        self.pool = self.pool + new_promising_conf


    def update(self,results, previous_tournament, instance_features=None):
        """
        Updated the model with given feedback
        """

        confs_w_feedback = previous_tournament.best_finisher + previous_tournament.worst_finisher

        for instance in previous_tournament.instance_set:
            results_on_instance = {}

            for c in self.pool + confs_w_feedback:
                self.update_feature_store(c, instance)

            for c in previous_tournament.configuration_ids:
                if not np.isnan(results[c][instance]):
                    results_on_instance[c] = results[c][instance]
                else:
                    results_on_instance[c] = self.scenario.cutoff_time

            best_conf_on_instance = min(results_on_instance, key=results_on_instance.get)

            if results_on_instance[best_conf_on_instance] >= self.scenario.cutoff_time:
                self.pool = [random_point(self.scenario, uuid.uuid4()) for _ in range(self.pool_size -1 )] + [default_point(self.scenario, uuid.uuid4())]
                for c in self.pool:
                    [self.update_feature_store(c, i) for i in previous_tournament.instance_set]
                continue

            tried = previous_tournament.configuration_ids.copy()

            self.t = self.t + 1

            self.update_model_single_observation(best_conf_on_instance, tried)

        self.delete_from_pool(previous_tournament.instance_set)

        self.add_to_pool(previous_tournament.instance_set)


    def get_suggestions(self, scenario, n_to_select, next_instance_set, instance_features=None):
        """
        Suggest configurations to run next based on the instances that are comming
        """

        for instance in next_instance_set:
            for c in self.pool:
                self.update_feature_store(c, instance)

        suggest, ranking = self.select_from_set(self.pool, next_instance_set, n_to_select)

        return suggest, ranking

