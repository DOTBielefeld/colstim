import copy
import sys
import os
from sklearn.preprocessing import OneHotEncoder


sys.path.append(os.getcwd())
import uuid
import numpy as np

from pool import Configuration, Generator, ParamType
from conf_generetors import check_no_goods, check_conditionals, default_point, random_point,reset_conditionals, variable_graph_structure, graph_crossover

class COLSTIM_FS:
    def __init__(self, scenario, seed, pool_size=15,random_prob=0.1, mutation_prob=0.1, dp =0.2):

        self.dp = dp

        self.scenario = scenario
        self.seed = seed

        self.pool_size = pool_size
        self.pool = [random_point(scenario, uuid.uuid4()) for _ in range(self.pool_size -1 )] + [default_point(scenario, uuid.uuid4())]

        self.w_store = {}
        self.n_store = {}

        for conf in self.pool:
            self.w_store[conf.id] = {}
            self.n_store[conf.id] = {}
            for cc in self.pool:
                self.w_store[conf.id][cc.id] = 0
                self.n_store[conf.id][cc.id] = 0

        self.random_prob = random_prob
        self.mutation_prob = mutation_prob
        self.number_new_confs = 4

        self.t = 0


    def update_model_single_observation(self, winner_id, tried_confs, instance_id):

        tried_confs.remove(winner_id)

        for conf in tried_confs:
            if winner_id not in self.w_store[conf].keys():
                self.w_store[winner_id][conf] = 0
            else:
                self.w_store[winner_id][conf] = self.w_store[winner_id][conf] + 1

            if winner_id not in self.n_store[conf].keys():
                self.n_store[winner_id][conf] = 0
            else:
                self.n_store[winner_id][conf] = self.n_store[winner_id][conf] + 1

            self.n_store[conf][winner_id] = self.n_store[winner_id][conf]


    def select_from_set(self, conf_set, instance_set, n_to_select):
        """
        For a set of configurations and instances select the most promising configurations
        """

        v_hat = np.zeros((len(conf_set), len(instance_set)))
        c_hat = np.zeros((len(conf_set), len(instance_set)))

        for c in range(len(conf_set)):
            conf = conf_set[c]
            for next_instance in range(len(instance_set)):
                epsilon = np.random.gumbel()

                F_S = 0
                n_sum = 0
                for r in conf_set:
                    w_j_r = self.w_store[conf.id][r.id]
                    n_j_r = self.n_store[conf.id][r.id]

                    if conf.id == r.id:
                        F = 0.5
                    elif n_j_r == 0:
                        F = 0
                    elif w_j_r == 0:
                        F = -10
                    elif w_j_r == n_j_r:
                        F = + 10
                    else:
                        x = w_j_r / n_j_r
                        F = - np.log((1/x) -1)

                    n_sum = n_sum + self.n_store[r.id][conf.id]
                    F_S = F_S + F

                if n_sum == 0:
                    c_hat[c][next_instance] = epsilon * 100000
                else:
                    c_hat[c][next_instance] = epsilon * np.sqrt(np.log((self.t))/n_sum)
                v_hat[c][next_instance] = 1/len(self.pool) *(1+F_S)


        v_hat = v_hat.sum(axis=1)
        c_hat = c_hat.sum(axis=1)

        v_hat_s = v_hat/ v_hat.sum()

        quality = v_hat_s + c_hat
        selection = (-quality).argsort()[:n_to_select]

        print(f"choosing {selection} {(-quality).argsort()} {[conf_set[i].id for i in selection]}")
        print(f" {[quality[i] for i in (-quality).argsort()]}, {list(quality)}")

        return [conf_set[i] for i in selection], [[v_hat_s[i], c_hat[i]] for i in (-quality).argsort() ]

    def delete_from_pool(self, instance_set):
        """
        Based on the feedback delete poor performing configurations from the pool
        """
        v_hat = np.zeros((self.pool_size, len(instance_set)))
        c_hat =  np.zeros((self.pool_size, len(instance_set)))

        for c in range(self.pool_size):
            conf = self.pool[c]
            for next_instance in range(len(instance_set)):
                epsilon = np.random.gumbel()

                F_S = 0
                n_sum = 0
                for r in self.pool:
                    w_j_r = self.w_store[conf.id][r.id]
                    n_j_r = self.n_store[conf.id][r.id]

                    if conf.id == r.id:
                        F = 0.5
                    elif n_j_r == 0:
                        F = 0
                    elif w_j_r == 0:
                        F = -10
                    elif w_j_r == n_j_r:
                        F = + 10
                    else:
                        x = w_j_r / n_j_r
                        F = - np.log((1 / x) -  1)

                    n_sum = n_sum + self.n_store[r.id][conf.id]
                    F_S = F_S + F

                if n_sum == 0:
                    c_hat[c][next_instance] = epsilon * 100000
                else:
                    c_hat[c][next_instance] = epsilon * np.sqrt(np.log((self.t)) / n_sum)
                v_hat[c][next_instance] = 1 / len(self.pool) * (1 + F_S)

        v_hat = v_hat.sum(axis=1)
        c_hat = c_hat.sum(axis=1)

        v_hat_s = v_hat / v_hat.sum()

        quality = v_hat_s + c_hat
        discard_index = (-quality).argsort()[len(quality)-int(len(quality)*self.dp ):]

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
            new_conf = {}
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

            for nc in range(number_to_create):
                new_promising_conf = new_promising_conf + [self.create_new_conf(conf_one, conf_two)]

            comb = self.pool + new_promising_conf
            for conf in new_promising_conf:
                self.w_store[conf.id] = {}
                self.n_store[conf.id] = {}
                for cc in comb:
                    self.w_store[conf.id][cc.id] = 0
                    self.n_store[conf.id][cc.id] = 0
                for cc in self.pool:
                    self.w_store[cc.id][conf.id] = 0
                    self.n_store[cc.id][conf.id] = 0

        self.pool = self.pool + new_promising_conf


    def update(self, results, previous_tournament, instance_features=None):
        """
        Updated the model with given feedback
        """

        best_conf_store = []
        rest_conf_store = []
        instance_store = []

        for instance in previous_tournament.instance_set:
            results_on_instance = {}

            for c in previous_tournament.configuration_ids:
                if not np.isnan(results[c][instance]):
                    results_on_instance[c] = results[c][instance]
                else:
                    results_on_instance[c] = self.scenario.cutoff_time

            best_conf_on_instance = min(results_on_instance, key=results_on_instance.get)

            if results_on_instance[best_conf_on_instance] >= self.scenario.cutoff_time:
                self.pool = [random_point(self.scenario, uuid.uuid4()) for _ in range(self.pool_size -1 )] + [default_point(self.scenario, uuid.uuid4())]
                continue

            tried = previous_tournament.configuration_ids.copy()

            instance_store.append(instance)
            best_conf_store.append(best_conf_on_instance)
            rest_conf_store.append(tried)

            self.t = self.t + 1

            self.update_model_single_observation(best_conf_on_instance, tried, instance)

        self.delete_from_pool(previous_tournament.instance_set)

        self.add_to_pool(previous_tournament.instance_set)


    def get_suggestions(self, scenario, n_to_select, next_instance_set, instance_features=None):
        """
        Suggest configurations to run next based on the instances that are comming
        """
        if self.t == 0:
            for conf in self.pool:
                self.w_store[conf.id] = {}
                self.n_store[conf.id] = {}
                for cc in self.pool:
                    self.w_store[conf.id][cc.id] = 0
                    self.n_store[conf.id][cc.id] = 0

        suggest, ranking = self.select_from_set(self.pool, next_instance_set, n_to_select)

        return suggest, ranking









