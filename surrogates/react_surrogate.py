import copy
import sys
import os

sys.path.append(os.getcwd())

import uuid
import numpy as np
from trueskill import Rating, TrueSkill

from pool import Configuration, Generator
from conf_generetors import check_no_goods, check_conditionals, default_point, random_point,reset_conditionals, variable_graph_structure, graph_crossover



class REACTR:
    def __init__(self, scenario, seed, features=None, pool_size = 15,random_prob=0.25, mutation_prob=0.4,dp=0.2):

        self.scenario = scenario
        self.seed = seed
        self.dp = dp

        self.pool_size = pool_size
        self.pool = [random_point(scenario, uuid.uuid4()) for _ in range(self.pool_size -1 )] + [default_point(scenario, uuid.uuid4())]
        self.ranking_store = {}
        self.env = TrueSkill()

        self.t = 0

        self.random_prob = random_prob
        self.mutation_prob = mutation_prob


    def update_model_single_observation(self, winner_id, tried_confs, instance_id):
        """
        Update the thetas of the model for a single instance feedback
        """

        for conf in tried_confs:
            if conf not in self.ranking_store.keys():
                self.ranking_store[conf] = Rating()

        rating_groups = []
        ranks = []
        for conf in tried_confs:
            rating_groups.append((self.ranking_store[conf],))
            if conf == winner_id:
                ranks.append(0)
            else:
                ranks.append(1)

        rated_rating_groups = self.env.rate(rating_groups, ranks=ranks)

        for i in range(len(tried_confs)):
            self.ranking_store[tried_confs[i]] = rated_rating_groups[i][0]


    def select_from_set(self, conf_set, instance_set, n_to_select):
        """
        For a set of configurations and instances select the most promising configurations
        """
        by_ranking = int(np.around(n_to_select * 1))

        for conf in conf_set:
            if conf.id not in self.ranking_store.keys():
                self.ranking_store[conf.id] = Rating()

        quality = np.array([self.ranking_store[conf.id].mu for conf in conf_set])


        selection = (-quality).argsort()

        # Basically I am breaking ties
        quality_last = quality[selection[n_to_select-1]]
        possible_choice = []
        for i in range(len(quality)):
            if quality[i] == quality_last:
                possible_choice.append(i)

        if len(possible_choice) == 1:
            selection = (-quality).argsort()[:by_ranking]
        else:
            selection = (-quality).argsort()
            c = 0
            idx = None
            for d in selection:
                if quality[d] == quality_last:
                    c = c + 1
                    if idx == None:
                        idx = d
            selection = list(selection[:idx]) + list(np.random.choice(possible_choice, c, replace=False))

        return [conf_set[i] for i in selection], [quality[i] for i in selection]

    def delete_from_pool(self, instance_set):
        """
        Based on the feedback delete poor performing configurations from the pool

        """

        if self.dp == 1:
            self.pool = []

        elif self.dp == 0:
            pass
        else:
            
            discard = int(np.around(len(self.pool) * (1- self.dp))) -1

            for conf in self.pool:
                if conf.id not in self.ranking_store.keys():
                    self.ranking_store[conf.id] = Rating()
            quality = np.array([self.ranking_store[conf.id].mu for conf in self.pool])
            sorted_quality = (-quality).argsort()

            quality_last = quality[sorted_quality[discard]]
            possible_discard = []
            for i in range(len(quality)):
                if quality[i] == quality_last:
                    possible_discard.append(i)

            if len(possible_discard) == 1:
                discard_index = sorted_quality[discard:]
            else:
                discard_index = sorted_quality[discard:]
                c = 0
                for d in discard_index:
                    if quality[d] == quality_last:
                        c = c + 1
                discard_index = list(discard_index[c:]) + list(np.random.choice(possible_discard, c, replace=False))

            print(f"dicsarding {discard_index}")
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

            no_good = check_no_goods(self.scenario, new_conf.conf)
        return new_conf

    def add_to_pool(self, past_instances):
        """
        Add the most promising newly created configurations to the pool
        """
        number_to_create = self.pool_size - len(self.pool)
        new_confs = []
        if number_to_create > 0:
            new_confs = []
            if len(self.pool) > 0:
                best_to_confs, _ = self.select_from_set(self.pool, past_instances, 5)
                best_to_confs = np.random.choice(best_to_confs, 2, replace=False)
                conf_one, conf_two = best_to_confs[0], best_to_confs[1]

                for nc in range(number_to_create):
                    new_confs = new_confs + [self.create_new_conf(conf_one, conf_two)]

            else:
                for nc in range(number_to_create):
                    new_confs = new_confs + [random_point(self.scenario, uuid.uuid4())]

        self.pool = self.pool + new_confs


    def update(self, results, previous_tournament, instance_features=None):
        """
        Updated the model with given feedback
        """

        for instance in previous_tournament.instance_set:
            results_on_instance = {}

            for c in previous_tournament.configuration_ids:
                if not np.isnan(results[c][instance]):
                    results_on_instance[c] = results[c][instance]
                else:
                    results_on_instance[c] = self.scenario.cutoff_time

            best_conf_on_instance = min(results_on_instance, key=results_on_instance.get)

            tried = previous_tournament.configuration_ids.copy()

            self.t = self.t + 1

            self.update_model_single_observation(best_conf_on_instance, tried, instance)


        self.delete_from_pool(previous_tournament.instance_set)

        self.add_to_pool(previous_tournament.instance_set)


    def get_suggestions(self, scenario, n_to_select, next_instance_set, instance_features=None):
        """
        Suggest configurations to run next based on the instances that are comming
        """

        suggest, ranking = self.select_from_set(self.pool, next_instance_set, n_to_select)

        return suggest, ranking








