import copy
import sys
import os
from sklearn.preprocessing import OneHotEncoder
import time


sys.path.append(os.getcwd())

import uuid
import numpy as np
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pool import Configuration, Generator, ParamType

np.set_printoptions(threshold=sys.maxsize)

from conf_generetors import check_no_goods, check_conditionals, default_point, random_point,reset_conditionals, variable_graph_structure, graph_crossover, PointGen





class CPPL:
    def __init__(self, scenario, seed,features, pool_size = 15, alpha=0.2, gamma=1,w=0.001,random_prob=0.1, mutation_prob=0.1,
                 pca_dimension_configurations=5, pca_dimension_instances=3, dp=0.2):
        """
        CPPL implementation based on the paper: "Pool-based realtime algorithm configuration: A preselection bandit approach"
        """

        self.scenario = scenario
        self.features = features
        self.seed = seed

        self.pool_size = pool_size
        self.pool = [random_point(scenario, uuid.uuid4()) for _ in range(self.pool_size -1 )] + [default_point(scenario, uuid.uuid4())]
        self.feature_store = {}

        self.alpha = alpha
        self.gamma = gamma
        self.w = w
        self.t = 0
        self.dp = dp

        self.random_prob = random_prob
        self.mutation_prob = mutation_prob
        self.number_new_confs = 4

        self.pca_dimension_configurations = pca_dimension_configurations
        self.pca_dimension_instances = pca_dimension_instances
        self.pca_number_confs_calibration = 100

        self.context_dim = self.pca_dimension_configurations * 2 + self.pca_dimension_instances * 2 + self.pca_dimension_configurations * self.pca_dimension_instances

        self.theta_hat = np.random.random_sample(self.context_dim)
        self.theta_bar = copy.copy(self.theta_hat)

        self.gradient_sum = {0: np.zeros((self.context_dim , self.context_dim ))}
        self.hessian_sum = {0: np.zeros((self.context_dim , self.context_dim ))}

        self.process_parameter()

        instance_feature_matrix = np.array(list(self.features.values()))
        self.instance_feature_standard_scaler = StandardScaler()
        transformed_features = self.instance_feature_standard_scaler.fit_transform(instance_feature_matrix)
        for instance, counter in zip(self.features.keys(), range(len(self.features.keys()))):
            self.features[instance] = transformed_features[counter]

        self.calibrate_pca()

    def process_parameter(self):
        """
        Figure out the parameter types and get values/calibrate encoding and scaling
        """

        cat_params = []
        cont_int_params = []
        # figure out the param type
        for param in self.scenario.parameter:
            if param.type == ParamType.categorical:
                cat_params.append(param)
            else:
                cont_int_params.append(param)

        # treat cont and int params
        self.lower_b_params = np.zeros(len(cont_int_params))
        self.upper_b_params = np.zeros(len(cont_int_params))

        for i in range(len(cont_int_params)):
            bound = cont_int_params[i].bound
            self.lower_b_params[i] = float(bound[0])
            self.upper_b_params[i] = float(bound[-1])

        # init one hot encoding for cat params:
        list_of_bounds = []
        list_o_of_default = []
        # With longest this is some super wired hack I have to do for the OneHotEncoder
        # If I do not find tha parameter with the longest string to parse in as default
        # the categories will become dtype Ux with x being the lenght of the longest string in the default
        # if then there are values that are longer then UX which get parsed in later these will be cut to the length..
        for param in cat_params:
            if isinstance(param.default, (bool, np.bool_)):
                list_of_bounds.append(list(map(str, list(map(int, param.bound)))))
                list_o_of_default.append( str(int(param.default)))
            else:
                longest = 0
                list_of_bounds.append(param.bound)
                for bound in param.bound:
                    if len(bound) > longest:
                        longest = len(bound)
                        cp = bound
                list_o_of_default.append(cp)

        self.cat_params_names = [p.name for p in cat_params]
        self.cont_int_params_names = [p.name for p in cont_int_params]

        if len(self.cat_params_names) > 0:
            self.o_h_enc = OneHotEncoder(categories=list_of_bounds)
            self.o_h_enc.fit(np.array(list_o_of_default).reshape(1, -1) )

    def scale_conf(self, configuration):
        """
        Scale and encode a conf. Cont/int parameter are scaled between 0 and 1. Cat. parameter are one hot encoded
        """
        cat_params_on_conf = np.zeros(len(self.cat_params_names ),dtype = object)
        cont_int_params_of_conf = np.zeros(len(self.cont_int_params_names ), dtype = float)

        for param, value in configuration.conf.items():
            if param in self.cat_params_names:
                if isinstance(value, (bool, np.bool_)):
                    cat_params_on_conf[self.cat_params_names.index(param)] = str(int(value))
                else:
                    cat_params_on_conf[self.cat_params_names.index(param)] = value
            else:
                cont_int_params_of_conf[self.cont_int_params_names .index(param)] = value

        cont_int_scaled = (cont_int_params_of_conf - self.lower_b_params) / (self.upper_b_params - self.lower_b_params).reshape(1, -1)

        if len(self.cat_params_names) > 0:
            cat_params_on_conf = self.o_h_enc.transform(cat_params_on_conf.reshape(1, -1)).toarray()

        return np.concatenate((cont_int_scaled, cat_params_on_conf), axis=None)

    def calibrate_pca(self):
        """
        Calibarte the PCA and Scalers for the featuremap
        """

        random_generator = PointGen(self.scenario, random_point)
        if len(self.cat_params_names) > 0:
            para_vector_size = len([item for sublist in self.o_h_enc.categories_ for item in sublist]) + len(self.cont_int_params_names)
        else:
            para_vector_size = len(self.cont_int_params_names)

        conf_matrix = np.zeros((self.pca_number_confs_calibration, para_vector_size))
        for i in range(self.pca_number_confs_calibration):
            point = random_generator.point_generator()
            conf_matrix[i] = self.scale_conf(point)

        self.pca_configurations = PCA(n_components=self.pca_dimension_configurations)
        pca_conf = self.pca_configurations.fit_transform(conf_matrix)

        self.pca_instances = PCA(n_components=self.pca_dimension_instances)
        pca_features = self.pca_instances.fit_transform(np.array(list(self.features.values())))

        sample_size =  min(pca_conf.shape[0], pca_features.shape[0])
        pca_conf = pca_conf[:sample_size]
        pca_features = pca_features[:sample_size]

        feature_map_matrix = np.zeros((sample_size * sample_size, self.context_dim))

        fc = 0
        for i in range(sample_size):
            for j in range(sample_size):
                feature_map_matrix[fc] = np.concatenate((pca_conf[i], pca_features[j], pca_conf[i] ** 2, pca_features[j] ** 2,
                                                         (pca_conf[i] * pca_features[j].reshape(-1,1)).flatten())).flatten()
                fc = fc + 1

        self.feature_map_scaler = StandardScaler()
        self.feature_map_scaler.fit(feature_map_matrix)


    def compute_feature_map(self, conf, instance_features):
        """
        For a conf/instance pair compute the quadratic feature map and scale
        """

        conf_values = self.scale_conf(conf)

        conf_values = self.pca_configurations.transform(conf_values.reshape(1, -1))
        instance_features = self.pca_instances.transform(instance_features.reshape(1, -1))

        features = np.concatenate((conf_values, instance_features, conf_values**2, instance_features ** 2,
                                   (conf_values * instance_features.reshape(-1,1)).flatten().reshape(1,-1)),axis=1).flatten()

        features = features/max(features)

        return features


    def update_feature_store(self, conf, instance):
        """
        For a conf/instance pair compute the features and store them in a feature store for later
        """
        if conf.id not in self.feature_store:
            self.feature_store[conf.id] = {}

        if instance not in self.feature_store[conf.id]:
            self.feature_store[conf.id][instance] = self.compute_feature_map(conf, self.features[instance])

    def compute_a(self, theta, tried_conf_ids, instance_id):
        """
        Compute a of the paper
        """
        sum = 0
        for conf in tried_conf_ids:
            sum = sum + self.feature_store[conf][instance_id] * \
                  np.exp(np.dot(self.feature_store[conf][instance_id], theta))
        return sum

    def compute_b(self,theta, tried_conf_ids, instance_id):
        """
        Compute b of the paper
        """
        sum = 0
        for conf in tried_conf_ids:
            sum = sum + np.exp(
                np.dot(self.feature_store[conf][instance_id], theta))
        return sum

    def compute_c(self, theta, tried_conf_ids, instance_id):
        """
        Compute c of the paper
        """
        sum = 0
        for conf in tried_conf_ids:
            sum = sum +  np.exp(np.dot(self.feature_store[conf][instance_id], theta)) * \
                  np.outer(self.feature_store[conf][instance_id], self.feature_store[conf][instance_id])
        return sum

    def compute_gradient(self, theta, winner_id, tried_confs, instance_id):
        """
        Compute the gradient for a given feedback
        """
        a = self.compute_a(theta, tried_confs, instance_id)
        b = self.compute_b(theta, tried_confs, instance_id)

        return self.feature_store[winner_id][instance_id] - (a/b)

    def compute_hessian(self,theta, winner_id, tried_confs, instance_id):
        """
        Compute the hessian for a given feedback
        """
        a = self.compute_a(theta, tried_confs, instance_id)
        b = self.compute_b(theta, tried_confs, instance_id)
        c = self.compute_c(theta, tried_confs, instance_id)
        return (np.outer(a,a) / b ** 2) - (c / b)

    def update_running_sums(self, theta, winner_id, tried_confs, instance_id):
        """
        Update the rolling sums of gradients and hessian for a feedback
        """

        if self.t not in self.gradient_sum.keys():
            gradient = self.compute_gradient(theta, winner_id, tried_confs, instance_id)
            self.gradient_sum[self.t] = self.gradient_sum[self.t - 1] + np.outer(gradient, gradient)

        if self.t not in self.hessian_sum.keys():
            hessian = self.compute_hessian(theta, winner_id, tried_confs, instance_id)
            self.hessian_sum[self.t] = self.hessian_sum[self.t - 1] + hessian ** 2

    def compute_confidence(self, theta, instance_id, conf):
        """
        Compute the confidence for an instance/conf combination
        """

        v = (1/self.t) * self.gradient_sum[self.t]

        s = (1/self.t) * self.hessian_sum[self.t]

        try:
            s_inv = np.linalg.inv(s)
        except:
            s_inv = np.linalg.pinv(s)

        sigma = (1/self.t) * np.dot(np.dot(s_inv, v), s_inv)

        M = np.exp(2 * np.dot(self.feature_store[conf][instance_id], theta)) * \
            np.outer (self.feature_store[conf][instance_id] ,self.feature_store[conf][instance_id])

        sigma_root = sqrtm(sigma)
        I_hat = np.linalg.norm(np.dot(np.dot(sigma_root, M), sigma_root))

        return self.w * np.sqrt((2 * np.log(self.t) + self.context_dim + 2 * np.sqrt((self.context_dim * np.log(self.t)))) * I_hat)

    def update_model_single_observation(self, winner_id, tried_confs, instance_id):
        """
        Update the thetas of the model for a single instance feedback
        """
        grad = self.compute_gradient(self.theta_hat , winner_id, tried_confs, instance_id)
        self.theta_hat = self.theta_hat + self.gamma * self.t ** (-self.alpha) * grad

        self.theta_hat = self.theta_hat/max(self.theta_hat)

        self.theta_bar = ((self.t - 1) * (self.theta_bar)) / self.t + (self.theta_hat / self.t)

    def update_model_mini_batch(self, winner_ids, tried_confs, instance_ids):
        """
        Update the thetas of the model for feedback over multiple instances
        """
        grad_mean = np.zeros(self.context_dim)

        for uc in range(len(winner_ids)):
            grad = self.compute_gradient(self.theta_hat , winner_ids[uc], tried_confs[uc], instance_ids[uc])
            grad_mean = grad_mean + grad

        grad_mean = grad_mean / len(winner_ids)
        self.theta_hat = self.theta_hat + self.gamma * self.t ** (-self.alpha) * grad_mean

        self.theta_bar = ((self.t - 1) * (self.theta_bar)) / self.t + (self.theta_hat / self.t)


    def select_from_set(self, conf_set, instance_set, n_to_select):
        """
        For a set of configurations and instances select the most promising configurations
        """

        v_hat = np.zeros((len(conf_set), len(instance_set)))
        confidence =  np.zeros((len(conf_set), len(instance_set)))

        for c in range(len(conf_set)):
            conf = conf_set[c]
            for next_instance in range(len(instance_set)):
                v_hat[c][next_instance] =  np.exp(np.inner(self.feature_store[conf.id][instance_set[next_instance]], self.theta_bar))

                if self.t > 0:
                    confidence[c][next_instance] = self.compute_confidence(self.theta_bar, instance_set[next_instance], conf.id)

        v_hat_s = v_hat.sum(axis=1)
        v_hat_s = v_hat_s/max(v_hat_s)

        if self.t > 0:
            confidence_s = confidence.sum(axis=1)
            confidence_s = confidence_s / max(confidence_s)
        else:
            confidence_s = np.zeros(v_hat_s.shape)

        quality = v_hat_s + confidence_s

        selection = (-quality).argsort()[:n_to_select]
        print(f"choosing {selection} {(-quality).argsort()} {[conf_set[i].id for i in selection]}")
        print(f" {[[v_hat_s[i], confidence_s[i]] for i in (-quality).argsort() ]}{list((-quality).argsort())}")
        return [conf_set[i] for i in selection], [[v_hat_s[i], confidence_s[i]] for i in (-quality).argsort() ]

    def delete_from_pool(self, instance_set):
        """
        Based on the feedback delete poor performing configurations from the pool
        """
        v_hat = np.zeros((self.pool_size, len(instance_set)))
        confidence =  np.zeros((self.pool_size, len(instance_set)))

        for c in range(self.pool_size):
            conf = self.pool[c]
            for next_instance in range(len(instance_set)):
                v_hat[c][next_instance] =  np.exp(np.inner(self.feature_store[conf.id][instance_set[next_instance]], self.theta_bar))

                if self.t > 0:
                    confidence[c][next_instance] = self.compute_confidence(self.theta_bar, instance_set[next_instance], conf.id)

        v_hat_s = v_hat.sum(axis=1)
        v_hat_s = v_hat_s/max(v_hat_s)

        if self.t > 0:
            confidence_s = confidence.sum(axis=1)
            confidence_s = confidence_s / max(confidence_s)
        else:
            confidence_s = np.zeros(self.pool_size)

        if self.dp:
            quality = v_hat_s + confidence_s
            discard_index = (-quality).argsort()[len(quality)-int(len(quality)* self.dp):]
        else:
            discard_index = []
            for c in range(self.pool_size):
                for oc in range(self.pool_size):
                    if c != oc and v_hat_s[oc] - confidence_s[oc] > v_hat_s[c] + confidence_s[c]:
                        discard_index.append(c)
                        break

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
            possible_new_confs = []
            for nc in range(self.number_new_confs * number_to_create):
                possible_new_confs = possible_new_confs + [self.create_new_conf(conf_one, conf_two)]
            for instance in past_instances:
                for c in possible_new_confs:
                    self.update_feature_store(c, instance)
            new_promising_conf, _ = self.select_from_set(possible_new_confs, past_instances, number_to_create)
        self.pool = self.pool + new_promising_conf


    def update(self, results, previous_tournament, instance_features=None):
        """
        Updated the model with given feedback
        :param results: nested dic containing rt feedback for the conf instance pairs in previous_tournament
        :param previous_tournament: Tournament
        :return:
        """
        if instance_features:
            instance_feature_matrix = np.array(list(instance_features.values()))
            transformed_features = self.instance_feature_standard_scaler.transform(instance_feature_matrix)
            for instance, counter in zip(instance_features.keys(), range(len(instance_features.keys()))):
                self.features[instance] = transformed_features[counter]

        best_conf_store = []
        rest_conf_store = []
        instance_store = []

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

            instance_store.append(instance)
            best_conf_store.append(best_conf_on_instance)
            rest_conf_store.append(tried)

            self.t = self.t + 1

            self.update_model_single_observation(best_conf_on_instance, tried, instance)

            self.update_running_sums(self.theta_bar, best_conf_on_instance, tried, instance)

        self.delete_from_pool(previous_tournament.instance_set)

        self.add_to_pool(previous_tournament.instance_set)


    def get_suggestions(self, scenario, n_to_select, next_instance_set, instance_features=None):
        """
        Suggest configurations to run next based on the instances that are comming
        """

        if instance_features:
            instance_feature_matrix = np.array(list(instance_features.values()))
            transformed_features = self.instance_feature_standard_scaler.transform(instance_feature_matrix)
            for instance, counter in zip(instance_features.keys(), range(len(instance_features.keys()))):
                self.features[instance] = transformed_features[counter]

        for instance in next_instance_set:
            for c in self.pool:
                self.update_feature_store(c, instance)

        suggest, ranking = self.select_from_set(self.pool, next_instance_set, n_to_select)

        return suggest, ranking


    def suggest_from_outside_pool(self, conf_set, n_to_select, next_instance_set, instance_features=None):

        if instance_features:
            instance_feature_matrix = np.array(list(instance_features.values()))
            transformed_features = self.instance_feature_standard_scaler.transform(instance_feature_matrix)
            for instance, counter in zip(instance_features.keys(), range(len(instance_features.keys()))):
                self.features[instance] = transformed_features[counter]

        for instance in next_instance_set:
            for c in conf_set:
                self.update_feature_store(c, instance)

        suggest, ranking = self.select_from_set(conf_set, next_instance_set, n_to_select)

        return suggest






