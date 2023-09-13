import sys
import os
sys.path.append(os.getcwd())

import importlib
import ray
import time
import uuid
import numpy as np
import logging
import json
np.set_printoptions(suppress=True)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from scenario import Scenario
from pool import Tournament, ParamType, Configuration, Generator

from surrogates.cppl_surrogate import CPPL
from surrogates.colstim_surrogate import COLSTIM
from surrogates.react_surrogate import REACTR
from surrogates.iss_surrogate import ISS
from surrogates.random_surrogate import RANDOM_CHOOSER
from surrogates.tsmnl import TSMNL
from surrogates.colstim_no_feature import COLSTIM_FS


from ta_execution import tae_from_cmd_wrapper_rt
from ta_result_store import TargetAlgorithmObserver
from arg_parse import parse_args



if __name__ == "__main__":

    selector_args = parse_args()

    # set up the wrapper to create the commands for the target algorithm
    wrapper_mod = importlib.import_module(selector_args["wrapper_mod_name"])
    wrapper_name = selector_args["wrapper_class_name"]
    wrapper_ = getattr(wrapper_mod, wrapper_name)
    ta_wrapper = wrapper_()

    # create scenario and set seed
    scenario = Scenario(selector_args["scenario_file"], selector_args)
    np.random.seed(int(scenario.seed) + 1)

    # params with large spans are log transfromed
    for param in scenario.parameter:
        if param.type == ParamType.continuous or param.type == ParamType.integer:
            if abs(param.bound[1] - param.bound[0]) >= 10000:
                param.scale = "l"

    # get the discard portion
    dp = selector_args["dp"]

    # use the first 100 instances to calibrate the pca
    first_x_features = {k: scenario.features[k] for k in scenario.instance_set[:100]}

    ray.init()
  #  ray.init(address="auto")
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[logging.FileHandler(f"./logs/{scenario.log_folder}/{wrapper_name}.log"),])

    # choose you favourite bandit method

    if scenario.algorithm == 1:
        model = COLSTIM(scenario, seed=int(scenario.seed),
                        pool_size=((scenario.tournament_size * scenario.number_tournaments)) * 2,
                        features=first_x_features, alpha=0.01, gamma=1,
                        mutation_prob=0.1, pca_dimension_configurations=5, pca_dimension_instances=3,
                        random_prob=0.1, w=1, dp=dp)

    elif scenario.algorithm == 2:
        model = REACTR(scenario, int(scenario.seed),
                       pool_size=((scenario.tournament_size * scenario.number_tournaments)) * 2, random_prob=0.1,
                       mutation_prob=0.1, dp=dp)

    elif scenario.algorithm == 3:
        model = ISS(scenario, seed=int(scenario.seed) ,
                    pool_size=((scenario.tournament_size * scenario.number_tournaments)) * 2,
                    eta=3.5, random_prob=0.1, mutation_prob=0.1, dp=dp)

    elif scenario.algorithm == 4:
        model = CPPL(scenario, int(scenario.seed), features=first_x_features,
                     pool_size=((scenario.tournament_size * scenario.number_tournaments)) * 2, alpha=0.2, gamma=1,
                     w=0.001, random_prob=0.1, mutation_prob=0.1,
                     pca_dimension_configurations=5, pca_dimension_instances=3, dp=dp)

    elif scenario.algorithm == 5:
        model = TSMNL(scenario, int(scenario.seed), features=first_x_features,
                 pool_size=((scenario.tournament_size * scenario.number_tournaments)) * 2, alpha=0.001, gamma=10,
                 random_prob=0.1, mutation_prob=0.1,
                 pca_dimension_configurations=5, pca_dimension_instances=3, dp=dp, sigma=0.0001)


    elif scenario.algorithm == 6:
        model = COLSTIM_FS(scenario, int(scenario.seed) + 1, pool_size=((scenario.tournament_size * scenario.number_tournaments)) *2,
                           random_prob = 0.1, mutation_prob = 0.1, dp=dp)

    elif scenario.algorithm == 7:
        model = RANDOM_CHOOSER(scenario, int(scenario.seed))

    # if provided set the pool of the method
    if selector_args["init_pool"]:
        pool_path = selector_args["init_pool"]
        with open(pool_path) as f:
            init_pool = f.read()

        new_pool = []
        spl = init_pool.split("}}")
        for conf in spl[:-1]:
            conf.replace("\n", "")
            conf = conf + "}}"
            conf_dic  = json.loads(conf)
            c = Configuration(uuid.UUID(list(conf_dic.keys())[0]),list(conf_dic.values())[0], Generator.random)
            new_pool.append(c)

        model.pool = new_pool

    global_cache = TargetAlgorithmObserver.remote(scenario)
    results = {}
    t_hist = []

    start = time.time()
    winner_dic = {}
    time_combined = {}
    counter = 0

    # main exp loop
    for instance_time_step in scenario.instance_set:

        start_loop = time.time()
        # get the confs to try
        to_run, ranking = model.get_suggestions(scenario, scenario.tournament_size*scenario.number_tournaments-1, [instance_time_step], {instance_time_step: scenario.features[instance_time_step]})

        # run the confs
        tasks = [tae_from_cmd_wrapper_rt.remote(c, instance_time_step, global_cache, ta_wrapper, scenario) for c in to_run]

        # wait for a conf to finish
        go_ta = True
        while go_ta:
            winner, not_ready = ray.wait(tasks)
            tasks = not_ready

            ta_return = ray.get(winner)[0]
            if ta_return[2] == False or len(tasks)==0:
                go_ta = False

        # get the conf finishing first
        best_finisher = ta_return[0]

        # cancel all other confs
        [ray.cancel(t, recursive=False) for t in not_ready]

        time_combined[counter] = time.time() - start_loop
        time.sleep(3)

        # get all the results
        results = ray.get(global_cache.get_results.remote())
        print(f"Winner {best_finisher.id}")

        # set up a fake tournament. this is need to fit with the syntax
        worst = [c for c in to_run if c.id !=best_finisher.id ]
        configuration_ids = [c.id for c in to_run]
        rtou = Tournament(uuid.uuid4(), [best_finisher], worst, [], configuration_ids, {}, [instance_time_step],1)

        # make sure all confs that did no finish have a nan indicating that they were not first
        for c in worst:
            if c.id not in results.keys():
                results[c.id] = {}
            if instance_time_step not in results[c.id].keys():
                results[c.id][instance_time_step] = np.nan

        # update the mode
        model.update(results, rtou, {instance_time_step: scenario.features[instance_time_step]})

        # this is for bookkeeping
        t_hist.append(rtou)
        
        winner_dic[counter] = results[best_finisher.id][instance_time_step]
        counter = counter + 1

        # save the results
        # only has the runtimes of the confs
        with open(f"./logs/{scenario.log_folder}/best_rt.json", 'w') as f:
            json.dump(winner_dic, f, indent=2)

        # has the runtime of the confs plus the time it took to select them
        with open(f"./logs/{scenario.log_folder}/best_rt_combined.json", 'w') as f:
            json.dump(time_combined, f, indent=2)

# more bookkeeping
with open(f"./logs/{scenario.log_folder}/run_history.json", 'a') as f:
    history = {str(k): v for k, v in results.items()}
    json.dump(history, f, indent=2)

