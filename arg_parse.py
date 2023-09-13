import argparse

def parse_args():

    parser = argparse.ArgumentParser()
    hp = parser.add_argument_group("")
    so = parser.add_argument_group("Scenario options")

    class LoadOptionsFromFile(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            with values as f:
                parser.parse_args(f.read().split(), namespace)

    hp.add_argument('--file', type=open, action=LoadOptionsFromFile)

    hp.add_argument('--seed', default=42, type=int)
    hp.add_argument('--log_folder', type=str, default="latest")
    hp.add_argument('--memory_limit', type=int, default=1023*1.5)

    hp.add_argument('--ta_run_type', type=str, default="import_wrapper")
    hp.add_argument('--wrapper_mod_name', type=str, default="")
    hp.add_argument('--wrapper_class_name', type=str, default="")

    hp.add_argument('--tournament_size', type=int, default=16 )
    hp.add_argument('--number_tournaments', type=int, default=1)

    hp.add_argument('--algorithm', type=int, default=2)
    hp.add_argument('--dp', type=int, default=0.2)
    hp.add_argument('--init_pool', type=str, default="")


    so.add_argument('--scenario_file', type=str)
    so.add_argument('--cutoff_time', type=str)
    so.add_argument('--instance_file', type=str)
    so.add_argument('--feature_file', type=str)
    so.add_argument('--paramfile', type=str)

    return vars(parser.parse_args())