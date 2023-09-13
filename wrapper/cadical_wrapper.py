import argparse
import json
import ast
import numpy as np


class Cadical_Wrapper():

    def get_command_line_args(self, runargs, config):

        instance = runargs["instance"]

        configuration =""
        for param, value in config.items():
            if param == "P" or param== "L":
                configuration += f" -{param}{value}"
            else:
                if isinstance(value, str):
                    value = value.replace(" ", "")
                if value == True and isinstance(value, (bool, np.bool_)):
                    value = "true"
                elif value == False and isinstance(value, (bool, np.bool_)):
                    value = "false"
                configuration += f" --{param}={value}"

        exc = './input/target_algorithms/cadical-sc17-proof/build/cadical/build/cadical'
        instance_p = f'./input/{instance}'

        cmd = f"stdbuf -oL {exc} {configuration} -q {instance_p}"
        print(cmd)
        return cmd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--runargs',type=ast.literal_eval)
    parser.add_argument('--config',type=ast.literal_eval)

    args = vars(parser.parse_args())

    wrapper = Cadical_Wrapper()
    print(wrapper.get_command_line_args(args["runargs"], args["config"]))
