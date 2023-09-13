# colstim

This is an implementation of diffrend bandit methods for realtime AC, as described in the paper:
Contextual Preselection Methods in Pool-based Realtime Algorithm Configuration.

#### Abstract 
Realtime algorithm configuration is concerned with the task of designing a dynamic algorithm configurator that observes sequentially arriving problem instances of an algorithmic problem class for which it selects suitable algorithm configurations (e.g., minimal runtime) of a specific target algorithm.
	The Contextual Preselection under the Plackett-Luce (CPPL) algorithm maintains a pool of configurations from which a set of algorithm configurations is selected that are run in parallel on the current problem instance.
	It uses the well-known UCB selection strategy from the bandit literature, while the pool of configurations is updated over time via a racing mechanism.
	In this paper, we investigate whether the performance of CPPL can be further improved by using different bandit-based selection strategies as well as a ranking-based strategy to update the candidate pool.
	Our experimental results show that replacing these components can indeed improve performance again significantly. 

#### Requirements
Python 3 and requirements.txt

#### Running the Code

+ Parameter *--algorithm* governs the bandit model choice
+ To run a wrapper that creates the command to execute the target algorithm given a configuration and an instance is needed. See the example wrapper for an example for Cadical.
+ The paramter *--tournament_size* governs how many ta runs are executed in parallel and should be set to the number of available cores.
+ Parameter *--dp* governs the discard portion

```
# Example
	python ./run_realtime.py  --file "./input/scenarios/cadical_example/realtime_args.txt" --scenario_file ./input/scenarios/cadical_example/scenario.txt --log_folder latest/cadical_example --seed 40 --algorithm ${i}

``` 