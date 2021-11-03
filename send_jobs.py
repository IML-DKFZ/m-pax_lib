from src.utils import computation


nodes = [computation.Euler()]

nb_repetitions = 1
gpus = 1
gpu_q = 4
gpu_model = 'GeForceGTX1080Ti'
email = False

experiment_config = 'opg_eval_1'
name = 'opg_veal_test'

script_name = 'run_eval.py'

for i in range(nb_repetitions):
    computation.run_experiment(nodes,
                               experiment_config,
                               gpus,
                               gpu_q,
                               gpu_model,
                               email,
                               name,
                               script_name)