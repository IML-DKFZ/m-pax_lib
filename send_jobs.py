from src.utils import computation


nodes = [computation.Euler()]

nb_repetitions = 1
gpus = 1
gpu_q = 4
gpu_model = "GeForceRTX2080Ti"
email = False

experiment_config = "diagvibsix_tcvae_1"
name = "diagvib_z90_head"

script_name = "run_tcvae.py"

for i in range(nb_repetitions):
    computation.run_experiment(
        nodes, experiment_config, gpus, gpu_q, gpu_model, email, name, script_name
    )
