from util_functions import *
from sklearn.model_selection import ParameterGrid
from itertools import repeat
from tqdm import tqdm
from pathos.pools import ProcessPool

from hparams import h_params, datasets

param_grid = {
    'wait_time_between_classes': [0, 5000, 20000],
    'sparsity': [0.1, 1, 100, 1000],
    'ratio_offline': [0.1, 0.4, 0.7],
    'ratio_known_classes': [0.3, 0.5, 0.7],
    'random_state': [42],
}


cpus = int(sys.argv[1])
index = int(sys.argv[2])
aim_dir = sys.argv[3]

print(f'JOB #: {index}', flush=True)

ds = index // 3
alg = index % 3

filename = datasets[ds]['filename']
target = datasets[ds]['target']

algorithm = {0: 'Minas', 1: 'ECSMiner', 2: 'ECSMinerWF'}

parameters = list(ParameterGrid(param_grid))

print(f'Loading dataset {filename}...', flush=True)
if filename.endswith('.arff'):
    df = load_MOA(f'/home/jgaud/projects/def-pbranco/jgaud/NoveltyDetectionMetricsExtension/data/prepared_data/{filename}')
elif filename.endswith('.csv'):
    df = load_CSV(f'/home/jgaud/projects/def-pbranco/jgaud/NoveltyDetectionMetricsExtension/data/prepared_data/{filename}')
else:
    print('File extension not supported', flush=True)

n_samples = df.shape[0]

pool = ProcessPool(cpus)


with tqdm(total=len(parameters)) as pbar:
    for result in pool.imap(start_experiment, repeat(algorithm[alg]), parameters, repeat(target), repeat(df), repeat(h_params[filename][algorithm[alg]]), repeat(n_samples), repeat(filename), repeat(aim_dir)):
        # Update the progress bar
        pbar.update(1)

pool.close()
pool.join()