from modules.experiment import run_experiment
from torch.utils.tensorboard import SummaryWriter
import sys
import yaml
from pprint import pprint

if __name__ == "__main__":  
    args = sys.argv
    if (len(args) > 2):
        CONFIG_PATH = str(args[1])
        EXPERIMENT_ID = int(args[2])
    else:
        print("arguments missing")

    with open(CONFIG_PATH) as f:
        hyper_params = yaml.load(f, Loader=yaml.SafeLoader)['parameters']

    hyper_params['experiment id'] = EXPERIMENT_ID
    pprint(hyper_params)
    
    run_experiment(hyper_params)

    

    