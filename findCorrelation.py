import argparse
import torch
from tqdm import tqdm

import yaml
from scene_parse.rel_net.config import RelNetConfiguration
from scene_parse.rel_net.tools.run_test import main


def runCorrelation():

    whitelist = None # {0,1} # for a range [0..999]
    save_path_dir = "./_results/neurons"
    load_saved_data = False

    if not load_saved_data:
        # SAVE NEURON DATA by running the NN

        # 1. run the run_test file, or get data from file. 
        parser = argparse.ArgumentParser()
        parser.add_argument('--config_fp', type=str, required=True)
        arguments = parser.parse_args()

        with open(arguments.config_fp) as fp:
            dataMap = yaml.safe_load(fp)

        # additional configs    
        dataMap['use_pretrained'] = False
        dataMap['img_ids'] = whitelist
        dataMap['save_neuron_values'] = True
        dataMap['save_dir_path'] = save_path_dir

        config = RelNetConfiguration(**dataMap)
        main(config)
        exit()
    else:
        # LOAD NEURON DATA for analysis
        all_data = {}
        for scene_id in tqdm(whitelist):
            path_to_bin = f'{save_path_dir}/{scene_id}.pt'
            neuron_data = torch.load(path_to_bin)
            all_data[scene_id] = neuron_data

        print(all_data)

# 1.1 do some kind of aggregation

# 2. gather metric values for the graphs (this may require running java code)

# 2.2 do some kind of aggregation

# 3. measure correlation

# 4. output some kind of figures



if __name__ == "__main__":
    runCorrelation()

    # python findCorrelation.py --config_fp clevr/sample_config.yaml