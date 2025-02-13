import argparse
import json
import os
import numpy as np
import torch
from tqdm import tqdm

import yaml
from correlation.metrics.metrics_neuron import NeuronMetrics
from scene_parse.rel_net.config import RelNetConfiguration
from scene_parse.rel_net.tools.run_test import main
from correlation.metrics.metrics_graph import GraphMetrics
from correlation.metrics import measUtil
from scipy.stats.stats import pearsonr


def runCorrelation():
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    ########################
    ## DATA GATHERING
    ########################

    # PARAMETERS
    whitelist = {3999, 3998} # range(4000)
    save_path_dir = "./_results_mini_coco_3"
    # whitelist = range(1000)  # None 
    # save_path_dir = "./_results"
    save_neuron_data = True
    # TODO add config path here?
    # TODO make all of these as command-line args? OR make a yaml file for this?

    ## STEP 1.0.0: SAVE NEURON DATA (if necessary)
    if save_neuron_data:
        # SAVE NEURON DATA by running the NN

        # 1. run the run_test file, or get data from file. 
        parser = argparse.ArgumentParser()
        parser.add_argument('--config_fp', type=str, required=True)
        arguments = parser.parse_args()

        with open(arguments.config_fp) as fp:
            dataMap = yaml.safe_load(fp)

        # additional configs 
        # TODO create a custom yaml file in the correlation folder and add these there?   
        dataMap['use_pretrained'] = False
        dataMap['img_ids'] = whitelist
        dataMap['save_neuron_values'] = True
        dataMap['save_feature_values'] = True
        dataMap['save_dir_path'] = save_path_dir

        config = RelNetConfiguration(**dataMap)
        main(config)
        exit()

    ## STEP 1.0.1: LOAD NEURON DATA for analysis
    all_neuron_data = {}
    if not whitelist :
        whitelist = range(1000) # TODO adjust this for the 4000 models
    for scene_id in tqdm(whitelist):
        path_to_bin = f'{save_path_dir}/neurons/{scene_id}.pt'
        neuron_data = torch.load(path_to_bin)
        all_neuron_data[scene_id] = neuron_data

    ## STEP 1.1.0: GATHER GRAPH NEIGHBORHOOD DATA (this may require running java code)
    depth = 1
    parallels = 0 # 2147483647
    
    neigh_path_dir = f"{save_path_dir}/neighborhoods"
    neigh_path = f'{neigh_path_dir}/neigh{depth}dep{parallels}pars.json'
    with open(neigh_path, 'r') as f:
        all_neigh_data = json.load(f)

    ## STEP 1.1.1: GATHER GROUND TRUTH GRAPH STRUCTURE DATA
    # TODO FURTHER DEVELOPMENT: get ground truth data from the rel_scenes.json file (I have done something super similar in Java)
    # transfor to networkx-compatible format
    # gather basic metric using networkx


    ########################
    ## MEASURING METRICS + AGGREGATING + PLOTING
    ########################

    fig_dir_path = f"{save_path_dir}/figures"

    ## STEP 2.0: TODO AGGREGATE NEURON DATA

    util_neu = NeuronMetrics(all_neuron_data)
    # MIN and MAX
    m10_neu, m11_neu = util_neu.getNeuronValueExtremumForAGivenLayer(1)
    measUtil.plotDataVsModels([m10_neu, m11_neu], show=True, save_file_path=f'{fig_dir_path}/neuron_extremum.pdf' , sort_index=1)

    # STEP 2.1: TODO AGREGATE GRAPH DATA

    util_nei = GraphMetrics(all_neigh_data)
    m1_nei=util_nei.getNumberOfDistinctNeighborhoods()  

    # OTHER STUFF
    allCoverages = []
    x_vals = []
    corr_vals = []
    ind = 0
    for t in np.arange(-3, 3, 0.1):
        
        x_vals.append(t)
        neu_cov = util_neu.getRatioOfNeuronsAboveThresholdValueForAGivenLayer(1, t)
        allCoverages.append(neu_cov)
        
        # print correlation
        c = pearsonr(m1_nei, neu_cov)
        print(f'{ind}, {t} = {c}')
        ind+=1
        corr_vals.append(c[0])
    
    # measUtil.plotDataVsModels(allCoverages, show=True, sort_index=0)
    # measUtil.plotDataVsModels([allCoverages[2]], show=True, sort_index=0)

    # STEP 3: measure correlation

    # TODO skip for now 

    # STEP 4: output some kind of figures

    measUtil.plotDataVsModels([corr_vals], x_values=x_vals, show=True, save_file_path=f'{fig_dir_path}/corr_vs_thresh.pdf')
    measUtil.plot2seriesVsModels(m1_nei, allCoverages[22], show=True, save_file_path=f'{fig_dir_path}/best_pos_corr.pdf' , sort_index=1)
    measUtil.plot2seriesVsModels(m1_nei, allCoverages[41], save_file_path=f'{fig_dir_path}/best_neg_corr.pdf' , show=True, sort_index=1)


if __name__ == "__main__":
    runCorrelation()

    # python findCorrelation.py --config_fp clevr/sample_config.yaml