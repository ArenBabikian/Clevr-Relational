import json
import os
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

scenes_path = "C:/git/Clevr-Relational/_clevr_data/_clevr_gnn/CLEVR_mini_coco_anns.json"
feature_dir_path = "_results/4000images/node2feature"

def main():
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    features_list = []
    neuron_list = []
    with open(scenes_path, 'r') as f:
        scenes = json.load(f)['scenes']

    # ITERATE THROUGH THE SCENES
    for scene in tqdm(scenes):
        ind = scene['image_index']

        feature_path = f'{feature_dir_path}/{ind}.pt'
        features = torch.load(feature_path)

        for f in features.detach().numpy():
            neuron_list.append(f)
        features_list.append(features)

    # features_tensor = torch.cat(features_list)
    neuron_tensor = torch.tensor(neuron_list)

    # DATA
    min = torch.min(neuron_tensor)
    max = torch.max(neuron_tensor)
    mean = torch.mean(neuron_tensor)
    std_dev = torch.std(neuron_tensor)
    variance = torch.var(neuron_tensor)

    # MEAN
    print(f'min : {min}')
    print(f'max : {max}')
    print(f'mean: {mean}')
    # print(f'mean: {torch.mean(features_tensor)}')

    # VARIANCE
    print(f'std-dev : {std_dev}')
    print(f'variance: {variance}')

    # PLOT
    bins = 100
    hist = torch.histc(neuron_tensor, bins = bins)

    # x1 is the neuron values an dnot the bins id
    # x1 = [(min + (i*(max-min) / bins)).item() for i in range(bins)]
    # x1 = np.arange(min, max, (max-min)/bins)

    x2 = range(bins)
    # print(torch.max(hist))
    plt.bar(x2, hist, align='center')
    plt.xlabel('bins')
    plt.show()

if __name__ == "__main__":
    main()
