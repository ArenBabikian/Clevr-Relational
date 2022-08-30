import argparse

import yaml

from scene_parse.rel_net.config import RelNetConfiguration
from scene_parse.rel_net.constraints import build_adjacency_matrix, adj_probability, \
    build_adjacency_matrix_from_edge_list
from scene_parse.rel_net.datasets import get_test_dataloader
from scene_parse.rel_net.models import RelNetModule, SceneBasedRelNetModule
import torch
from tqdm import tqdm
import json


def predict_pair_based(opt, model, dataloader, scenes, relation_names, device):
    for sources, targets, _, source_ids, target_ids, img_ids, _ in tqdm(dataloader, 'processing objects batches'):
        sources = sources.to(device)
        targets = targets.to(device)
        preds = model.forward(sources, targets)

        if not opt.use_proba:
            preds = torch.round(preds).int().cpu().numpy()

        for i, label in enumerate(preds):
            img_id = img_ids[i]
            edge = (source_ids[i].item(), target_ids[i].item())
            for j, pred in enumerate(preds[i]):
                relation = relation_names[j]
                if not opt.use_proba and pred[i][j].item() == 1:
                    scenes[img_id]['relationships'][relation][edge[0]].append(edge[1])
                else:
                    scenes[img_id]['relationships'][relation][edge[0]].append((edge[1], preds[i][j].item()))


def predict_scene_based(opt, model, dataloader, scenes, relation_names, device):
    for data, sources, targets, labels, image_id, (num_nodes, num_edges), _ in tqdm(dataloader, 'processing objects batches'):
        num_nodes = num_nodes.item()
        num_edges = num_edges.item()

        data = data[:num_nodes].squeeze(dim=0).to(device)
        sources = sources[:, :num_edges].squeeze(dim=0).to(device).long()
        targets = targets[:, :num_edges].squeeze(dim=0).to(device).long()

        img_id = image_id.item()

        # Run the NN
        all_neuron_values = model.forward(data, sources, targets)

        # Find [src,tgt] pairs, also referred to as indices
        src_tgt_pairs = [[sources.tolist()[j], targets.tolist()[j]] for j in range(len(sources))]
        
        # NOTE ASSERTION
        for layer in all_neuron_values:
            assert len(layer) == len(src_tgt_pairs)
        
        # SAVE the neuron values to a bin file
        if opt.save_neuron_values:
            # Save as binary
            neuron_data = {"indices":src_tgt_pairs, "layers":all_neuron_values}
            torch.save(neuron_data, f'{opt.save_dir_path}/{img_id}.pt') # save as binary
            # Save as text
            neuron_data_txt = {"indices":src_tgt_pairs, "layers":[l.detach().numpy().tolist() for l in all_neuron_values]}
            with open(f'{opt.save_dir_path}-txt/{img_id}.json', 'w') as fp:
                json.dump(neuron_data_txt, fp) # save as text

        if opt.use_sigmoid:
            preds = all_neuron_values[-1]
        else:
            preds = all_neuron_values[-2]

        # SAVING THE PEDICTIONS
        if not opt.use_proba:
            preds = torch.round(preds).int().cpu().numpy()

        for i, label in enumerate(preds):
            edge = (sources[i].item(), targets[i].item())
            for j, pred in enumerate(preds[i]):
                relation = relation_names[j]
                if not opt.use_proba:
                    if preds[i][j].item() == 1:
                        scenes[img_id]['relationships'][relation][edge[0]].append(edge[1])
                else:
                    scenes[img_id]['relationships'][relation][edge[0]].append((edge[1], preds[i][j].item()))


def predict_scene_adj_based(opt, model, dataloader, scenes, relation_names, relation_map):
    for data, sources, targets, labels, image_id, (num_nodes, num_edges), _ in tqdm(dataloader,
                                                                                    'processing objects batches'):
        num_nodes = num_nodes.item()
        num_edges = num_edges.item()

        data = data[:num_nodes].squeeze(dim=0).to('cuda')
        sources = sources[:, :num_edges].squeeze(dim=0).to('cuda').long()
        targets = targets[:, :num_edges].squeeze(dim=0).to('cuda').long()

        img_id = image_id.item()

        preds = model.forward(data, sources, targets)

        adj = build_adjacency_matrix_from_edge_list(sources, targets, preds, num_nodes)
        adj = adj_probability(adj)

        if not opt.use_proba:
            adj = torch.round(preds).int().cpu().numpy()

        for i, _ in enumerate(preds):
            edge = (sources[i].item(), targets[i].item())
            prediction = adj[:, edge[0], [edge[1]]]
            for j, pred in enumerate(prediction):
                relation = relation_names[j]
                opposite_relation = relation_map[relation]
                pred = pred.item()
                if not opt.use_proba and pred == 1:
                    scenes[img_id]['relationships'][relation][edge[0]].append(edge[1])
                    scenes[img_id]['relationships'][opposite_relation][edge[1]].append(edge[0])
                else:
                    scenes[img_id]['relationships'][relation][edge[0]].append((edge[1], pred))
                    scenes[img_id]['relationships'][opposite_relation][edge[0]].append((edge[1], 1 - pred))

def main(opt):
    relation_map = None # {'left': 'right', 'front': 'behind'}

    #1 creates the model (search based)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if opt.use_pretrained:
        model = SceneBasedRelNetModule.load_from_checkpoint(opt.model_path, args=opt) if opt.model_type == 'scene_based' \
        else RelNetModule.load_from_checkpoint(opt.model_path, args=opt)
    else:
        model = SceneBasedRelNetModule(opt)

    #2 handles the input h5 file
    dataloader = get_test_dataloader(opt)

    #3 handle detected objects (input)
    with open(opt.scenes_path, 'r') as f:
        result = json.load(f)

    #4 set up the output json format
    scenes = result['scenes']
    relation_names = opt.label_names

    for scene in scenes:
        if 'relationships' not in scene:
            scene['relationships'] = {}

        if relation_map is None:
            for relation in relation_names:
                scene['relationships'][relation] = [[] for _ in scene['objects']]
        else:
            for relation in relation_map.keys() + relation_map.values():
                scene['relationships'][relation] = [[] for _ in scene['objects']]

    model.to(device)

    #5 make prediction by calling the smallNN
    if opt.model_type == 'scene_based':
        if relation_map is None:
            # NOTE We are in here
            predict_scene_based(opt, model, dataloader, scenes, relation_names, device)
        else:
            predict_scene_adj_based(opt, model, dataloader, scenes, relation_names, relation_map, device)
    else:
        predict_pair_based(opt, model, dataloader, scenes, relation_names)

    with open(opt.output_path, 'w') as f:
        json.dump({'scenes': [scenes[i] if i in opt.img_ids or not opt.img_ids  else {} for i in range(len(scenes))]}, f)
    print('Output scenes saved!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_fp', type=str, required=True)
    arguments = parser.parse_args()

    with open(arguments.config_fp) as fp:
        dataMap = yaml.safe_load(fp)

    config = RelNetConfiguration(**dataMap)
    main(config)
