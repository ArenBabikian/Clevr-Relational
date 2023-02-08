import yaml
from correlation.gnn.config import SceneReconstructionConfig
from correlation.gnn.dataset import SceneGraphDatasetModule
from correlation.gnn.models.gnn_model import SceneConstructionModule
import torch
from tqdm import tqdm
import numpy as np
import h5py


def main(opt):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for model_name, model_path in opt.model_paths.items():
        print(model_name)
        print(model_path)
        print()

        # Load the model
        model = SceneConstructionModule.load_from_checkpoint(model_path, args=opt)
        model.to(device)

        # Define input graph structure
        # These are defined in _datasets/IEPVQA/processed/data-{something}
        # note that these also contain the gt output y (either random, or the features derived from the valudation images (resnet), saved at _data/iepvqa/val_features_0_3000.h5)

        # opt.intermediate_gt = model_name
        dataloader = SceneGraphDatasetModule(opt).train_dataloader()

        # Add <save_dir : "_data"> to yaml for non DISTINCT
        dataset_path = f'{opt.save_dir}/k_{model_name}_{opt.min_id}_{opt.max_id}.h5'
        with h5py.File(dataset_path, 'w') as f:

            all_features_dataset = f.create_dataset('features', (opt.max_id-opt.min_id, 1024, 14, 14),
                                        dtype=np.float32)

            for batch_graph, num_obj_seq, image_indices in tqdm(dataloader):
                # print(batch_graph)
                # print(image_indices)

                features_per_node = model.forward(batch_graph)

                # derive feature per model from feature per node
                features_per_model = [] 
                start_id = 0
                for n in num_obj_seq:
                    features_per_model.append(torch.mean(features_per_node[start_id:start_id+n.data], 0).reshape(1024, 14, 14))
                    start_id = start_id+n.data
                
                assert image_indices.size(dim=0) == len(features_per_model)
                # ADD to dataset
                for i, image_i in enumerate(image_indices):
                    all_features_dataset[image_i] = features_per_model[i].detach().numpy()

                # print(len(features_per_model))
                # print(features_per_model[0].size())


if __name__ == "__main__":

    config_path = "correlation/other/feature_extraction_config.yaml"
    config_path = "correlation/other/distinct_feat_extr_config.yaml"
    with open(config_path) as fp:
        dataMap = yaml.safe_load(fp)

    config = SceneReconstructionConfig(dataMap)
    main(config)
