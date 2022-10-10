import yaml
from correlation.gnn.config import SceneReconstructionConfig
from correlation.gnn.train import main

config_fp = "correlation/gnn/measurements/default_config.yaml"

def runTraining(encoder, root_dir, intermediate_gt, min_id, max_id):
    # For the 4000 images
    with open(config_fp) as f:
        dataMap = yaml.safe_load(f)

    dataMap['encoder'] = encoder
    dataMap['dataset_name'] = root_dir
    if root_dir == "SCENES":
        dataMap['obj_ann_path'] = "C:/git/Clevr-Relational/_clevr_data/_clevr_gnn/CLEVR_mini_coco_anns.json"
        dataMap['train_size'] = 3600
    elif root_dir == 'IEPVQA':
        dataMap['obj_ann_path'] = "C:/git/Clevr-Relational/_data/iepvqa/CLEVR_val_scenes.json"
        dataMap['train_size'] = 2700
    else:
        print("Dataset root directory not")
        exit(1)

    dataMap['intermediate_gt'] = intermediate_gt
    dataMap['min_id'] = min_id
    dataMap['max_id'] = max_id

    config = SceneReconstructionConfig(dataMap)
    main(config)


if __name__ == '__main__':
    runTraining("gatiep", "IEPVQA", "randommore1", 0, 3000)
    # runTraining("rgcn", "SCENES", "features", 0, 2000)
    # runTraining("rgcn2", "features")
    # runTraining("gat", "random-s1")
    # runTraining("rgcn", "random-s1")
    # runTraining("gat", "random-s2")
    # runTraining("rgcn", "random-s2")
    # runTraining("gat", "random-s3")
    # runTraining("rgcn", "random-s3")

