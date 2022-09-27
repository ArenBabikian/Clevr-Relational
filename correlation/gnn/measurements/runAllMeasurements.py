import yaml
from correlation.gnn.config import SceneReconstructionConfig
from correlation.gnn.train import main

config_fp = "correlation/gnn/measurements/default_config.yaml"

def runTraining(encoder, intermediate_gt):
    # For the 4000 images
    with open(config_fp) as f:
        dataMap = yaml.safe_load(f)

    dataMap['encoder'] = encoder
    dataMap['intermediate_gt'] = intermediate_gt

    config = SceneReconstructionConfig(dataMap)
    main(config)


if __name__ == '__main__':
    runTraining("gat", "features")
    # runTraining("rgcn2", "features")
    # runTraining("gat", "random-s1")
    # runTraining("rgcn", "random-s1")
    # runTraining("gat", "random-s2")
    # runTraining("rgcn", "random-s2")
    # runTraining("gat", "random-s3")
    # runTraining("rgcn", "random-s3")

