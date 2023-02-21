import yaml
from correlation.gnn.config import SceneReconstructionConfig
from correlation.gnn.train import main

config_fp = "correlation/gnn/measurements/default_config.yaml"
git_path = '/home/arenbabikian/git'

def runTraining(encoder, dataset, target_type, min_id, max_id, iep_answers=None):
    # For the 4000 images
    with open(config_fp) as f:
        dataMap = yaml.safe_load(f)

    dataMap['encoder'] = encoder
    dataMap['source_type'] = dataset # (dataset_name)
    dataMap['target_type'] = target_type # (target_type)
    dataMap['min_id'] = min_id
    dataMap['max_id'] = max_id
    dataMap['iep_answer_details'] = iep_answers

    # DATASET specification
    
    if dataset == 'IEPVQA':
        # scenes: 0..15000
        # features: 0:3000
        dataMap['obj_ann_path'] = f"{git_path}/Clevr-Relational/_data/iepvqa/CLEVR_val_scenes.json"
        dataMap['target_path'] = f"{git_path}/Clevr-Relational/_data/iepvqa/val_features_0_3000.h5"
        dataMap['global_max_id'] = 3000
    elif dataset == "CLEVR-GNN":
        # 0..4000
        # NOTE: only 'features' works. Cannot do random selection since y size changes between images
        dataMap['obj_ann_path'] = f"{git_path}/Clevr-Relational/_data/clevr-gnn/CLEVR_mini_coco_anns.json"
        dataMap['target_path'] = f'{git_path}/Clevr-Relational/_data/clevr-gnn/features'


    # TODO wip
    elif dataset == 'IEPVQA-Q':
        # scenes: 0..15000
        # features: 0:3000
        dataMap['obj_ann_path'] = f"{git_path}/Clevr-Relational/_data/iepvqa/CLEVR_val_scenes.json"
        # below comes from the clevr-iep repo
        dataMap['questions_path'] = f"{git_path}/Clevr-Relational/_data/iepvqa/val_questions_0_3000.h5"
        dataMap['answers_path'] = f"{git_path}/Clevr-Relational/_data/iepvqa/answers/resnet101/700k_strong_0_3000__1.h5"
        dataMap['vocab_path'] = f"{git_path}/Clevr-Relational/_data/iepvqa/val_vocab_0_3000.json"
        dataMap['decoder'] = '700k_strong'

        dataMap['dataset_save_path'] = f"{git_path}/Clevr-Relational/_data/iepvqa/saved_old"
        dataMap['handle_scores'] = False
        dataMap['clevr_iep_path'] = f'{git_path}/clevr-iep/'

        dataMap['global_max_id'] = 3000



    # TODO missing
    elif dataset == 'IEPVQA-DIS':
        dataMap['obj_ann_path'] = f"{git_path}/clevr-dataset-gen/clevr-distinct/scenes.json"
        dataMap['target_path'] = f"{git_path}/clevr-iep/data/distinct/k-resnet101_0_1000.h5"
        dataMap['global_max_id'] = 1000
    # elif dataset == "CLEVR-S":
    #     # 0..1000
    #     dataMap['obj_ann_path'] = f"{git_path}/Clevr-Relational/_data/clevr-small/scenes.json"
    #     dataMap['target_path'] = f'{git_path}/Clevr-Relational/_data/clevr-small/features'
    else:
        print("Dataset root directory not")
        exit(1)
    
    
    dataMap['train_size'] = None if max_id == None else int(0.9*(max_id-min_id))


    config = SceneReconstructionConfig(dataMap)
    main(config)


if __name__ == '__main__':
    # TODO clean this up
    # IEPVQA: "{scores, results}-{features, random, randomwithreplacement}"
    # print(torch.load("_datasets/IEPVQA/processed/data-features.pt")[0])
    # exit()

    runTraining("gatiep", "IEPVQA-Q", "features", 0, 150)
    exit()

    runTraining("gatiep", "IEPVQA", "features", 0, 3000)
    runTraining("gatiep", "IEPVQA", "random", 0, 3000)
    runTraining("gatiep", "IEPVQA", "randomwithreplacement", 0, 3000)
    
    runTraining("gat", "CLEVR-GNN", "features", 0, 3000)
    runTraining("rgcn", "CLEVR-GNN", "features", 0, 3000)
    runTraining("rgcn2", "CLEVR-GNN", "features", 0, 3000)

    # for e in ['gatiep']:
    #     for d in ['IEPVQA']:
    #         for t in ['features', 'random', 'randomwithreplacement']:
    #             runTraining(e, d, t, 0, 3000)
    # for e in ['gatiep']:
    #     for d in ['IEPVQA']:
    #         for t in ['random', 'randomwithreplacement']:
    #             for _ in range(3):
    #                 runTraining(e, d, t, 0, 3000)
    

    # for e in ['gat', 'rgcn', 'rgcn2']:
    #     for d in ['CLEVR-GNN']:
    #         # for t in ['features', 'random', 'randomwithreplacement']:
    #         for t in ['features']:
    #             runTraining(e, d, t, 0, 4000)

    # runTraining("rgcn2", "CLEVR-GNN", "features", 0, 4000)
    # runTraining("rgcn", "IEPVQA", "features", 0, 10000)
    # runTraining("gatiep", "IEPVQA", "randomwithreplacement", 0, 1000)
    # runTraining("gatiep", "IEPVQA", "random", 0, 1000)
    # runTraining("gatiep", "IEPVQA-QA", "scores-features", 0, 300, ["9k", 0, 300, 1])
    # runTraining("gatiep", "IEPVQA-DIS", "features", 0, 1000)
    # runTraining("gatiep", "IEPVQA-DIS", "randomwithreplacement1", 0, 1000)    
    # runTraining("gatiep", "IEPVQA-DIS", "randomwithreplacement1", 0, 3000)

