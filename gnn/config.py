class SceneReconstructionConfig:
    def __init__(self, config_dict):
        # obj_ann_path, schema_path, train_size, num_workers, num_rels, dropout_p, use_sigmoid, edge_dim, batch_size=1
        for key, value in config_dict.items():
            setattr(self, key, value)
