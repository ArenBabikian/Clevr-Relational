from correlation.gnn.dataset.graph_dataset import SceneGraphDataset
  
# This will create the dataset file if it does not already exist
dataset = SceneGraphDataset(root='_datasets/SCENES2', obj_ann_path="_clevr_gnn/CLEVR_mini_coco_anns.json", schema_path="_clevr_gnn/clevr_attr_map.json", data_to_process="features")

print(dataset)
print(dataset.data)