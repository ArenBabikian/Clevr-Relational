
### Installation
Need to add pyg to the conda environment of this repo

```
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cpu.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+cpu.html
pip install torch-geometric
```
___

### Execution

To extract neuron values and features from each image, and to save the raw data:
```
python correlation/other/findCorrelation.py --config_fp _results_mini_coco/sample_config.yaml
```

To collect the raw neuron values or feature values within an InMemoryDataSet:
```
python correlation/other/produce_in_memory_dataset.py
```

To train the NN:
```
python correlation/gnn/train.py --config_fp=correlation/gnn/config.yaml
```

To run the TensorBoard
```
python -m tensorboard.main --logdir=correlation/gnn/_output/logs/default/version_<#>
```
