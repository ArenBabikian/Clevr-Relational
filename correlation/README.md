
### Installation
Need to add pyg to the conda environment of this repo

```
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cpu.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+cpu.html
pip install torch-geometric
```
OR
```
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
```
___
### Neuron value Collection

To extract neuron values and features from each image, and to save the raw data:
```
python correlation/other/findCorrelation.py --config_fp _results_mini_coco/sample_config.yaml
```

To collect the raw neuron values or feature values within an InMemoryDataSet:
```
python correlation/other/produce_in_memory_dataset.py
```
___
### Training

To train the NN, look at the `correlation/gnn/measurements/runAllMeasurements.py` file.

Otherwise, you can manually train the NN by running:
```
python correlation/gnn/train.py --config_fp=correlation/gnn/config.yaml
```

___
### TensorBoard

To run the TensorBoard
```
python -m tensorboard.main --logdir=correlation/gnn/_output/logs/default/version_<#>
```
To run the TensorBoard on the current results for feature learning
```
python -m tensorboard.main --logdir=correlation/gnn/measurements/_out/_official
```
