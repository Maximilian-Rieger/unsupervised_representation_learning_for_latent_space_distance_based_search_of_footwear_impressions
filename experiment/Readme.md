## Pytorch impress 

- Impress is used as dataset
- Data is located in /data/public/impress schuhe+spezial
- for setting up pycharm see https://smithers.cvl.tuwien.ac.at/deeplearning/deep-learning-resources

## train network
- copy config.py to myconfig.py and adjust values
    - log_dir is created in home directory
    - datasets_dir: directory of datasets
- data_zoo.py
    - training/test dataset is defined here
    - allows switching dataset for training/evaluation
    - ImageDataset - loads data (images + image labels)

- main.py
    - starts experiment
    - configures GPU/CPU + logging paths

- experiment.py
    - training + evaluation

## utils
- Tensorboardlogger