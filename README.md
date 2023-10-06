# log6309e-group

## Setup

1. Create a parent folder and clone this repo

```
mkdir log6309
cd log6309 && git clone git@github.com:34yu34/log6309e-group.git
```

2. Download the HDFS and BGL datasets from [here](https://zenodo.org/record/8196385), unzip them, and put them in the `/data` folder.

3. Install and clone dependencies in the parent folder

```
conda create -n logpai python=3.8
conda activate logpai
pip install logparser3 torch tqdm numpy scikit-learn pandas
git clone https://github.com/simonchamorro/loglizer.git
git clone https://github.com/logpai/deep-loglizer.git
```

## Parsing

Run the parsing scripts from this repo's root
```
cd log6309e-group
python scripts/01-parse_data_bgl.py
python scripts/01-parse_data_hdfs.py
```

## Running classification models

```
python scripts/02-svm-hdfs.py
python scripts/02-svm-bgl.py
python scripts/03-dt-hdfs.py
python scripts/03-dt-bgl.py
```

## Running deep learning models
Run the preprocessing scripts to generate datasets, then run the model training and eval scripts
```
python scripts/04-preprocess-bgl.py
python scripts/04-preprocess-hdfs.py
python scripts/05-cnn.py --dataset="HDFS" --data_dir="data/HDFS_v1/hdfs_1.0_tar"
python scripts/05-cnn.py --dataset="BGL" --data_dir="data/BGL/bgl_1.0_tar"
python scripts/06-lstm.py --dataset="HDFS" --data_dir="data/HDFS_v1/hdfs_1.0_tar"
python scripts/06-lstm.py --dataset="BGL" --data_dir="data/BGL/bgl_1.0_tar"
```
