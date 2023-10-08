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
pip install -r requirements.txt
```