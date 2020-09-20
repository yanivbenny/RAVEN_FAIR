# RAVEN-FAIR
Balanced RAVEN dataset from the paper: 'Scale-Localized Abstract Reasoning'.

[Paper](https://github.com/yanivbenny/MRNet) [Code](https://github.com/yanivbenny/MRNet)

## Requirements
Tested on both linux and windows 10.
* python 3.6
* eventlet
* tqdm
* numpy
* scipy
* opencv-python
* pillow


## Generating the dataset
To create the dataset, run:
```
python main.py --fair FAIR --save-dir DEST
```
* FAIR - bool, (0,1) generate the original RAVEN dataset or RAVEN-FAIR. default: 0.
* DEST - str, the destination of the directory to save the data. default: ./Datasets/

Original RAVEN will be created at \<DEST\>/RAVEN.
RAVEN-FAIR will be created at \<DEST\>/RAVEN-F. 
## Downloading the dataset
Upcoming.


## Acknowledgement
We thank the original creators of the RAVEN dataset:
Chi Zhang, Feng Gao, Baoxiong Jia, Yixin Zhu, Song-Chun Zhu.
The original code can be found at the repository: [RAVEN](https://github.com/WellyZhang/RAVEN).
