# RAVEN-FAIR
Balanced RAVEN dataset from the paper: 'Scale-Localized Abstract Reasoning', presented at CVPR 2021.

[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Benny_Scale-Localized_Abstract_Reasoning_CVPR_2021_paper.pdf) 
[Code](https://github.com/yanivbenny/MRNet)

## Requirements
Tested on both linux and windows 10.
* python 2.7
* eventlet (windows)
* tqdm
* numpy=1.16.6
* scipy=1.2.3
* opencv-python=4.2.0.32
* pillow=6.2.2


## Generating the dataset
To create the dataset, run:
```
python main.py --fair FAIR --save-dir DEST
```
* FAIR - bool, (0,1) generate the original RAVEN dataset or RAVEN-FAIR. default: 0.
* DEST - str, the destination of the directory to save the data. default: ./Datasets/

Original RAVEN will be created at \<DEST\>/RAVEN.
RAVEN-FAIR will be created at \<DEST\>/RAVEN-F. 


## Acknowledgement
We thank the original creators of the RAVEN dataset:
Chi Zhang, Feng Gao, Baoxiong Jia, Yixin Zhu, Song-Chun Zhu.
The original code can be found at the repository: [RAVEN](https://github.com/WellyZhang/RAVEN).


## Citation
We thank you for showing interest in our work. 
If our work was beneficial for you, please consider citing us using:

```
@inproceedings{benny2021scale,
  title={Scale-localized abstract reasoning},
  author={Benny, Yaniv and Pekar, Niv and Wolf, Lior},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12557--12565},
  year={2021}
}
```

If you have any question, please feel free to contact us.
