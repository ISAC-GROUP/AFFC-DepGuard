# DepGuard

This is the PyTorch implementation of ***DepGuard*** for depression recognition and episodes monitoring. 

Yufei Zhang, Shuo Jin, Wenting Kuang, Yuda Zheng, Qifeng Song, Changhe Fan, Yongpan Zou, Victor C. M. Leung, Kaishun Wu. "DepGuard: Depression Recognition and Episode Monitoring System with A Ubiquitous Wrist-worn Device", just accepted as a regular paper in ***IEEE Transactions on Mobile Computing***.



## Dataset 📖

Regarding the dataset, we collected multimodal physiological signals from 17 depressed patients as well as 18 normal subjects, including heart rate variability, blood oxygen saturation, galvanic skin response (IR and IRed), and skin temperature signals. Data on depressed patients were collected from the Second People's Hospital of Guangdong Province, and data on normal subjects were collected from students recruited from Shenzhen University. **Our data paper is being submitted to other journal, and we will open access to the dataset after it is published.** 

| Signal type             | Sensor model | Sampling rate | Data type                                   |
| ----------------------- | ------------ | ------------- | ------------------------------------------- |
| Heart rate              | Pulse sensor | 400 Hz        | Raw analog data                             |
| Blood oxygen saturation | Max30102     | 400 Hz        | Red light and infrared light intensity data |
| Galvanic skin response  | Grove GSR    | 200 Hz        | Voltage simulation data                     |
| Skin temperature        | LMT70        | 100 Hz        | Voltage simulation data                     |



## Code 📖

Environment installation:

- `Python` 3.8

```shell
conda create -n depguard python=3.8
```

```shell
conda activate depguard
```

Before running our code, please install the following packages:

```
os
seaborn
matplotlib
numpy
tqdm
torch
torchvision
scikit-learn
pandas
multiprocessing
```



## Citation 🖊️

If you find our work useful, please consider citing our paper:

```
@ARTICLE{zhang2025depguard,
  author={Zhang, Yufei and Jin, Shuo and Kuang, Wenting and Zheng, Yuda and Song, Qifeng and Fan, Changhe and Zou, Yongpan and Leung, Victor C. M. and Wu, Kaishun},
  journal={IEEE Transactions on Mobile Computing}, 
  title={DepGuard: Depression Recognition and Episode Monitoring System With a Ubiquitous Wrist-Worn Device}, 
  year={2026},
  volume={25},
  number={1},
  pages={197-214},
  doi={10.1109/TMC.2025.3591096}
}
```

## Acknowledgement ✉️

If you are interested in our collection of ubiquitous wearable devices, please contact  [Prof. Yongpan Zou](https://yongpanzou.github.io/), College of Computer Science and Software Engineering, Shenzhen University.
