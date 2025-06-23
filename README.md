# DepGuard

This is the PyTorch implementation of ***DepGuard*** for depression recognition and episodes monitoring. 

Yufei Zhang, Shuo Jin, Wenting Kuang, Yuda Zheng, Qifeng Song, Changhe Fan, Yongpan Zou, Victor C. M. Leung, Kaishun Wu. "DepGuard: Depression Recognition and Episode Monitoring System with A Ubiquitous Wrist-worn Device", just accepted as a regular paper in ***IEEE Transactions on Mobile Computing***.



## Dataset 📖

Regarding the dataset, we collected multimodal physiological signals from 17 depressed patients as well as 18 normal subjects, including heart rate variability, blood oxygen saturation, galvanic skin response (IR and IRed), and skin temperature signals. Data on depressed patients were collected from the Second People's Hospital of Guangdong Province, and data on normal subjects were collected from students recruited from Shenzhen University. **Our data paper is being submitted to the journal, and we will open access to the dataset after it is published.** 

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
@inproceedings{zhang2025depguard,
  title={DepGuard: Depression Recognition and Episode Monitoring System with A Ubiquitous Wrist-worn Device},
  author={Yufei Zhang, Shuo Jin, Wenting Kuantg, Yuda Zheng, Qifeng Song, Changhe Fan, Yongpan Zou, Victor C. M. Leung, Kaishun Wu},
  journal={IEEE Transactions on Mobile Computing},
  pages={},
  year={2025},
  organization={IEEE Computer Society}
}
```

## Acknowledgement ✉️

If you are interested in our collection of ubiquitous wearable devices, please contact  [Yongpan Zou Prof.](https://yongpanzou.github.io/), College of Computer Science and Software Engineering, Shenzhen University.
