# EBES Easy Benchmarking for Event Sequences.

[![arXiv](https://img.shields.io/badge/arXiv-2410.03399-b31b1b.svg)](https://arxiv.org/abs/2410.03399)
[![Docs](https://badgen.net/static/Docs/EBES/green)](https://on-point-rnd.github.io/EBES/)

EBES is an easy-to-use development and application toolkit for Event Sequence(EvS) Assesment, with key features in configurability, compatibility and reproducibility. We hope this project could benefit both researchers and practitioners with the goal of easily customized development and open benchmarking in EvS.


## Setup
### Installation
To install the latest stable version:
```
pip install ebes
```
### Datasets
| Dataset         | Source Link                                                                 | Preprocessing Script Link                                                                 | Download Instructions                                                                                               |
|-----------------|----------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| Physionet2012   | [Physionet2012](https://physionet.org/content/challenge-2012/1.0.0/)        | [physionet2012.py](https://github.com/On-Point-RND/EBES/blob/main/preprocess/physionet2012.py) | Straightforward download on site                                                                                    |
| MIMIC-III       | [MIMIC-III](https://physionet.org/content/mimiciii/1.4/)                    | [mimic-3.py](https://github.com/On-Point-RND/EBES/blob/main/preprocess/mimic-3.py)        | Only credentialed users who sign the DUA can access the files.                                                      |
| Age             | [Age](https://ods.ai/competitions/sberbank-sirius-lesson/data)             | [age.py](https://github.com/On-Point-RND/EBES/blob/main/preprocess/age.py)                | [Download here](https://storage.yandexcloud.net/datasouls-competitions/sirius/data.zip) if you have difficulties navigating site |
| Retail          | [Retail](https://ods.ai/competitions/x5-retailhero-uplift-modeling/data)   | [x5-retail.py](https://github.com/On-Point-RND/EBES/blob/main/preprocess/x5-retail.py)    | [Download here](https://storage.yandexcloud.net/datasouls-ods/materials/9c6913e5/retailhero-uplift.zip) if you have difficulties navigating site |
| MBD             | [MBD](https://huggingface.co/datasets/ai-lab/MBD)                           | [mbd.py](https://github.com/On-Point-RND/EBES/blob/main/preprocess/mbd.py)                | Straightforward download on site                                                                                    |
| Taobao          | [Taobao](https://tianchi.aliyun.com/dataset/46)                             | [taobao.py](https://github.com/On-Point-RND/EBES/blob/main/preprocess/taobao.py)          | Need to login on site to download. After that pass `tianchi_mobile_recommend_train_user.csv` into script           |
| BPI17           | [BPI17](https://data.4tu.nl/articles/_/12696884/1)                          | [bpi_17.py](https://github.com/On-Point-RND/EBES/blob/main/preprocess/bpi_17.py)          | Straightforward download on site                                                                                    |
| ArabicDigits    | [ArabicDigits](https://www.timeseriesclassification.com/description.php?Dataset=SpokenArabicDigits) | [SpokenArabicDigits.py](https://github.com/On-Point-RND/EBES/blob/main/preprocess/SpokenArabicDigits.py) | Either just run preprocessing script and it will download automatically, or straightforward download on site       |
| ElectricDevices | [ElectricDevices](https://www.timeseriesclassification.com/description.php?Dataset=ElectricDevices) | [electric_devices.py](https://github.com/On-Point-RND/EBES/blob/main/preprocess/electric_devices.py) | Straightforward download on site                                                                                    |
| Pendulum        | We created it ourselves                                                     | [pendulum.py](https://github.com/On-Point-RND/EBES/blob/main/preprocess/pendulum.py)      | Run preprocessing script in order to generate from scratch. Make sure to keep default `seed=0` in order to get exactly same dataset. |


## Usage
>python main -d age -m gru -e correlation -s best

### Results:
![image](https://github.com/user-attachments/assets/68532c78-af68-4c78-86e8-f7677fdf635d)



Performance of various models as a function of number of sequences. Metrics from Table 1 are reported. Number of sequences is presented in log scale. Standard deviation across 3 runs is depicted as vertical lines.

<img src="https://arxiv.org/html/2410.03399v1/x5.png" width="500">

Performance metric relationships and correlations of different subsets among all methods on PhysioNet2012 are presented. We do not observe a correlation between the test metric and train-val on PhysioNet2012, as seen in the right upper corner. For the Taobao dataset, we do not observe a clear linear trend between hpo-val and the test metric suggesting the presence of distribution shift.

<img src="https://arxiv.org/html/2410.03399v1/x11.png">
