# EBES Easy Benchmarking for Event Sequences.
https://arxiv.org/abs/2410.03399




EBES is an easy-to-use development and application toolkit for Event Sequence(EvS) Assesment, with key features in configurability, compatibility and reproducibility. We hope this project could benefit both researchers and practitioners with the goal of easily customized development and open benchmarking in EvS.

## Installation
To install the latest stable version:
```
pip install ebes
```

## Usage
>python main -d age -m gru -e correlation -s best

### Results:



Performance of various models as a function of number of sequences. Metrics from Table 1 are reported. Number of sequences is presented in log scale. Standard deviation across 3 runs is depicted as vertical lines.
<img src="https://arxiv.org/html/2410.03399v1/x5.png">

<img src="https://arxiv.org/html/2410.03399v1/x5.png">


Performance metric relationships and correlations of different subsets among all methods on PhysioNet2012 are presented. We do not observe a correlation between the test metric and train-val on PhysioNet2012, as seen in the right upper corner. For the Taobao dataset, we do not observe a clear linear trend between hpo-val and the test metric suggesting the presence of distribution shift.

<img src="https://arxiv.org/html/2410.03399v1/x2.png">
