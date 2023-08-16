# Variational Information Pursuit for Interpretable Predictions
**Aditya Chattopadhyay, Kwan Ho Ryan Chan, Benjamin D. Haeffele, Donald Geman, René Vidal** <br>
***Mathematical Institute for Data Science, Johns Hopkins University*** <br>
**`{achatto1, kchan49, bhaeffele, geman, rvidal}jhu.edu`**

This is the offical repository for *Variational Information Pursuit for Interpretable Predictions (ICLR 2023)*.  For our paper, please visit [link](https://arxiv.org/abs/2302.02876).


## Overview
<p align="center">
<img src="./assets/teaser.png" alt="teaser.png" width="500"/>
</p>

There is a growing interest in the machine learning community in developing
predictive algorithms that are “interpretable by design”. Towards this end, recent work proposes to make interpretable decisions by sequentially asking interpretable queries about data until a prediction can be made with high confidence based on the answers obtained (the history). To promote short query-answer
chains, a greedy procedure called Information Pursuit (IP) is used, which adaptively chooses queries in order of information gain *(See Figure above)*. Generative models are employed to learn the distribution of query-answers and labels, which is in turn used to estimate the most informative query. However, learning and inference with a
full generative model of the data is often intractable for complex tasks. In this work, we propose Variational Information Pursuit (V-IP), a variational characterization of IP which bypasses the need for learning generative models. V-IP is based on finding a query selection strategy and a classifier that minimizes the expected cross-entropy between true and predicted labels. We then demonstrate that the IP strategy is the optimal solution to this problem. Therefore, instead of learning generative models, we can use our optimal strategy to directly pick the most informative query given any history. We then develop a practical algorithm by defining a finite-dimensional parameterization of our strategy and classifier using deep networks and train them end-to-end using our objective. A pipeline of our framework is shown below.
<p align="center">
<img src="./assets/pipeline.png" alt="pipeline" width="450"/>
</p>

## Requirements
Please check out `requirements.txt` for detailed requirements. Overall, our code uses basic operations and do not require the latest version of PyTorch or CUDA to work. We also use `wandb` to moderate training and testing performance. One may remove lines related to `wandb` and switch to other packages if they desire. 


## Training MNIST
There are two stages of training: *Initial Random Sampling (IRS)* and *Subsequent Biased Sampling (SBS)*.

To run IRS:

```
python3 main_mnist.py \
  --epochs 100 \
  --data mnist \
  --batch_size 128 \
  --max_queries 676 \
  --max_queries_test 21 \
  --lr 0.0001 \
  --tau_start 1.0 \
  --tau_end 0.2 \
  --sampling random \
  --seed 0 \
  --name mnist_random
```

To run SBS:

```
python3 main_mnist.py \
  --epochs 20 \
  --data mnist \
  --batch_size 128 \
  --max_queries 21 \
  --max_queries_test 21 \
  --lr 0.0001 \
  --tau_start 0.2 \
  --tau_end 0.2 \
  --sampling biased \
  --seed 0 \
  --ckpt_path <CKPT_PATH> \
  --name mnist_biased
```
where `<CKPT_PATH>` is the path to the pre-trained model using IRS.

## Checkpoints
Checkpoint to the models used to obtain the results in our paper are listed in the table below. A jupyter notebook named `loading.ipynb` with checkpoint loading instructions for each dataset. is located in `pretrain/`. One may put downloaded models in this directory.


| Dataset | OneDrive Link |
| :---: | :-------: |
| MNIST | [Link](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/kchan49_jh_edu/Ec1de2HcJ9dMuT9ScOhFsfcBeZ25A55rAo7lkdUMQpQoMg?e=PFvayh) |
| KMNIST | [Link](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/kchan49_jh_edu/EX5CH3HbXA5Eo1yIC7JMswAB5GaanEcRBDtd-kSjHOCEXw?e=UdHQYi) |
| Fashion MNIST | [Link](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/kchan49_jh_edu/EbocJetI_vpNmMZ0w33cQHIBGiH8_nxOT75YbfjP5ma47g?e=kAEevd) |
| Huffington News | [Link](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/kchan49_jh_edu/ETe8rbzfY0BKh5EmHe-mx-sBRRd_1BROEHEJU58O57my3g?e=7tXnnn) |
| CUB-200 | [Link](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/kchan49_jh_edu/Eda0xGUGQ39Kl1d4LACN2agByKqByRMM0QZm6Rnibq4gBw?e=dXCcpw) |
| CUB-200 (concept) | [Link](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/kchan49_jh_edu/Ef3cdrFhegRFuqePkJJvOk0Bacw_lkh4iWl8rXECb7UrxA?e=V64Q5V) |  
| CIFAR10 | [Link](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/kchan49_jh_edu/ES_orEvtEc9Kjw4u1wgfiC8BvH7Y_6kaNVs-ZWvPqLcwjA?e=7a4Ylc) |
| SymCAT200 | [Link](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/kchan49_jh_edu/Edi9NVj6171DpfX4hgpJH3MB6xHxke2j7XRCunZKmb_CUw?e=SdTKxO) |
| SymCAT300 | [Link](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/kchan49_jh_edu/ERgvhBjxLj9GodXjFzGZTAAB4j0TP0EWd7EL1ZqL9eA_kQ?e=MAxemp) |
| SymCAT400 | [Link](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/kchan49_jh_edu/EbBJ8nDEk8dMmc6SXdh9rp0By5XG3Gf8Z0wLD8zHaJcXRw?e=gmJE68) |


## License
This project is under the MIT License. See [LICENSE](./LISENSE.md) for details.


## Cite
If you find our work useful for your research, please cite:

```
@article{chattopadhyay2023variational,
  title={Variational Information Pursuit for Interpretable Predictions},
  author={Chattopadhyay, Aditya and Chan, Kwan Ho Ryan and Haeffele, Benjamin D and Geman, Donald and Vidal, Ren{\'e}},
  journal={arXiv preprint arXiv:2302.02876},
  year={2023}
}
```

