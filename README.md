# Official Implementation for Variational Information Pursuit for Interpretable Predictions (2022)
##### Authors: Aditya Chattopadhyay, Kwan Ho Ryan Chan, Benjamin D. Haeffele, Donald Geman, René Vidal


## Requirements


## Training
There are two stages of training: *Initial Random Sampling (IRS)* and *Subsequent Biased Sampling (SBS)*.

To run IRS:

```
python3 main_mnist.py \
  --epochs 100 \
  --data mnist \
  --batch_size 128 \
  --max_queries 676 \
  --max_queries_test 21 \
  --lr 0.05 \
  --tau_start 1.0 \
  --tau_end 0.2 \
  --sampling random \
  --seed 0 \
  --name mnist_random
```

To run SBS:

```
python3 main_mnist.py \
  --epochs 100 \
  --data mnist \
  --batch_size 128 \
  --max_queries 21 \
  --max_queries_test 21 \
  --lr 0.05 \
  --tau_start 0.2 \
  --tau_end 0.2 \
  --sampling biased \
  --seed 0 \
  --ckpt_path <CKPT_PATH> \
  --name mnist_biased
```
where `<CKPT_PATH>` is the path to the pre-trained model using IRS.
## License



## Cite


