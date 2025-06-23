### The Scheme with DA
The proposed method.
#### 1) SVHN to MNIST
```bash
$ python main_svhn_mnist.py --use_labels=True --use_reconst_loss=False
```

### The Scheme without DA
One benchmark, where the newly observed data is directly transmitted without retraining the coders.
#### 1) SVHN
```bash
$ python main_only_svhn.py --use_labels=True --use_reconst_loss=False
```
