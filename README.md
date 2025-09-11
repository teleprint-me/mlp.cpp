# mlp

A multi-layer perceptron.

## Dependencies

- g++
- gdb
- cmake
- clangd
- clang-format
- clib
- libmath
- openmp

## Setup

```sh
git clone https://github.com/teleprint-me/mlp.cpp mlp
cd mlp
```

## Build

### Debug

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Debug
```

### Release

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
```

### Compile

```sh
cmake --build build -j 16
```

### Model Checkpoints

```sh
mkdir models
```

## XOR Gate

The XOR gate is a canonical non-linear test bed—an ideal problem for
multi-layer perceptrons.

There are three binaries for the XOR model:

- `train`: Train an XOR model from scratch or resume from a checkpoint.
- `inspect`: View model dimensions and internal state.
- `inference`: Run predictions using a trained model.

Executables are located in `./build/xor`.

### Training

- To train from scratch:

```sh
./build/xor/xor_train
```

- To resume from an existing checkpoint:

```sh
./build/xor/xor_train --ckpt xor-latest.bin
```

- For available options:

```sh
./build/xor/xor_train -h
```

#### Organizing Checkpoints

To group checkpoints in a directory:

```sh
./build/xor/xor_train --ckpt models/xor-latest.bin
```

### Inspect

```sh
./build/xor/xor_inspect --ckpt xor-latest.bin
```

### Inference

```sh
./build/xor/xor_inference --ckpt xor-latest.bin
```

## MNIST

The MNIST dataset provides a classic benchmark for handwritten digit
classification using MLPs.

### Setup

#### Download (optional)

```sh
wget 'https://huggingface.co/datasets/teleprint-me/mnist/resolve/main/mnist.tar.gz?download=true' -O mnist.tar.gz
wget 'https://huggingface.co/datasets/teleprint-me/mnist/resolve/main/sha256sum.txt?download=true' -O sha256sum.txt
```

#### Validate

```sh
sha256sum -c sha256sum.txt  # Should output: mnist.tar.gz: OK
```

#### Extract

```sh
mkdir -p data
tar xvf mnist.tar.gz
mv mnist.tar.gz mnist/ data/
```

This creates `data/mnist/training/0` ... `9` and `data/mnist/testing/0` ...
`9`.

#### Dataset quick stats

- **Total images:** 70,000 (PNG, 28×28, 8-bit grayscale)
- **Training:** 60,000 images
- **Testing:** 10,000 images
- **Disk usage:** \~276MB
- **Layout:**
  - `data/mnist/training/0-9/`
  - `data/mnist/testing/0-9/`

#### Data inspection (optional)

```sh
file mnist/training/0/10005.png
identify mnist/training/0/10005.png   # Needs ImageMagick
```

### Training & Evaluation

#### Train

```sh
./build/mnist/mnist_train [options]
```

For example:

```sh
./build/mnist/mnist_train --samples 1000 --layers 5 --hidden 32
```

All options:

```sh
./build/mnist/mnist_train --help
```

Checkpoints are saved as `models/mnist-latest.bin` and as timestamped
snapshots per epoch.

#### Evaluate

```sh
./build/mnist/mnist_eval --ckpt models/mnist-latest.bin --data data/mnist/testing --samples 1000
```

## References

- [1957 — The Perceptron: A probabilistic model for information storage and organization in the brain](https://archive.org/details/sim_psychological-review_1958-11_65_6/page/386/mode/2up?q=the+perceptron+rosenblatt+1957)
- [1986 — Learning representations by back-propagating errors](https://www.semanticscholar.org/paper/Learning-representations-by-back-propagating-errors-Rumelhart-Hinton/052b1d8ce63b07fec3de9dbb583772d860b7c769)
- [1989 — Multilayer feedforward networks are universal approximators](https://www.semanticscholar.org/paper/Multilayer-feedforward-networks-are-universal-Hornik-Stinchcombe/f22f6972e66bdd2e769fa64b0df0a13063c0c101)
- [1998 — Gradient-based learning applied to document recognition](https://www.semanticscholar.org/paper/Gradient-based-learning-applied-to-document-LeCun-Bottou/162d958ff885f1462aeda91cd72582323fd6a1f4)
- [2010 — Understanding the difficulty of training deep feedforward neural networks](https://www.semanticscholar.org/paper/Understanding-the-difficulty-of-training-deep-Glorot-Bengio/ea9d2a2b4ce11aaf85136840c65f3bc9c03ab649)
- [2013 — On the importance of initialization and momentum in deep learning](https://www.semanticscholar.org/paper/On-the-importance-of-initialization-and-momentum-in-Sutskever-Martens/aa7bfd2304201afbb19971ebde87b17e40242e91)
- [2016 — Deep Learning](https://www.deeplearningbook.org)
