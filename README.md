# mlp

A multi-layer perceptron.

## dependencies

- g++
- clangd
- clang-format
- clib
- libmath
- openmp

## setup

```sh
git clone https://github.com/teleprint-me/mlp.cpp mlp
cd mlp
```

## build

### debug

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Debug
```

### release

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
```

### compile

```sh
cmake --build build -j 16
```

### checkpoints

```sh
mkdir models
```

## XOR Gate

The XOR gate is a perfect non-linear test bed and is a classic problem well suited for multi-layer preceptrons.

There are 3 binaries available for the XOR model.

- train: train an xor model from scratch.
- inspect: view the models internal dimensions and state.
- inference: run predictions on a pre-trained model.

Executables can be found in `./build/xor`.

### pre-train

- For training from scratch:

```sh
./build/xor/train --ckpt mlp-latest.bin
```

- Training from a pre-existing checkpoint:

```sh
./build/xor/train --ckpt mlp-latest.bin
```

- For more options, see help text:

```sh
./build/xor/train -h
```

### @todo checkpoints

To group checkpoints, include the directory:

```sh
./build/xor/train --ckpt models/mlp-latest.bin
```

### inspect

```sh
./build/xor/inspect --ckpt mlp-latest.bin
```

### inference

```sh
./build/xor/inference --ckpt mlp-latest.bin
```

## references

- [1957 - The Perceptron: A probabilistic model for information storage and organization in the brain](https://archive.org/details/sim_psychological-review_1958-11_65_6/page/386/mode/2up?q=the+perceptron+rosenblatt+1957)
- [1986 - Learning representations by back-propagating errors](https://www.semanticscholar.org/paper/Learning-representations-by-back-propagating-errors-Rumelhart-Hinton/052b1d8ce63b07fec3de9dbb583772d860b7c769)
- [1989 - Multilayer feedforward networks are universal approximators](https://www.semanticscholar.org/paper/Multilayer-feedforward-networks-are-universal-Hornik-Stinchcombe/f22f6972e66bdd2e769fa64b0df0a13063c0c101)
- [2010 - Understanding the difficulty of training deep feedforward neural networks](https://www.semanticscholar.org/paper/Understanding-the-difficulty-of-training-deep-Glorot-Bengio/ea9d2a2b4ce11aaf85136840c65f3bc9c03ab649)
- [2013 - On the importance of initialization and momentum in deep learning](https://www.semanticscholar.org/paper/On-the-importance-of-initialization-and-momentum-in-Sutskever-Martens/aa7bfd2304201afbb19971ebde87b17e40242e91)
