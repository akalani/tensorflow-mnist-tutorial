# tensorflow-mnist-tutorial

Tutorial code for training on MNIST dataset using high-level APIs in TensorFlow.

### Prerequisites

- TensorFlow 1.4.1

### Setup

Download the MNIST dataset and persist as binary files using the following command:

```sh
python3 experiment.py --download_dataset True --data_dir <data_dir>
```

- data_dir: Path where to save the MNIST dataset

Now, run the training and evaluation via the following command:

```sh
python3 experiment.py --batch_size <batch size> --epochs <epochs> --data_dir <data dir> --model_dir <model dir>
```

- data_dir: Path from where to load input data
- model_dir: Path where model params and summaries will be saved
- batch_size: Batch size
- epochs: Number of epochs