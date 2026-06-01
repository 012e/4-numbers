"""Command-line training and inference helpers for network3.py."""

import argparse
import gzip
import pickle
from pathlib import Path

import numpy as np
import theano

import network3
from network3 import ConvPoolLayer, FullyConnectedLayer, Network, ReLU, SoftmaxLayer


ARCHITECTURES = ("shallow", "basic-conv", "dbl-conv-relu", "post")
ROOT = Path(__file__).resolve().parent.parent


def build_network(architecture, mini_batch_size, fc_neurons=100, dropout=0.5):
    if architecture == "shallow":
        return Network(
            [
                FullyConnectedLayer(n_in=784, n_out=fc_neurons),
                SoftmaxLayer(n_in=fc_neurons, n_out=10),
            ],
            mini_batch_size,
        )

    if architecture == "basic-conv":
        return Network(
            [
                ConvPoolLayer(
                    image_shape=(mini_batch_size, 1, 28, 28),
                    filter_shape=(20, 1, 5, 5),
                    poolsize=(2, 2),
                ),
                FullyConnectedLayer(n_in=20 * 12 * 12, n_out=fc_neurons),
                SoftmaxLayer(n_in=fc_neurons, n_out=10),
            ],
            mini_batch_size,
        )

    if architecture == "dbl-conv-relu":
        return Network(
            [
                ConvPoolLayer(
                    image_shape=(mini_batch_size, 1, 28, 28),
                    filter_shape=(20, 1, 5, 5),
                    poolsize=(2, 2),
                    activation_fn=ReLU,
                ),
                ConvPoolLayer(
                    image_shape=(mini_batch_size, 20, 12, 12),
                    filter_shape=(40, 20, 5, 5),
                    poolsize=(2, 2),
                    activation_fn=ReLU,
                ),
                FullyConnectedLayer(
                    n_in=40 * 4 * 4, n_out=fc_neurons, activation_fn=ReLU
                ),
                SoftmaxLayer(n_in=fc_neurons, n_out=10),
            ],
            mini_batch_size,
        )

    if architecture == "post":
        return Network(
            [
                ConvPoolLayer(
                    image_shape=(mini_batch_size, 1, 28, 28),
                    filter_shape=(20, 1, 5, 5),
                    poolsize=(2, 2),
                    activation_fn=ReLU,
                ),
                ConvPoolLayer(
                    image_shape=(mini_batch_size, 20, 12, 12),
                    filter_shape=(40, 20, 5, 5),
                    poolsize=(2, 2),
                    activation_fn=ReLU,
                ),
                FullyConnectedLayer(
                    n_in=40 * 4 * 4,
                    n_out=1000,
                    activation_fn=ReLU,
                    p_dropout=dropout,
                ),
                FullyConnectedLayer(
                    n_in=1000,
                    n_out=1000,
                    activation_fn=ReLU,
                    p_dropout=dropout,
                ),
                SoftmaxLayer(n_in=1000, n_out=10, p_dropout=dropout),
            ],
            mini_batch_size,
        )

    raise ValueError("unknown architecture: {}".format(architecture))


def save_model(path, net, args):
    model = {
        "architecture": args.architecture,
        "mini_batch_size": args.mini_batch_size,
        "fc_neurons": args.fc_neurons,
        "dropout": args.dropout,
        "params": [param.get_value(borrow=True) for param in net.params],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(str(path), "wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path):
    with gzip.open(str(path), "rb") as handle:
        model = pickle.load(handle)
    net = build_network(
        model["architecture"],
        model["mini_batch_size"],
        model.get("fc_neurons", 100),
        model.get("dropout", 0.5),
    )
    for param, value in zip(net.params, model["params"]):
        param.set_value(value.astype(theano.config.floatX), borrow=True)
    return net, model


def image_to_mnist_vector(path, invert="auto"):
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "Pillow is required for image prediction. Add `--with pillow` to uv."
        ) from exc

    image = Image.open(path).convert("L").resize((28, 28))
    values = np.asarray(image, dtype=theano.config.floatX) / 255.0
    if invert == "yes" or (invert == "auto" and values.mean() > 0.5):
        values = 1.0 - values
    return values.reshape(1, 784)


def train(args):
    training_data, validation_data, test_data = network3.load_data_shared(args.data)
    net = build_network(
        args.architecture, args.mini_batch_size, args.fc_neurons, args.dropout
    )
    net.SGD(
        training_data,
        args.epochs,
        args.mini_batch_size,
        args.eta,
        validation_data,
        test_data,
        lmbda=args.lmbda,
    )
    save_model(args.output, net, args)
    print("Saved model to {}".format(args.output))


def predict(args):
    net, model = load_model(args.model)
    image = image_to_mnist_vector(args.image, args.invert)
    batch = np.repeat(image, model["mini_batch_size"], axis=0).astype(
        theano.config.floatX
    )
    predict_batch = theano.function([net.x], net.layers[-1].y_out)
    prediction = int(predict_batch(batch)[0])
    print(prediction)


def parser():
    root = argparse.ArgumentParser(
        description="Train network3.py models and predict digit images."
    )
    subparsers = root.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument(
        "--architecture", choices=ARCHITECTURES, default="dbl-conv-relu"
    )
    train_parser.add_argument("--data", default=str(ROOT / "data" / "mnist.pkl.gz"))
    train_parser.add_argument("--epochs", type=int, default=60)
    train_parser.add_argument("--mini-batch-size", type=int, default=10)
    train_parser.add_argument("--eta", type=float, default=0.03)
    train_parser.add_argument("--lmbda", type=float, default=0.1)
    train_parser.add_argument("--fc-neurons", type=int, default=100)
    train_parser.add_argument("--dropout", type=float, default=0.5)
    train_parser.add_argument(
        "--output", default=str(ROOT / "models" / "network3.pkl.gz")
    )
    train_parser.set_defaults(func=train)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--model", required=True)
    predict_parser.add_argument("--image", required=True)
    predict_parser.add_argument(
        "--invert", choices=("auto", "yes", "no"), default="auto"
    )
    predict_parser.set_defaults(func=predict)
    return root


def main():
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
