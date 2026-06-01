"""Command-line training and inference helpers for network3.py."""

import argparse
import gzip
import pickle
import random
from pathlib import Path

import numpy as np
import theano

import network3
from network3 import ConvPoolLayer, FullyConnectedLayer, Network, ReLU, SoftmaxLayer


ARCHITECTURES = ("shallow", "basic-conv", "dbl-conv-relu", "post")
ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png"}


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

    image = Image.open(path).convert("L")
    values = np.asarray(image, dtype=theano.config.floatX) / 255.0
    if invert == "yes" or (invert == "auto" and values.mean() > 0.5):
        values = 1.0 - values

    ys, xs = np.where(values > 0.05)
    if len(xs) == 0:
        values = np.zeros((28, 28), dtype=theano.config.floatX)
        return values.reshape(1, 784)

    values = values[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    cropped = Image.fromarray((values * 255).astype("uint8"), mode="L")
    scale = 20.0 / max(cropped.size)
    resized_size = tuple(max(1, int(round(size * scale))) for size in cropped.size)
    resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    digit = cropped.resize(resized_size, resampling)

    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(digit, ((28 - resized_size[0]) // 2, (28 - resized_size[1]) // 2))
    values = np.asarray(canvas, dtype=theano.config.floatX) / 255.0

    total = values.sum()
    if total > 0:
        y_coords, x_coords = np.indices(values.shape)
        shift_x = int(round(13.5 - float((x_coords * values).sum() / total)))
        shift_y = int(round(13.5 - float((y_coords * values).sum() / total)))
        shifted = np.zeros_like(values)
        src_x0 = max(0, -shift_x)
        src_x1 = min(28, 28 - shift_x)
        dst_x0 = max(0, shift_x)
        dst_x1 = min(28, 28 + shift_x)
        src_y0 = max(0, -shift_y)
        src_y1 = min(28, 28 - shift_y)
        dst_y0 = max(0, shift_y)
        dst_y1 = min(28, 28 + shift_y)
        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = values[src_y0:src_y1, src_x0:src_x1]
        values = shifted
    return values.reshape(1, 784)


def random_labeled_image(dataset_dir):
    dataset_dir = Path(dataset_dir)
    images = [
        path
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not images:
        raise SystemExit("No image files found under {}".format(dataset_dir))
    path = random.choice(images)
    try:
        expected = int(path.parent.name)
    except ValueError:
        expected = None
    return path, expected


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
    expected = None
    if args.random_from:
        image_path, expected = random_labeled_image(args.random_from)
    else:
        image_path = Path(args.image)

    image = image_to_mnist_vector(image_path, args.invert)
    batch = np.repeat(image, model["mini_batch_size"], axis=0).astype(
        theano.config.floatX
    )
    predict_batch = theano.function([net.x], net.layers[-1].y_out)
    prediction = int(predict_batch(batch)[0])
    if expected is None:
        print(prediction)
        return
    print("image: {}".format(image_path))
    print("expected: {}".format(expected))
    print("predicted: {}".format(prediction))
    print("correct: {}".format(prediction == expected))


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
    image_source = predict_parser.add_mutually_exclusive_group(required=True)
    image_source.add_argument("--image")
    image_source.add_argument(
        "--random-from",
        default=None,
        help="Pick a random image from a labeled directory tree such as draw_dataset/.",
    )
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
