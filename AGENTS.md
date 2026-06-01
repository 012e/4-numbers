# AGENTS.md

## Runtime
- Use `uv run ...` from the repository root; `pyproject.toml` and `uv.lock` pin this project to Python 3.10 because Theano/Numpy here are legacy (`numpy<1.22`, `theano==1.0.5` via lockfile).
- Do not upgrade this repo to Python 3.11+ or modern NumPy unless the task is explicitly a port; Theano 1.0.5 compatibility is the constraint.
- The MNIST dataset is expected at `data/mnist.pkl.gz`. Legacy modules such as `src/mnist_loader.py`, `src/network3.py`, `src/conv.py`, and `src/expand_mnist.py` use `../data/...` defaults, so run them from `src` unless using `src/network3_cli.py`.

## Main Entry Points
- `src/network3.py` is a library, not a script; running it directly does not train a model. Use `src/network3_cli.py` or import `Network`, layers, and `load_data_shared()`.
- Train and save a quick smoke model from repo root: `uv run src/network3_cli.py train --architecture shallow --epochs 1 --eta 0.1 --lmbda 0 --fc-neurons 30 --output models/network3-smoke.pkl.gz`.
- Predict a digit image: `uv run src/network3_cli.py predict --model models/network3-cpu-good.pkl.gz --image path/to/image.png`.

## CLI Flags And Presets
- `--architecture` choices: `shallow` is fastest/weakest, `basic-conv` is the best under-5-minute CPU preset, `dbl-conv-relu` is stronger but slower, and `post` is the full dropout-heavy architecture and usually too slow on CPU.
- `--epochs` controls runtime linearly; use `1` for smoke tests, `3` with `basic-conv` for a short CPU run, `20+` with `dbl-conv-relu` only when the user expects a longer run.
- `--eta` should match the architecture defaults used in `src/conv.py`: `0.1` for `shallow` and `basic-conv`, `0.03` for `dbl-conv-relu` and `post`.
- `--lmbda` is L2 regularization; use `0` for smoke/short `shallow` or `basic-conv` runs, `0.1` for the stronger `dbl-conv-relu` CPU preset.
- `--fc-neurons` sets the hidden fully connected width for `shallow`, `basic-conv`, and `dbl-conv-relu`; use `30` for fastest smoke tests and `100` for normal CPU training. `post` ignores this and uses two 1000-neuron FC layers.
- `--output` should point under `models/` because that directory is gitignored.
- Under-5-minute CPU training command: `uv run src/network3_cli.py train --architecture basic-conv --epochs 3 --eta 0.1 --lmbda 0 --fc-neurons 100 --output models/network3-basic-conv-3ep.pkl.gz`.
- Stronger CPU command when time is acceptable: `uv run src/network3_cli.py train --architecture dbl-conv-relu --epochs 20 --eta 0.03 --lmbda 0.1 --fc-neurons 100 --output models/network3-cpu-good.pkl.gz`.
- Prediction uses `--invert auto` by default; use `--invert yes` for dark-on-light images if auto-inversion guesses wrong, or `--invert no` for MNIST-like light digit on dark background.
- Validate one random labeled drawing from `draw_dataset/<digit>/...`: `uv run src/network3_cli.py predict --model models/network3-cpu-good.pkl.gz --random-from draw_dataset`.

## Training Notes
- `src/network3_cli.py` architectures are `shallow`, `basic-conv`, `dbl-conv-relu`, and `post`; avoid `post` on CPU unless the user expects a very long run.
- Model artifacts belong under `models/`, which is gitignored. The CLI saves gzip-pickled parameter values plus architecture metadata.
- `src/network3.py` prints a GPU attempt message because `GPU = True`, but CPU fallback is normal in this environment.
- Expanded-data experiments in `src/conv.py` require generating `data/mnist_expanded.pkl.gz` first; run `uv run expand_mnist.py` from `src` for that legacy path to resolve.

## Verification
- There is no test suite or CI config in this repo. Use focused smoke commands instead of inventing tests.
- Fast import/runtime check from repo root: `uv run src/network3_cli.py train --architecture shallow --epochs 1 --eta 0.1 --lmbda 0 --fc-neurons 30 --output models/network3-smoke.pkl.gz`.
- Validate saved-model inference with an existing image: `uv run src/network3_cli.py predict --model models/network3-smoke.pkl.gz --image fig/mnist_first_digit.png`.
