# CPP Desktop Version

Native desktop version of the machine learning visualizer, drawn with raylib.

## Build and run

Needs raylib installed (on macOS: `brew install raylib`).

```
make
./mlvisualizer
```

## What it does

Same idea as the web version. Starts with the 2 input 1 output network.

- Top half: the network. Circle border color is the bias, fill is the neuron value, lines are weights (green positive, red negative). In XGBoost mode this shows the decision trees instead (base, latest, or all with scrolling), with the prediction path highlighted in red.
- Bottom half: the model output over the input space. Fine gradient grid on the left, 11x11 grid with values on the right. The 1 input format shows a line graph instead.
- Sidebar: network format, test function, data format, view, training method (genetic, backprop, or XGBoost), and sliders.

The network architecture is editable: pick the hidden and output activation functions, and type the hidden layer sizes as a comma list (for example `5,8,5`). XGBoost has shrinkage, max depth, tree limit, and training grid resolution settings.

## Seeds

- Copy Seed: puts the current model on the clipboard as a seed string and also writes `seed.txt`. Neural network methods make an `NNSEED1` seed, XGBoost makes an `XGBSEED1` seed.
- Load Seed: reads a seed from the clipboard, switching the training method to match the seed type.

Seeds are compatible with the TypeScript web version in both directions.

`neural_network_by_seed.cpp` and `xgboost_by_seed.cpp` are standalone. Copy either file anywhere and call `neuralNetworkBySeed(seed, inputs)` or `xgboostBySeed(seed, inputs)` to run a trained model. They can also be built as small CLIs:

```
make run_seed
./run_seed seed.txt 0.5 -0.2

make run_xgb_seed
./run_xgb_seed seed.txt 0.5 -0.2
```

## Not ported

The N category format only exists in the web version for now.
