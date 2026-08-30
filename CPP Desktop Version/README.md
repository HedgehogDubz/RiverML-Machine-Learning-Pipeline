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

- Top half: the network. Circle border color is the bias, fill is the neuron value, lines are weights (green positive, red negative).
- Bottom half: the model output over the input space. Fine gradient grid on the left, 11x11 grid with values on the right. The 1 input format shows a line graph instead.
- Sidebar: network format, test function, data format, network view (best or all 16), training method (genetic or backprop), and sliders.

## Seeds

- Copy Seed: puts the best network on the clipboard as a seed string and also writes `seed.txt`.
- Load Seed: reads a seed from the clipboard and replaces the population with it.

Seeds are compatible with the TypeScript web version in both directions.

`neural_network_by_seed.cpp` is standalone. Copy that one file anywhere and call `neuralNetworkBySeed(seed, inputs)` to run a trained network. It can also be built as a small CLI:

```
make run_seed
./run_seed seed.txt 0.5 -0.2
```

## Not ported

XGBoost / decision tree training and the N category format only exist in the web version for now.
