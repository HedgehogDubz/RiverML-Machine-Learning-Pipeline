# RiverML-Machine-Learning-Pipeline

RiverML is a lightweight machine learning pipeline you can quickly train, export, and run live inside your code. Watch in real-time as your model trains, export the trained model as a single seed string, and load that seed anywhere else in the pipeline to make predictions in any application!

The pipeline has three stages:

1. **Train** in either the C++ desktop app or the TypeScript web app run on Github Pages.
https://hedgehogdubz.github.io/Machine-Learning-Visualizer/
2. **Export** the trained model as a seed string with Copy Seed.
3. **Run** that seed in any RiverML target, in either language (C++, Typescript)

Both languages implement the same `RiverML` namespace, the same models, and the same seed format,
so a model trained in one stage runs unchanged in the other.

---

## CPP Desktop Version

The native build, and the fast end of the pipeline. Drawn with raylib.

```
brew install raylib
make && ./mlvisualizer
```

Built with C++23. What it gives you:

- Trains about 36x faster than the browser build, roughly 570 genetic generations per second against 16
- Model view on top: the network with bias rings, value fills, and green/red weight lines, or the
  XGBoost decision trees with the live prediction path highlighted in red
- Data view below: a fine gradient of the model output over the input space, plus an 11x11 grid of
  actual values, or a line graph for the 1 input format
- Training methods: genetic, backpropagation, and XGBoost
- Editable architecture: activation functions and a comma list of hidden layer sizes such as `5,8,5`
- Copy Seed and Load Seed with a seed box, plus a `seed.txt` written next to the binary

Those timings use the default network, `2 -> 7 -> 10 -> 20 -> 20 -> 10 -> 7 -> 1`, which is 77 nodes and
961 connections. Each generation scores all 16 networks against the training grid: 21 x 21 input points
spanning -1 to 1 on both axes, so 441 points per network, 6.8 million multiply adds per generation.

Everything lives in three headers: `neuralnetwork.h`, `xgboost.h`, and `decisiontree.h`.

## Typescript Web Version

The shareable end of the pipeline. Runs entirely in the browser.

```
npm install
npm run dev
```

Same models and controls as the desktop build, plus the N class categorical format that the desktop
build does not implement yet. This is the version deployed to GitHub Pages.

---

## Using the Seeds

A seed is the entire trained model as one line of text. Copy Seed puts it on the clipboard and in the
seed box; Load Seed reads whatever is pasted in that box and switches the training method to match the
seed type automatically.

Structure:
- `NNSEED1|activation|outputActivation|layerSizes|params` for neural networks
- `XGBSEED1|inputSize,outputSize|shrinkage|trees` for XGBoost

Loading a seed is the last stage of the pipeline, and it needs no visualizer at all. Copy
`neuralnetwork.h` (or `xgboost.h` plus `decisiontree.h`) into any project:

```cpp
RiverML::NeuralNetwork nn(seed);
std::vector<double> outputs = nn.run({0.5, -0.2});

RiverML::XGBoost model(seed);
std::vector<double> outputs2 = model.run({0.5, -0.2});
```

The TypeScript side is the same, from `neuralnetwork.ts` and `xgboost.ts`:

```ts
const nn = new RiverML.NeuralNetwork(seed);
const outputs = nn.run([0.5, -0.2]);

const model = new RiverML.XGBoost(seed);
const outputs2 = model.run([0.5, -0.2]);
```

Seeds move in both directions between the two languages. Neural network outputs agree to about 1e-15,
the limit of double precision, and XGBoost outputs match exactly.

---

## Latest update: Caching Seeds in Memory

Running a seed used to reparse the whole seed string on every single prediction. Now the seed is parsed
once when the model is constructed, and `run()` is pure arithmetic. Measured over 10,000 predictions:

| Model | Before, parse per call | After, built once | Speedup |
| --- | --- | --- | --- |
| C++ neural network | 242.5 ms | 2.3 ms | **105x** |
| C++ XGBoost | 165.7 ms | 1.3 ms | **126x** |
| TypeScript neural network | 795.4 ms | 8.2 ms | **97x** |
| TypeScript XGBoost | 310.8 ms | 2.3 ms | **133x** |

Parsing itself is now a one time cost of about 0.02 ms for a 20 tree model. Seed parsing also uses
string views in C++, so the parameter block, which runs to hundreds of thousands of characters on a
large ensemble, is never copied.


## Why I Built This

Machine learning can feel abstract when you only see formulas or final results. I wanted something where
you can see the model update live, follow what the network is actually doing, understand the logic step
by step instead of treating it like a black box, and change parameters to see how it affects the model.

This was also an opportunity for me to code from scratch. First and foremost, I created this project to
level up my coding ability. It was also made to show my fundamental understanding of some of the most
popular machine learning models.




## Directions I might explore later

- More algorithms and training options
- Richer debugging views and visualization controls
- More customizability in the models and algorithms
- The N class categorical format in the C++ build
