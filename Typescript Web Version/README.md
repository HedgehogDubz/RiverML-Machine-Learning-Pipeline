# Typescript Web Version

The browser version of the machine learning visualizer.

## Build and run

```
npm install
npm run dev
```

## Seeds

- Copy Seed: copies the current model to the clipboard and also drops it in the seed box below the buttons. Neural network methods make an `NNSEED1` seed, XGBoost makes an `XGBSEED1` seed.
- Load Seed: paste a seed into the seed box, then press Load Seed. The training method switches to match the seed type, and the network format (inputs and outputs) has to match.

Seeds are compatible with the C++ desktop version in both directions.

`neural_network_by_seed.ts` and `xgboost_by_seed.ts` are standalone. Copy either file anywhere to run a trained model without any of the other files. Both live in a `RiverML` namespace and parse the seed once in the constructor, so repeated predictions stay fast:

```ts
const nn = new RiverML.NeuralNetwork(seed);
const outputs = nn.run([0.5, -0.2]);

const model = new RiverML.XGBoost(seed);
const outputs2 = model.run([0.5, -0.2]);
```

## Architecture controls

The Activation and Output Act. dropdowns change the activation functions, and the Hidden Layers box takes a comma list of layer sizes (for example `5,8,5`). Changing any of them rebuilds the networks from scratch.
