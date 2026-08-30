# Typescript Web Version

The browser version of the machine learning visualizer.

## Build and run

```
npm install
npm run dev
```

## Seeds

- Copy Seed: copies the current model to the clipboard as a seed string. Neural network methods make an `NNSEED1` seed, XGBoost makes an `XGBSEED1` seed.
- Load Seed: paste a seed to load that model, switching the training method to match the seed type. The network format (inputs and outputs) has to match.

Seeds are compatible with the C++ desktop version in both directions.

`neural_network_by_seed.ts` and `xgboost_by_seed.ts` are standalone. Copy either file anywhere and call `neuralNetworkBySeed(seed, inputs)` or `xgboostBySeed(seed, inputs)` to run a trained model without any of the other files.

## Architecture controls

The Activation and Output Act. dropdowns change the activation functions, and the Hidden Layers box takes a comma list of layer sizes (for example `5,8,5`). Changing any of them rebuilds the networks from scratch.
