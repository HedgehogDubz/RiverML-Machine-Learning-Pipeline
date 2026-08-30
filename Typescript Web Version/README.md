# Typescript Web Version

The browser version of the machine learning visualizer.

## Build and run

```
npm install
npm run dev
```

## Seeds

- Copy Seed: copies the best network to the clipboard as a seed string.
- Load Seed: paste a seed to replace the current population with that network. The network format (inputs and outputs) has to match.

Seeds are compatible with the C++ desktop version in both directions.

`neural_network_by_seed.ts` is standalone. Copy that one file anywhere and call `neuralNetworkBySeed(seed, inputs)` to run a trained network without any of the other files.
