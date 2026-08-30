// Standalone neural network runner.
// Copy and paste this file anywhere, no other files needed.
// Get a seed by training a model in the visualizer and clicking Copy Seed.
// Works with seeds from both the TypeScript and C++ versions.
//
// Usage:
//   const outputs = neuralNetworkBySeed(seed, [0.5, -0.2]);

export function neuralNetworkBySeed(seed: string, inputs: number[]): number[] {
    // Seed format: NNSEED1|activation|outputActivation|layerSizes(csv)|params(csv)
    // params: for each layer after the input, per neuron: bias then weights
    const parts = seed.trim().split('|');
    if (parts.length !== 5 || parts[0] !== 'NNSEED1') {
        throw new Error('Invalid seed: expected NNSEED1|act|outAct|sizes|params');
    }
    const activation = parts[1];
    const outputActivation = parts[2];
    const sizes = parts[3].split(',').map(Number);
    const params = parts[4].split(',').map(Number);

    if (inputs.length !== sizes[0]) {
        throw new Error(`Expected ${sizes[0]} inputs, received ${inputs.length}`);
    }

    const activate = (fn: string, x: number): number => {
        if (fn === 'relu') return x > 0 ? x : 0;
        if (fn === 'sigmoid') return 1 / (1 + Math.exp(-x));
        return Math.tanh(x);
    };

    // Forward pass, layer by layer
    let values = inputs.slice();
    let p = 0;
    for (let l = 1; l < sizes.length; l++) {
        const fn = l === sizes.length - 1 ? outputActivation : activation;
        const next: number[] = new Array(sizes[l]);
        for (let n = 0; n < sizes[l]; n++) {
            let sum = params[p++];
            for (let w = 0; w < values.length; w++) {
                sum += params[p++] * values[w];
            }
            next[n] = activate(fn, sum);
        }
        values = next;
    }

    if (p !== params.length) {
        throw new Error('Invalid seed: wrong number of parameters');
    }
    return values;
}
