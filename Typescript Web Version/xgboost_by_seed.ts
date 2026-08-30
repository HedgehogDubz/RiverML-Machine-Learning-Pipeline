// Standalone XGBoost runner.
// Copy and paste this file anywhere, no other files needed.
// Get a seed by training an XGBoost model in the visualizer and clicking Copy Seed.
// Works with seeds from both the TypeScript and C++ versions.
//
// Usage:
//   const outputs = xgboostBySeed(seed, [0.5, -0.2]);

export function xgboostBySeed(seed: string, inputs: number[]): number[] {
    // Seed format: XGBSEED1|inputSize,outputSize|shrinkage|tree;tree;...
    // each tree is preorder csv tokens: d,feature,threshold,... or l,v0,v1,...
    const parts = seed.trim().split('|');
    if (parts.length !== 4 || parts[0] !== 'XGBSEED1') {
        throw new Error('Invalid seed: expected XGBSEED1|sizes|shrinkage|trees');
    }
    const sizes = parts[1].split(',').map(Number);
    const shrinkage = Number(parts[2]);
    if (sizes.length !== 2 || sizes.some(isNaN) || isNaN(shrinkage)) {
        throw new Error('Invalid seed: bad sizes or shrinkage');
    }
    const [inputSize, outputSize] = sizes;
    if (inputs.length !== inputSize) {
        throw new Error(`Expected ${inputSize} inputs, received ${inputs.length}`);
    }

    // walk one tree to its leaf, reading tokens as we go
    const predictTree = (tokens: string[]): number[] => {
        let pos = 0;
        // skips a whole subtree without evaluating it
        const skipNode = () => {
            const tag = tokens[pos++];
            if (tag === 'l') {
                pos += outputSize;
            } else if (tag === 'd') {
                pos += 2;
                skipNode();
                skipNode();
            } else {
                throw new Error('Invalid seed: unknown node tag');
            }
        };
        const walk = (): number[] => {
            const tag = tokens[pos++];
            if (tag === 'l') {
                const value: number[] = [];
                for (let i = 0; i < outputSize; i++) value.push(Number(tokens[pos++]));
                if (value.some(isNaN)) throw new Error('Invalid seed: bad leaf value');
                return value;
            }
            if (tag !== 'd') throw new Error('Invalid seed: unknown node tag');
            const feature = parseInt(tokens[pos++], 10);
            const threshold = Number(tokens[pos++]);
            if (inputs[feature] < threshold) {
                const value = walk();
                skipNode();
                return value;
            }
            skipNode();
            return walk();
        };
        const value = walk();
        if (pos !== tokens.length) throw new Error('Invalid seed: leftover tree tokens');
        return value;
    };

    // sum the trees, first tree full weight, the rest scaled by shrinkage
    const result = new Array(outputSize).fill(0);
    const trees = parts[3].split(';');
    for (let t = 0; t < trees.length; t++) {
        const pred = predictTree(trees[t].split(','));
        const weight = t === 0 ? 1.0 : shrinkage;
        for (let j = 0; j < outputSize; j++) result[j] += pred[j] * weight;
    }
    return result;
}
