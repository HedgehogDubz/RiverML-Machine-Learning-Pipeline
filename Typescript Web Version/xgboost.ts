import { RiverML as Trees } from "./decisiontree.js";

// XGBoost stage of the RiverML pipeline, an ensemble of decision trees trained on residuals.
// Seed format is shared with the C++ version.
//
// Copy this file plus decisiontree.ts anywhere to run a trained model:
//   const model = new RiverML.XGBoost(seed);
//   const outputs = model.run([0.5, -0.2]);
export namespace RiverML {

import DecisionTree = Trees.DecisionTree;
import TreeNode = Trees.TreeNode;

// XGBoost Ensemble of Decision Trees
export class XGBoost {
    trees: DecisionTree[] = [];
    generation: number = 0;
    inputSize: number;
    outputSize: number;
    shrinkage: number = 0.1;
    maxDepth: number = 4;
    maxTrees: number = Infinity;
    isClassification: boolean = false;
    trainRMSE: number = 0;
    trainMAE: number = 0;
    testRMSE: number = 0;
    testMAE: number = 0;

    // Build an empty ensemble, or build straight from a seed string
    constructor(seed: string);
    constructor(inputSize: number, outputSize: number, shrinkage?: number, maxDepth?: number, maxTrees?: number, isClassification?: boolean);
    constructor(inputSizeOrSeed: number | string, outputSize: number = 1, shrinkage: number = 0.1, maxDepth: number = 4, maxTrees: number = Infinity, isClassification: boolean = false) {
        if (typeof inputSizeOrSeed === 'string') {
            const loaded = XGBoost.fromSeed(inputSizeOrSeed);
            this.inputSize = loaded.inputSize;
            this.outputSize = loaded.outputSize;
            this.shrinkage = loaded.shrinkage;
            this.maxDepth = loaded.maxDepth;
            this.maxTrees = loaded.maxTrees;
            this.isClassification = loaded.isClassification;
            this.trees = loaded.trees;
            this.generation = loaded.generation;
            return;
        }
        this.inputSize = inputSizeOrSeed;
        this.outputSize = outputSize;
        this.shrinkage = shrinkage;
        this.maxDepth = maxDepth;
        this.maxTrees = maxTrees;
        this.isClassification = isClassification;
    }

    // Train: Add one new tree that learns the residuals
    train(inputs: number[][], outputs: number[][], testInputs?: number[][], testFn?: (inputs: number[]) => number[]) {
        this.generation++;

        // First call: train a base tree on the actual data
        if (this.trees.length === 0) {
            const baseTree = new DecisionTree(this.inputSize, this.outputSize, this.maxDepth);
            baseTree.isClassification = this.isClassification;
            baseTree.train(inputs, outputs);
            this.trees.push(baseTree);
            this.computeErrors(inputs, outputs, testInputs, testFn);
            return;
        }

        // Only add new trees if under the limit
        if (this.trees.length < this.maxTrees) {
            // Calculate residuals from current ensemble
            const residuals: number[][] = [];
            for (let i = 0; i < inputs.length; i++) {
                const pred = this.run(inputs[i]);
                const residual = outputs[i].map((val, idx) => val - pred[idx]);
                residuals.push(residual);
            }

            // Create and train new tree on residuals
            const newTree = new DecisionTree(this.inputSize, this.outputSize, this.maxDepth);
            newTree.isClassification = this.isClassification;
            newTree.train(inputs, residuals);
            this.trees.push(newTree);
        }

        this.computeErrors(inputs, outputs, testInputs, testFn);
    }

    private computeErrors(inputs: number[][], outputs: number[][], testInputs?: number[][], testFn?: (inputs: number[]) => number[]) {
        // Training errors
        let sqSum = 0, absSum = 0;
        for (let i = 0; i < inputs.length; i++) {
            const pred = this.run(inputs[i]);
            for (let j = 0; j < this.outputSize; j++) {
                const diff = Math.abs(pred[j] - outputs[i][j]);
                sqSum += diff ** 2;
                absSum += diff;
            }
        }
        this.trainRMSE = Math.sqrt(sqSum / inputs.length);
        this.trainMAE = absSum / inputs.length;

        // Test errors
        if (!testInputs || !testFn) { this.testRMSE = 0; this.testMAE = 0; return; }
        sqSum = 0; absSum = 0;
        for (let i = 0; i < testInputs.length; i++) {
            const pred = this.run(testInputs[i]);
            const expected = testFn(testInputs[i]);
            for (let j = 0; j < this.outputSize; j++) {
                const diff = Math.abs(pred[j] - expected[j]);
                sqSum += diff ** 2;
                absSum += diff;
            }
        }
        this.testRMSE = Math.sqrt(sqSum / testInputs.length);
        this.testMAE = absSum / testInputs.length;
    }

    // Sums every tree, base tree at full weight and the rest scaled by shrinkage
    run(input: number[]): number[] {
        if (this.trees.length === 0) {
            return new Array(this.outputSize).fill(0);
        }

        const result = new Array(this.outputSize).fill(0);

        for (let i = 0; i < this.trees.length; i++) {
            const treePred = this.trees[i].predict(input);
            const weight = i === 0 ? 1.0 : this.shrinkage; // First tree full weight, others scaled

            for (let j = 0; j < this.outputSize; j++) {
                result[j] += treePred[j] * weight;
            }
        }

        return result;
    }

    // Seed format shared with the C++ version:
    // XGBSEED1|inputSize,outputSize|shrinkage|tree;tree;...
    // each tree is preorder csv tokens: d,feature,threshold,... or l,v0,v1,...
    toSeed(): string {
        const serNode = (node: TreeNode): string => {
            if (node.isLeaf()) return 'l,' + node.value!.join(',');
            return `d,${node.feature},${node.threshold},` + serNode(node.left!) + ',' + serNode(node.right!);
        };
        const trees = this.trees.map(t => serNode(t.root)).join(';');
        return `XGBSEED1|${this.inputSize},${this.outputSize}|${this.shrinkage}|${trees}`;
    }

    static fromSeed(seed: string): XGBoost {
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
        const ens = new XGBoost(inputSize, outputSize, shrinkage, 4, Infinity, outputSize > 1);

        for (const treeStr of parts[3].split(';')) {
            const tokens = treeStr.split(',');
            let pos = 0;
            const parseNode = (): TreeNode => {
                const node = new TreeNode();
                const tag = tokens[pos++];
                if (tag === 'l') {
                    node.value = [];
                    for (let i = 0; i < outputSize; i++) node.value.push(Number(tokens[pos++]));
                    if (node.value.some(isNaN)) throw new Error('Invalid seed: bad leaf value');
                } else if (tag === 'd') {
                    node.feature = parseInt(tokens[pos++], 10);
                    node.threshold = Number(tokens[pos++]);
                    if (isNaN(node.feature) || isNaN(node.threshold)) throw new Error('Invalid seed: bad split');
                    node.left = parseNode();
                    node.right = parseNode();
                } else {
                    throw new Error('Invalid seed: unknown node tag');
                }
                return node;
            };
            const tree = new DecisionTree(inputSize, outputSize);
            tree.isClassification = outputSize > 1;
            tree.root = parseNode();
            if (pos !== tokens.length) throw new Error('Invalid seed: leftover tree tokens');
            ens.trees.push(tree);
        }
        ens.generation = ens.trees.length;
        return ens;
    }

    static MIN_CELL_HEIGHT = 200;

    // Returns the total content height (for scroll calculations)
    getContentHeight(displayHeaderHeight: number, panelHeight: number): number {
        if (this.trees.length === 0) return displayHeaderHeight;
        const cols = Math.min(4, this.trees.length);
        const rows = Math.ceil(this.trees.length / cols);
        const availableHeight = panelHeight - displayHeaderHeight;
        const cellHeight = Math.max(XGBoost.MIN_CELL_HEIGHT, availableHeight / rows);
        return displayHeaderHeight + rows * cellHeight;
    }

    public drawHeader(ctx: CanvasRenderingContext2D, left: number, top: number, width: number, displayHeaderHeight: number) {
        ctx.fillStyle = "#e0e0e0";
        ctx.fillRect(left, top, width, displayHeaderHeight);
        ctx.fillStyle = "#000000";
        ctx.font = '12px sans-serif';
        ctx.textBaseline = 'middle';
        const text = `Gen: ${this.generation} | Trees: ${this.trees.length} | Train ϵ RMSE: ${this.trainRMSE.toFixed(4)} μ MAE: ${this.trainMAE.toFixed(4)} | Test ϵ RMSE: ${this.testRMSE.toFixed(4)} μ MAE: ${this.testMAE.toFixed(4)}`;
        ctx.fillText(text, left + 5, top + displayHeaderHeight / 2);
    }

    // Draw a single tree (base or latest) filling the full content area (no header — caller draws it)
    drawSingleTree(ctx: CanvasRenderingContext2D, left: number, top: number, width: number, height: number,
                   treeIndex: number, input?: number[]) {
        ctx.save();

        if (this.trees.length === 0 || treeIndex < 0 || treeIndex >= this.trees.length) {
            ctx.restore();
            return;
        }

        const tree = this.trees[treeIndex];

        // Draw label
        ctx.fillStyle = "#000000";
        ctx.font = '11px sans-serif';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        const label = treeIndex === 0 ? `Tree 0 (base)` : `Tree ${treeIndex}`;
        ctx.fillText(label, left + 5, top + 2);

        const highlightPath = input ? tree.getPath(input) : undefined;
        tree.draw(ctx, left, top + 15, width, height - 15, highlightPath);

        // Show prediction value if input provided
        if (input && highlightPath) {
            const pred = tree.predict(input);
            ctx.fillStyle = "#E53935";
            ctx.font = 'bold 12px sans-serif';
            ctx.textAlign = 'right';
            ctx.textBaseline = 'top';
            ctx.fillText(`Prediction: ${pred.map(v => v.toFixed(3)).join(', ')}`, left + width - 5, top + 2);
        }

        ctx.restore();
    }

    // Draw all trees in a scrollable grid, with input path highlighting
    draw(ctx: CanvasRenderingContext2D, left: number, top: number, width: number, height: number,
         displayHeaderHeight: number, scrollY: number = 0, input?: number[]) {
        ctx.save();

        // Clip to the content area below the header
        ctx.beginPath();
        ctx.rect(left, top + displayHeaderHeight, width, height - displayHeaderHeight);
        ctx.clip();

        if (this.trees.length === 0) {
            ctx.restore();
            return;
        }

        // Calculate grid layout with minimum cell size
        const cols = Math.min(4, this.trees.length);
        const rows = Math.ceil(this.trees.length / cols);
        const cellWidth = width / cols;
        const availableHeight = height - displayHeaderHeight;
        const cellHeight = Math.max(XGBoost.MIN_CELL_HEIGHT, availableHeight / rows);

        // Draw each tree in the ensemble (scrolled)
        for (let i = 0; i < this.trees.length; i++) {
            const col = i % cols;
            const row = Math.floor(i / cols);
            const x = left + col * cellWidth;
            const y = top + displayHeaderHeight + row * cellHeight - scrollY;

            // Skip if entirely outside visible area
            if (y + cellHeight < top + displayHeaderHeight || y > top + height) continue;

            // Draw tree background
            ctx.fillStyle = i === 0 ? "#f0f0ff" : "#f8f8f8";
            ctx.fillRect(x + 2, y + 2, cellWidth - 4, cellHeight - 4);

            // Draw tree index
            ctx.fillStyle = "#000000";
            ctx.font = '10px sans-serif';
            ctx.textAlign = 'left';
            ctx.fillText(`Tree ${i}${i === 0 ? ' (base)' : ''}`, x + 5, y + 12);

            // Draw the tree with path highlighting
            const highlightPath = input ? this.trees[i].getPath(input) : undefined;
            this.trees[i].draw(ctx, x + 5, y + 20, cellWidth - 10, cellHeight - 25, highlightPath);
        }

        // Draw scroll indicator if content overflows
        const totalContentHeight = rows * cellHeight;
        if (totalContentHeight > availableHeight) {
            const scrollBarHeight = Math.max(20, availableHeight * (availableHeight / totalContentHeight));
            const scrollBarY = top + displayHeaderHeight + (scrollY / (totalContentHeight - availableHeight)) * (availableHeight - scrollBarHeight);
            ctx.fillStyle = "rgba(0, 0, 0, 0.3)";
            ctx.fillRect(left + width - 6, scrollBarY, 4, scrollBarHeight);
        }

        ctx.restore();
    }
}

} // namespace RiverML
