import { DecisionTree, TreeNode } from "./decisiontree.js";

// XGBoost Ensemble of Decision Trees
export class XGBoostEnsemble {
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

    constructor(inputSize: number, outputSize: number, shrinkage: number = 0.1, maxDepth: number = 4, maxTrees: number = Infinity, isClassification: boolean = false) {
        this.inputSize = inputSize;
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
                const pred = this.predict(inputs[i]);
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
            const pred = this.predict(inputs[i]);
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
            const pred = this.predict(testInputs[i]);
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

    // Predict: Sum predictions from all trees
    predict(input: number[]): number[] {
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

    static MIN_CELL_HEIGHT = 200;

    // Returns the total content height (for scroll calculations)
    getContentHeight(displayHeaderHeight: number, panelHeight: number): number {
        if (this.trees.length === 0) return displayHeaderHeight;
        const cols = Math.min(4, this.trees.length);
        const rows = Math.ceil(this.trees.length / cols);
        const availableHeight = panelHeight - displayHeaderHeight;
        const cellHeight = Math.max(XGBoostEnsemble.MIN_CELL_HEIGHT, availableHeight / rows);
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
        const cellHeight = Math.max(XGBoostEnsemble.MIN_CELL_HEIGHT, availableHeight / rows);

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
