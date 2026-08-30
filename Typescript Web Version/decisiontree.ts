// Decision Tree Node
export class TreeNode {
    feature: number | null = null; // Which input feature to split on (0 or 1)
    threshold: number | null = null; // Split threshold
    left: TreeNode | null = null;
    right: TreeNode | null = null;
    value: number[] | null = null; // Leaf value (for each output)

    isLeaf(): boolean {
        return this.value !== null;
    }
}

// Decision Tree for Regression
export class DecisionTree {
    root: TreeNode = new TreeNode();
    maxDepth: number = 4;
    minSamplesLeaf: number = 2;
    inputSize: number;
    outputSize: number;
    isClassification: boolean = false;
    private _drawLeft: number = 0;
    private _drawRight: number = 0;

    constructor(inputSize: number, outputSize: number, maxDepth: number = 4) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.maxDepth = maxDepth;
    }

    // Train the decision tree on input-output pairs
    train(inputs: number[][], outputs: number[][]) {
        this.root = this.buildTree(inputs, outputs, 0);
    }

    // Recursively build the tree
    private buildTree(inputs: number[][], outputs: number[][], depth: number): TreeNode {
        const node = new TreeNode();

        // Base cases: create leaf
        if (depth >= this.maxDepth || inputs.length < this.minSamplesLeaf) {
            node.value = this.calculateMean(outputs);
            return node;
        }

        // Find best split
        const bestSplit = this.findBestSplit(inputs, outputs);

        if (bestSplit === null) {
            node.value = this.calculateMean(outputs);
            return node;
        }

        // Apply split
        node.feature = bestSplit.feature;
        node.threshold = bestSplit.threshold;

        const [leftInputs, leftOutputs, rightInputs, rightOutputs] = this.splitData(
            inputs, outputs, bestSplit.feature, bestSplit.threshold
        );

        // Recursively build children
        node.left = this.buildTree(leftInputs, leftOutputs, depth + 1);
        node.right = this.buildTree(rightInputs, rightOutputs, depth + 1);

        return node;
    }

    // Find the best feature and threshold to split on
    private findBestSplit(inputs: number[][], outputs: number[][]): {feature: number, threshold: number} | null {
        let bestVariance = Infinity;
        let bestSplit: {feature: number, threshold: number} | null = null;

        // Try each feature
        for (let feature = 0; feature < this.inputSize; feature++) {
            // Get unique values for this feature
            const values = inputs.map(inp => inp[feature]);
            const uniqueValues = [...new Set(values)].sort((a, b) => a - b);

            // Try splits between adjacent unique values
            for (let i = 0; i < uniqueValues.length - 1; i++) {
                const threshold = (uniqueValues[i] + uniqueValues[i + 1]) / 2;

                const [leftInputs, leftOutputs, rightInputs, rightOutputs] = this.splitData(
                    inputs, outputs, feature, threshold
                );

                if (leftOutputs.length === 0 || rightOutputs.length === 0) continue;

                // Calculate weighted variance
                const leftVar = this.calculateVariance(leftOutputs);
                const rightVar = this.calculateVariance(rightOutputs);
                const totalVar = (leftOutputs.length * leftVar + rightOutputs.length * rightVar) / outputs.length;

                if (totalVar < bestVariance) {
                    bestVariance = totalVar;
                    bestSplit = {feature, threshold};
                }
            }
        }

        return bestSplit;
    }

    // Split data based on feature and threshold
    private splitData(inputs: number[][], outputs: number[][], feature: number, threshold: number):
        [number[][], number[][], number[][], number[][]] {
        const leftInputs: number[][] = [];
        const leftOutputs: number[][] = [];
        const rightInputs: number[][] = [];
        const rightOutputs: number[][] = [];

        for (let i = 0; i < inputs.length; i++) {
            if (inputs[i][feature] < threshold) {
                leftInputs.push(inputs[i]);
                leftOutputs.push(outputs[i]);
            } else {
                rightInputs.push(inputs[i]);
                rightOutputs.push(outputs[i]);
            }
        }

        return [leftInputs, leftOutputs, rightInputs, rightOutputs];
    }

    // Calculate mean of outputs
    private calculateMean(outputs: number[][]): number[] {
        if (outputs.length === 0) return new Array(this.outputSize).fill(0);

        const sum = new Array(this.outputSize).fill(0);
        for (const output of outputs) {
            for (let i = 0; i < this.outputSize; i++) {
                sum[i] += output[i];
            }
        }

        return sum.map(s => s / outputs.length);
    }

    // Calculate variance of outputs
    private calculateVariance(outputs: number[][]): number {
        if (outputs.length === 0) return 0;

        const mean = this.calculateMean(outputs);
        let variance = 0;

        for (const output of outputs) {
            for (let i = 0; i < this.outputSize; i++) {
                variance += (output[i] - mean[i]) ** 2;
            }
        }

        return variance / outputs.length;
    }

    // Predict output for given input
    predict(input: number[]): number[] {
        let node = this.root;

        while (!node.isLeaf()) {
            if (input[node.feature!] < node.threshold!) {
                node = node.left!;
            } else {
                node = node.right!;
            }
        }

        return node.value!;
    }

    // Get the set of nodes on the prediction path for a given input
    getPath(input: number[]): Set<TreeNode> {
        const path = new Set<TreeNode>();
        let node = this.root;
        while (node) {
            path.add(node);
            if (node.isLeaf()) break;
            if (input[node.feature!] < node.threshold!) {
                node = node.left!;
            } else {
                node = node.right!;
            }
        }
        return path;
    }

    // Draw the tree, optionally highlighting a prediction path
    draw(ctx: CanvasRenderingContext2D, left: number, top: number, width: number, height: number, highlightPath?: Set<TreeNode>) {
        ctx.save();
        this._drawLeft = left;
        this._drawRight = left + width;
        ctx.fillStyle = "#f8f8f8";
        ctx.fillRect(left, top, width, height);

        if (this.root) {
            this.drawNode(ctx, this.root, left + width / 2, top + 15, width * 0.8, height - 20, 0, highlightPath);
        }

        ctx.restore();
    }

    // Recursively draw tree nodes
    private drawNode(ctx: CanvasRenderingContext2D, node: TreeNode, x: number, y: number,
                     width: number, height: number, depth: number, highlightPath?: Set<TreeNode>) {
        const onPath = highlightPath ? highlightPath.has(node) : false;
        const nodeRadius = onPath ? 10 : 8;
        const verticalGap = 25;

        // Clamp text x so it doesn't overflow left/right of the draw area
        const clampX = (textX: number, text: string, font: string) => {
            ctx.font = font;
            const tw = ctx.measureText(text).width;
            const minX = this._drawLeft + tw / 2 + 2;
            const maxX = this._drawRight - tw / 2 - 2;
            return Math.max(minX, Math.min(maxX, textX));
        };

        if (node.isLeaf()) {
            // Draw leaf node
            ctx.fillStyle = onPath ? "#E53935" : "#4CAF50";
            ctx.beginPath();
            ctx.arc(x, y, nodeRadius, 0, 2 * Math.PI);
            ctx.fill();

            // Draw value
            const leafFont = onPath ? "bold 10px sans-serif" : "9px sans-serif";
            ctx.fillStyle = onPath ? "#E53935" : "#000";
            ctx.font = leafFont;
            ctx.textAlign = "center";
            let valStr: string;
            if (this.isClassification && node.value!.length > 1) {
                let maxIdx = 0;
                for (let k = 1; k < node.value!.length; k++) {
                    if (node.value![k] > node.value![maxIdx]) maxIdx = k;
                }
                valStr = (maxIdx + 1).toString();
            } else {
                valStr = node.value!.map(v => v.toFixed(2)).join(', ');
            }
            ctx.fillText(valStr, clampX(x, valStr, leafFont), y + nodeRadius + 10);
        } else {
            // Draw decision node
            ctx.fillStyle = onPath ? "#E53935" : "#2196F3";
            ctx.beginPath();
            ctx.arc(x, y, nodeRadius, 0, 2 * Math.PI);
            ctx.fill();

            // Draw split label
            const splitFont = onPath ? "bold 9px sans-serif" : "8px sans-serif";
            ctx.fillStyle = onPath ? "#E53935" : "#000";
            ctx.font = splitFont;
            ctx.textAlign = "center";
            const featureName = node.feature === 0 ? "x" : "y";
            const splitText = `${featureName}<${node.threshold!.toFixed(1)}`;
            ctx.fillText(splitText, clampX(x, splitText, splitFont), y - nodeRadius - 3);

            // Draw children
            if (node.left && node.right) {
                const childWidth = width / 2;
                const leftX = x - width / 4;
                const rightX = x + width / 4;
                const childY = y + verticalGap;

                // Left line
                const leftOnPath = highlightPath ? highlightPath.has(node.left) : false;
                ctx.strokeStyle = leftOnPath ? "#E53935" : "#666";
                ctx.lineWidth = leftOnPath ? 2.5 : 1;
                ctx.beginPath();
                ctx.moveTo(x, y + nodeRadius);
                ctx.lineTo(leftX, childY - (leftOnPath ? 10 : 8));
                ctx.stroke();

                // Right line
                const rightOnPath = highlightPath ? highlightPath.has(node.right) : false;
                ctx.strokeStyle = rightOnPath ? "#E53935" : "#666";
                ctx.lineWidth = rightOnPath ? 2.5 : 1;
                ctx.beginPath();
                ctx.moveTo(x, y + nodeRadius);
                ctx.lineTo(rightX, childY - (rightOnPath ? 10 : 8));
                ctx.stroke();

                // Recursively draw children
                this.drawNode(ctx, node.left, leftX, childY, childWidth, height - verticalGap, depth + 1, highlightPath);
                this.drawNode(ctx, node.right, rightX, childY, childWidth, height - verticalGap, depth + 1, highlightPath);
            }
        }
    }
}
