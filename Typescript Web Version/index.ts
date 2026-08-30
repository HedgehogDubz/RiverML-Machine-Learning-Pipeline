import { ActivationFunction, NeuralNetwork, NeuralNetworkList } from "./neuralnetwork.js";
import { XGBoostEnsemble } from "./xgboost.js";

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const ctx = canvas?.getContext('2d');

if (!canvas || !ctx) {
    console.error("Canvas or context undefined");
    throw new Error("Canvas initialization failed");
}




let frame = 0;
window.addEventListener('resize', resizeCanvas);
window.onload = function() {
    // Reset all form elements to defaults so browser-cached values don't desync
    document.querySelectorAll('select').forEach(sel => {
        const opts = sel.options;
        let found = false;
        for (let i = 0; i < opts.length; i++) {
            if (opts[i].defaultSelected) { sel.selectedIndex = i; found = true; break; }
        }
        if (!found) sel.selectedIndex = 0;
    });
    document.querySelectorAll('input[type="range"]').forEach(el => {
        (el as HTMLInputElement).value = (el as HTMLInputElement).defaultValue;
    });
    document.querySelectorAll('input[type="number"]').forEach(el => {
        (el as HTMLInputElement).value = (el as HTMLInputElement).defaultValue;
    });
    document.querySelectorAll('input[type="checkbox"]').forEach(el => {
        (el as HTMLInputElement).checked = (el as HTMLInputElement).defaultChecked;
    });
    start();
    resizeCanvas();
    setInterval(()=>{
        update();
        frame++;
    }, 1);
};

let isStarted = false;
let networkFormat: networkType = 'Val1in1Out';
type networkType = 'Val1in1Out' | 'Val2in1out' | 'Cat2in2out' | 'CatNout';
let numCategories = 3;
let showDataFormat: ShowDataType = 'output';
type ShowDataType = 'none' | 'output' | 'error' | 'test' | 'testvsoutput';
let showNetworkFormat: ShowNetworkType = 'best';
type ShowNetworkType = 'best' | 'all' | 'base' | 'latest';
let testFunctionVal1in1out: TestFunctionTypeVal1 = 'sine';
type TestFunctionTypeVal1 = 'sine' | 'square' | 'sawtooth' | 'abs' | 'cubic' | 'step' | 'gaussian' | 'triangle' | 'tanh' | 'sinc' | 'noise' | 'polynomial';
let testFunctionVal2in1out: TestFunctionTypeVal2 = 'wave';
type TestFunctionTypeVal2 = 'wave' | 'radial' | 'xy' | 'checkerboard' | 'spiral' | 'diagonal' | 'gaussian2d' | 'saddle' | 'ripple' | 'peaks' | 'step2d' | 'swiss';
let testFunctionCat2in2out: TestFunctionTypeCat2in2Out = 'circle';
type TestFunctionTypeCat2in2Out = 'circle' | 'square' | 'quadrants' | 'donut' | 'xor' | 'diagonal' | 'stripes' | 'checkerboard' | 'spiral' | 'moons' | 'diamond' | 'cross';
let testFunctionCatNout: TestFunctionTypeCatN = 'sectors';
type TestFunctionTypeCatN = 'sectors' | 'rings' | 'grid' | 'spiral' | 'stripes' | 'checkerboard' | 'voronoi' | 'waves';
let showTrainingData: ShowTrainingDataType = 'none';
type ShowTrainingDataType = 'none' | 'output' | 'error' | 'test';
let trainingMethod: TrainingMethod = 'genetic';
type TrainingMethod = 'genetic' | 'backprop' | 'XGBoost';
let input1 = 0;
let input2 = 0;
let generationsPerDrawCycle = 1;
let learningRate = 0.01; // Learning rate for backpropagation
let momentum = 0.9; // Momentum for backpropagation
let geneticMutationWeights = 100;
let geneticMutationWeightStrength = 0.01;
let geneticMutationBiases = 100;
let geneticMutationBiasStrength = 0.01;
let xgbMaxTrees = Infinity;
let xgbLimitTrees = false;
let xgbShrinkage = 0.1;
let xgbMaxDepth = 4;
let xgbResolution = 0.02;

// Colors for Cat2in2out visualization
let color1 = {r: 100, g: 150, b: 255}; // Light blue for class 1
let color2 = {r: 255, g: 100, b: 100}; // Light red for class 2

// Generate N distinct colors using HSL
function getCategoryColors(n: number): {r: number, g: number, b: number}[] {
    const colors: {r: number, g: number, b: number}[] = [];
    for (let i = 0; i < n; i++) {
        const hue = (i / n) * 360;
        const s = 0.7, l = 0.55;
        // HSL to RGB
        const c = (1 - Math.abs(2 * l - 1)) * s;
        const x = c * (1 - Math.abs(((hue / 60) % 2) - 1));
        const m = l - c / 2;
        let r1 = 0, g1 = 0, b1 = 0;
        if (hue < 60) { r1 = c; g1 = x; }
        else if (hue < 120) { r1 = x; g1 = c; }
        else if (hue < 180) { g1 = c; b1 = x; }
        else if (hue < 240) { g1 = x; b1 = c; }
        else if (hue < 300) { r1 = x; b1 = c; }
        else { r1 = c; b1 = x; }
        colors.push({
            r: Math.round((r1 + m) * 255),
            g: Math.round((g1 + m) * 255),
            b: Math.round((b1 + m) * 255)
        });
    }
    return colors;
}

function drawNClassGrid(ctx: CanvasRenderingContext2D, left: number, top: number, width: number, height: number,
    axis1low: number, axis2low: number, axis1high: number, axis2high: number,
    rows: number, columns: number, showText: boolean,
    getOutputs: (in1: number, in2: number) => number[],
    colors: {r: number, g: number, b: number}[]) {
    const spaceX = width / columns;
    const spaceY = height / rows;
    ctx.save();
    ctx.fillStyle = '#e0e0e0';
    ctx.fillRect(left, top, width, height);
    for (let i = 0; i < columns; i++) {
        for (let j = 0; j < rows; j++) {
            const x = left + i * spaceX;
            const y = top + j * spaceY;
            const in1 = axis1low + i * (axis1high - axis1low) / (columns - 1);
            const in2 = axis2low + j * (axis2high - axis2low) / (rows - 1);
            const outputs = getOutputs(in1, in2);
            // Argmax to find winning class
            let maxIdx = 0;
            for (let k = 1; k < outputs.length; k++) {
                if (outputs[k] > outputs[maxIdx]) maxIdx = k;
            }
            const col = colors[maxIdx % colors.length];
            // Blend with confidence
            const confidence = Math.max(0, Math.min(1, outputs[maxIdx]));
            const r = Math.round(col.r * confidence + 240 * (1 - confidence));
            const g = Math.round(col.g * confidence + 240 * (1 - confidence));
            const b = Math.round(col.b * confidence + 240 * (1 - confidence));
            ctx.fillStyle = `rgb(${r},${g},${b})`;
            ctx.fillRect(x, y, spaceX + (showText ? -1 : 1), spaceY + (showText ? -1 : 1));
            if (showText) {
                ctx.fillStyle = '#000';
                ctx.font = `${Math.min(spaceX, spaceY) * 0.3}px sans-serif`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(maxIdx.toString(), x + spaceX / 2, y + spaceY / 2);
            }
        }
    }
    ctx.restore();
}

function drawLineGraph(ctx: CanvasRenderingContext2D, left: number, top: number, width: number, height: number,
    xLow: number, xHigh: number, yLow: number, yHigh: number,
    lines: {fn: (x: number) => number, color: string, lineWidth?: number}[],
    numPoints: number = 200, showAxes: boolean = true) {
    ctx.save();
    ctx.fillStyle = '#f8f8f8';
    ctx.fillRect(left, top, width, height);

    const toScreenX = (x: number) => left + ((x - xLow) / (xHigh - xLow)) * width;
    const toScreenY = (y: number) => top + height - ((y - yLow) / (yHigh - yLow)) * height;

    // Draw grid lines
    if (showAxes) {
        ctx.strokeStyle = '#ddd';
        ctx.lineWidth = 0.5;
        for (let v = Math.ceil(xLow * 5) / 5; v <= xHigh; v += 0.2) {
            const sx = toScreenX(v);
            ctx.beginPath(); ctx.moveTo(sx, top); ctx.lineTo(sx, top + height); ctx.stroke();
        }
        for (let v = Math.ceil(yLow * 5) / 5; v <= yHigh; v += 0.2) {
            const sy = toScreenY(v);
            ctx.beginPath(); ctx.moveTo(left, sy); ctx.lineTo(left + width, sy); ctx.stroke();
        }
        // Axes
        ctx.strokeStyle = '#999';
        ctx.lineWidth = 1;
        if (yLow <= 0 && yHigh >= 0) {
            const y0 = toScreenY(0);
            ctx.beginPath(); ctx.moveTo(left, y0); ctx.lineTo(left + width, y0); ctx.stroke();
        }
        if (xLow <= 0 && xHigh >= 0) {
            const x0 = toScreenX(0);
            ctx.beginPath(); ctx.moveTo(x0, top); ctx.lineTo(x0, top + height); ctx.stroke();
        }
        // Axis labels
        ctx.fillStyle = '#666';
        ctx.font = '9px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        for (let v = Math.ceil(xLow * 2) / 2; v <= xHigh; v += 0.5) {
            ctx.fillText(v.toFixed(1), toScreenX(v), top + height + 1);
        }
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        for (let v = Math.ceil(yLow * 2) / 2; v <= yHigh; v += 0.5) {
            ctx.fillText(v.toFixed(1), left - 2, toScreenY(v));
        }
    }

    // Draw each line
    for (const line of lines) {
        ctx.strokeStyle = line.color;
        ctx.lineWidth = line.lineWidth ?? 2;
        ctx.beginPath();
        for (let i = 0; i <= numPoints; i++) {
            const x = xLow + (i / numPoints) * (xHigh - xLow);
            const y = line.fn(x);
            const sx = toScreenX(x);
            const sy = toScreenY(Math.max(yLow, Math.min(yHigh, y)));
            if (i === 0) ctx.moveTo(sx, sy);
            else ctx.lineTo(sx, sy);
        }
        ctx.stroke();
    }

    ctx.restore();
}

///////////////////////MAIN AREA//////////////////////////////////
// Number of networks depends on training method: genetic needs 16, backprop needs 1
// Default is genetic (16 networks)
let numOfNeuralNetworks = 16;
let inputSize = 2;
let outputSize = 1;
let hiddenLayerSizes = [7, 10, 20, 20, 10, 7];
let activationFunction: ActivationFunction = 'relu';
let outputActivationFunction: ActivationFunction = 'tanh';
let nnl: NeuralNetworkList | XGBoostEnsemble = new NeuralNetworkList(numOfNeuralNetworks, inputSize, hiddenLayerSizes, outputSize, activationFunction, outputActivationFunction);
let xgboost: XGBoostEnsemble | null = null;
let lastError = Infinity;
let topPanelScroll = 0;
let topPanelMaxScroll = 0;

canvas.addEventListener('wheel', (e) => {
    const rect = canvas.getBoundingClientRect();
    const mouseY = e.clientY - rect.top;
    const hch = canvas.height / 2;
    if (mouseY < hch && topPanelMaxScroll > 0) {
        e.preventDefault();
        // Normalize scroll delta across browsers/devices
        let delta = e.deltaY;
        if (e.deltaMode === 1) delta *= 30; // Line mode
        else if (e.deltaMode === 2) delta *= canvas.height; // Page mode
        topPanelScroll = Math.max(0, Math.min(topPanelMaxScroll, topPanelScroll + delta));
    }
}, { passive: false });

function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    update();
}

function start() {
    // Sync all JS state from the (reset) UI
    showDataFormat = (document.getElementById('showDataFormat') as HTMLSelectElement).value as ShowDataType;
    showNetworkFormat = (document.getElementById('showNetworkFormat') as HTMLSelectElement).value as ShowNetworkType;
    showTrainingData = (document.getElementById('showTrainingData') as HTMLSelectElement).value as ShowTrainingDataType;
    trainingMethod = (document.getElementById('trainingMethod') as HTMLSelectElement).value as TrainingMethod;
    const resEl = document.getElementById('xgbResolution') as HTMLSelectElement;
    if (resEl) xgbResolution = parseFloat(resEl.value);
    networkChange(); // sets up network format, creates trials, etc.
    updateSettingsVisibility();
}
function update() {
    if (isStarted){
        let bestFitness = 0;

        if (trainingMethod === 'genetic') {
            // Genetic algorithm training
            if (nnl instanceof NeuralNetworkList) {
                for (let i = 0; i < generationsPerDrawCycle; i++){
                    bestFitness = nnl.runGeneration(geneticMutationWeights, geneticMutationWeightStrength, geneticMutationBiases, geneticMutationBiasStrength);
                    if(bestFitness === lastError){
                        geneticMutationBiasStrength = Math.max(0.0001, geneticMutationBiasStrength / 1.001);
                        geneticMutationWeightStrength = Math.max(0.0001, geneticMutationWeightStrength / 1.001);
                    } else {
                        geneticMutationBiasStrength = Math.min(1, geneticMutationBiasStrength * 1.001);
                        geneticMutationWeightStrength = Math.min(1, geneticMutationWeightStrength * 1.001);
                    }
                }
            }
        } else if (trainingMethod === 'backprop') {
            // Update learning rate and momentum for all networks
            if (nnl instanceof NeuralNetworkList) {
                nnl.setLearningRate(learningRate);
                nnl.setMomentum(momentum);

                for (let i = 0; i < generationsPerDrawCycle; i++){
                    bestFitness = nnl.trainBackpropagation(1);
                }
            }
        } else if (trainingMethod === 'XGBoost') {
            // XGBoost: decision tree ensemble trained on residuals
            if (xgboost && nnl instanceof NeuralNetworkList) {
                for (let i = 0; i < generationsPerDrawCycle; i++){
                    xgboost.train(xgbTrainInputs, xgbTrainOutputs, testInputsGrid, test);
                }
                bestFitness = xgboost.trainRMSE;
            }
        }

        // Compute test error for neural network methods
        if (trainingMethod !== 'XGBoost' && nnl instanceof NeuralNetworkList) {
            nnl.computeTestErr((inputs) => {
                const result = nnl instanceof NeuralNetworkList ? nnl.neuralNetworks[0].run(inputs) : {neurons: [{value: 0}]};
                return (result as any).neurons.map((n: any) => n.value);
            });
        }

        lastError = bestFitness;
    }
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const padding = 0.001;

    const hcw = canvas.width / 2;
    const hch = canvas.height / 2;

    ctx.fillStyle = '#000000';
    ctx.fillRect(0, hch - padding * canvas.width, canvas.width, padding * canvas.width * 2);

    const displayErrorDigits = 5;
    const displayMeanError = true;
    const displayHeaderHeight = 50;
    const displayHeader = true;
    const rowSize = 4;
    const currentInputs = inputSize === 1 ? [input1] : [input1, input2];
    const xgbInput = currentInputs;
    const contentTop = displayHeaderHeight;
    const contentHeight = hch - displayHeaderHeight;

    // Always draw header first
    if (trainingMethod === 'XGBoost' && xgboost) {
        xgboost.drawHeader(ctx, 0, 0, canvas.width, displayHeaderHeight);
    } else if (nnl instanceof NeuralNetworkList) {
        nnl.drawHeader(ctx, 0, 0, canvas.width, displayHeaderHeight);
    }

    switch(showNetworkFormat){
        case "all":
            if (trainingMethod === 'XGBoost' && xgboost) {
                topPanelMaxScroll = Math.max(0, xgboost.getContentHeight(displayHeaderHeight, hch) - hch);
                topPanelScroll = Math.min(topPanelScroll, topPanelMaxScroll);
                xgboost.draw(ctx, 0, 0, canvas.width, hch, displayHeaderHeight, topPanelScroll, xgbInput);
            } else if (nnl instanceof NeuralNetworkList) {
                topPanelMaxScroll = Math.max(0, nnl.getContentHeight(rowSize, displayHeaderHeight, displayHeader, hch) - hch);
                topPanelScroll = Math.min(topPanelScroll, topPanelMaxScroll);
                nnl.draw(ctx, 0, 0, canvas.width, hch, rowSize, displayHeaderHeight, displayHeader, displayErrorDigits, displayMeanError, topPanelScroll);
            }
            break;
        case "base":
            if (xgboost) {
                topPanelMaxScroll = 0;
                topPanelScroll = 0;
                xgboost.drawSingleTree(ctx, 0, contentTop, canvas.width, contentHeight, 0, xgbInput);
            }
            break;
        case "latest":
            if (xgboost) {
                topPanelMaxScroll = 0;
                topPanelScroll = 0;
                xgboost.drawSingleTree(ctx, 0, contentTop, canvas.width, contentHeight, xgboost.trees.length - 1, xgbInput);
            }
            break;
        case "best":
            if (nnl instanceof NeuralNetworkList) {
                topPanelMaxScroll = 0;
                topPanelScroll = 0;
                let n = nnl.neuralNetworks[0].clone();
                n.run(currentInputs);
                n.draw(ctx, 0, contentTop, canvas.width, contentHeight, displayErrorDigits, displayMeanError);
            }
            break;
    }




    const axis1low = -1;
    const axis1high = 1;
    const axis2low = -1;
    const axis2high = 1;


    const decimals = -1;
    const rows = 101;
    const columns = 101;

    const ouputMiddle = 0;
    const outputRange = 1;
    //V for visible (as in the 11x11 one cuz you can see the values) not vendetta
    const decimalsV = 2;
    const rowsV = 11;
    const columnsV = 11;



    // Prediction function wrapper (handles both neural networks and XGBoost)
    const predict = (inputs: number[]): number[] => {
        if (trainingMethod === 'XGBoost' && xgboost) {
            return xgboost.predict(inputs);
        } else if (nnl instanceof NeuralNetworkList) {
            const result = nnl.neuralNetworks[0].run(inputs);
            return result.neurons.map(n => n.value);
        }
        return [0];
    };

    // Display based on network format
    if (nnl instanceof NeuralNetworkList) {
        const displayNetwork = nnl.neuralNetworks[0];
        const isXGB = trainingMethod === 'XGBoost' && xgboost;

        if (networkFormat === 'Val1in1Out') {
            const graphMargin = 15;
            const graphLeft = graphMargin;
            const graphTop = hch + 5;
            const graphWidth = canvas.width - graphMargin * 2;
            const graphHeight = hch - 10;
            const predictLine = (x: number) => predict([x])[0];
            const testLine = (x: number) => test([x])[0];
            const errorLine = (x: number) => predictLine(x) - testLine(x);

            switch(showDataFormat) {
                case "none":
                    drawLineGraph(ctx, graphLeft, graphTop, graphWidth, graphHeight,
                        axis1low, axis1high, -1.2, 1.2, []);
                    break;
                case "testvsoutput":
                    drawLineGraph(ctx, graphLeft, graphTop, graphWidth, graphHeight,
                        axis1low, axis1high, -1.2, 1.2, [
                            {fn: testLine, color: '#4CAF50', lineWidth: 1.5},
                            {fn: predictLine, color: '#2196F3', lineWidth: 2},
                        ]);
                    break;
                case "output":
                    drawLineGraph(ctx, graphLeft, graphTop, graphWidth, graphHeight,
                        axis1low, axis1high, -1.2, 1.2, [
                            {fn: predictLine, color: '#2196F3', lineWidth: 2},
                        ]);
                    break;
                case "test":
                    drawLineGraph(ctx, graphLeft, graphTop, graphWidth, graphHeight,
                        axis1low, axis1high, -1.2, 1.2, [
                            {fn: testLine, color: '#4CAF50', lineWidth: 2},
                        ]);
                    break;
                case "error":
                    drawLineGraph(ctx, graphLeft, graphTop, graphWidth, graphHeight,
                        axis1low, axis1high, -1.2, 1.2, [
                            {fn: errorLine, color: '#E53935', lineWidth: 2},
                        ]);
                    break;
            }
            // Overlay training data as dots and/or lines
            if (showTrainingData !== 'none' && nnl instanceof NeuralNetworkList) {
                const toSX = (v: number) => graphLeft + ((v - axis1low) / (axis1high - axis1low)) * graphWidth;
                const toSY = (v: number) => graphTop + graphHeight - ((v - (-1.2)) / (1.2 - (-1.2))) * graphHeight;
                let dotFn: ((inp: number[]) => number) | null = null;
                let dotColor = '#000';
                if (showTrainingData === 'output') { dotFn = (inp) => predict(inp)[0]; dotColor = '#2196F3'; }
                else if (showTrainingData === 'test') { dotFn = (inp) => test(inp)[0]; dotColor = '#4CAF50'; }
                else if (showTrainingData === 'error') { dotFn = (inp) => predict(inp)[0] - test(inp)[0]; dotColor = '#E53935'; }
                if (dotFn) {
                    const fn = dotFn;
                    ctx.fillStyle = dotColor;
                    nnl.trialInputsList.forEach((inp) => {
                        const sx = toSX(inp[0]);
                        const sy = toSY(Math.max(-1.2, Math.min(1.2, fn(inp))));
                        ctx.beginPath();
                        ctx.arc(sx, sy, 3, 0, 2 * Math.PI);
                        ctx.fill();
                    });
                }
            }
            // Draw input marker
            const markerX = graphLeft + ((input1 - axis1low) / (axis1high - axis1low)) * graphWidth;
            ctx.strokeStyle = 'rgba(0,0,0,0.4)';
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
            ctx.beginPath(); ctx.moveTo(markerX, graphTop); ctx.lineTo(markerX, graphTop + graphHeight); ctx.stroke();
            ctx.setLineDash([]);
        } else if (networkFormat === 'Val2in1out') {
            switch(showDataFormat){
                case "none":
                    ctx.fillRect(0, hch, hcw - padding * canvas.width, hch);
                    if (isXGB) {
                        displayNetwork.displayGrid2Input1Output(ctx, hcw + padding * canvas.width, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, rowsV, columnsV, decimalsV, true, true,
                            (in1, in2) => predict([in1, in2])[0],
                            (val) => val * outputRange - ouputMiddle);
                    } else {
                        displayNetwork.display2Input1Output(ctx, hcw + padding * canvas.width, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, ouputMiddle, outputRange, rowsV, columnsV,  decimalsV, true, true);
                    }
                    break;
                case "error":
                    let errorRange = 1;
                    if (isXGB) {
                        const errorFn = (in1: number, in2: number) => predict([in1, in2])[0] - test([in1, in2])[0];
                        displayNetwork.displayGrid2Input1Output(ctx, 0, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, rows, columns, decimals, false, false, errorFn, (val) => val * errorRange);
                        displayNetwork.displayGrid2Input1Output(ctx, hcw + padding * canvas.width, hch, hcw, hch, axis1low, axis2low, axis1high, axis2high, rowsV, columnsV, decimalsV, true, true, errorFn, (val) => val * errorRange);
                    } else {
                        displayNetwork.display2Input1OutputError(ctx, 0, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, errorRange, rows, columns, decimals, false, false, test);
                        displayNetwork.display2Input1OutputError(ctx, hcw + padding * canvas.width, hch, hcw, hch, axis1low, axis2low, axis1high, axis2high, errorRange, rowsV, columnsV, decimalsV, true, true, test);
                    }
                    break;
                case "output":
                    if (isXGB) {
                        const outputFn = (in1: number, in2: number) => predict([in1, in2])[0];
                        const colorFn = (val: number) => val * outputRange - ouputMiddle;
                        displayNetwork.displayGrid2Input1Output(ctx, 0, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, rows, columns, decimals, false, false, outputFn, colorFn);
                        displayNetwork.displayGrid2Input1Output(ctx, hcw + padding * canvas.width, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, rowsV, columnsV, decimalsV, true, true, outputFn, colorFn);
                    } else {
                        displayNetwork.display2Input1Output(ctx, 0, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, ouputMiddle, outputRange, rows, columns, decimals, false, false);
                        displayNetwork.display2Input1Output(ctx, hcw + padding * canvas.width, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, ouputMiddle, outputRange, rowsV, columnsV,  decimalsV, true, true);
                    }
                    break;
                case "test":
                    displayNetwork.display2Input1OutputTest(ctx, 0, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, ouputMiddle, outputRange, rows, columns, decimals, false, false, test);
                    displayNetwork.display2Input1OutputTest(ctx, hcw + padding * canvas.width, hch, hcw, hch, axis1low, axis2low, axis1high, axis2high,  ouputMiddle, outputRange, rowsV, columnsV, decimalsV, true, true, test);
                    break;
            }
        } else if (networkFormat === 'Cat2in2out') {
            switch(showDataFormat){
                case "none":
                    ctx.fillRect(0, hch, hcw - padding * canvas.width, hch);
                    if (isXGB) {
                        displayNetwork.displayGrid2Input2Output(ctx, hcw + padding * canvas.width, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, rowsV, columnsV, decimalsV, true, true, color1, color2,
                            (in1, in2) => predict([in1, in2]));
                    } else {
                        displayNetwork.display2Input2Output(ctx, hcw + padding * canvas.width, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, rowsV, columnsV, decimalsV, true, true, color1, color2);
                    }
                    break;
                case "error":
                    if (isXGB) {
                        const errorColor1 = {r: 255, g: 255, b: 255};
                        const errorColor2 = {r: 255, g: 0, b: 0};
                        const errorFn = (in1: number, in2: number) => {
                            let pred = predict([in1, in2]);
                            let expected = test([in1, in2]);
                            let error0 = Math.abs(pred[0] - expected[0]);
                            let error1 = Math.abs(pred[1] - expected[1]);
                            let totalError = (error0 + error1) / 2;
                            return [totalError, 0];
                        };
                        displayNetwork.displayGrid2Input2Output(ctx, 0, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, rows, columns, decimals, false, false, errorColor1, errorColor2, errorFn);
                        displayNetwork.displayGrid2Input2Output(ctx, hcw + padding * canvas.width, hch, hcw, hch, axis1low, axis2low, axis1high, axis2high, rowsV, columnsV, decimalsV, true, true, errorColor1, errorColor2, errorFn);
                    } else {
                        displayNetwork.display2Input2OutputError(ctx, 0, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, rows, columns, decimals, false, false, color1, color2, test);
                        displayNetwork.display2Input2OutputError(ctx, hcw + padding * canvas.width, hch, hcw, hch, axis1low, axis2low, axis1high, axis2high, rowsV, columnsV, decimalsV, true, true, color1, color2, test);
                    }
                    break;
                case "output":
                    if (isXGB) {
                        const outputFn = (in1: number, in2: number) => predict([in1, in2]);
                        displayNetwork.displayGrid2Input2Output(ctx, 0, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, rows, columns, decimals, false, false, color1, color2, outputFn);
                        displayNetwork.displayGrid2Input2Output(ctx, hcw + padding * canvas.width, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, rowsV, columnsV, decimalsV, true, true, color1, color2, outputFn);
                    } else {
                        displayNetwork.display2Input2Output(ctx, 0, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, rows, columns, decimals, false, false, color1, color2);
                        displayNetwork.display2Input2Output(ctx, hcw + padding * canvas.width, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, rowsV, columnsV, decimalsV, true, true, color1, color2);
                    }
                    break;
                case "test":
                    displayNetwork.display2Input2OutputTest(ctx, 0, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, rows, columns, decimals, false, false, color1, color2, test);
                    displayNetwork.display2Input2OutputTest(ctx, hcw + padding * canvas.width, hch, hcw, hch, axis1low, axis2low, axis1high, axis2high, rowsV, columnsV, decimalsV, true, true, color1, color2, test);
                    break;
            }
        } else if (networkFormat === 'CatNout') {
            const catColors = getCategoryColors(numCategories);
            const outputFn = (in1: number, in2: number) => predict([in1, in2]);
            const testFn = (in1: number, in2: number) => test([in1, in2]);
            switch(showDataFormat){
                case "none":
                    ctx.fillRect(0, hch, hcw - padding * canvas.width, hch);
                    drawNClassGrid(ctx, hcw + padding * canvas.width, hch, hcw - padding * canvas.width, hch,
                        axis1low, axis2low, axis1high, axis2high, rowsV, columnsV, true, outputFn, catColors);
                    break;
                case "output":
                    drawNClassGrid(ctx, 0, hch, hcw - padding * canvas.width, hch,
                        axis1low, axis2low, axis1high, axis2high, rows, columns, false, outputFn, catColors);
                    drawNClassGrid(ctx, hcw + padding * canvas.width, hch, hcw - padding * canvas.width, hch,
                        axis1low, axis2low, axis1high, axis2high, rowsV, columnsV, true, outputFn, catColors);
                    break;
                case "test":
                    drawNClassGrid(ctx, 0, hch, hcw - padding * canvas.width, hch,
                        axis1low, axis2low, axis1high, axis2high, rows, columns, false, testFn, catColors);
                    drawNClassGrid(ctx, hcw + padding * canvas.width, hch, hcw - padding * canvas.width, hch,
                        axis1low, axis2low, axis1high, axis2high, rowsV, columnsV, true, testFn, catColors);
                    break;
                case "error": {
                    const errFn = (in1: number, in2: number) => {
                        const pred = predict([in1, in2]);
                        const expected = test([in1, in2]);
                        let sum = 0;
                        for (let k = 0; k < pred.length; k++) sum += Math.abs(pred[k] - expected[k]);
                        const err = sum / pred.length;
                        // Return as single-class with error as confidence
                        return [err];
                    };
                    const errColors = [{r: 255, g: 0, b: 0}];
                    drawNClassGrid(ctx, 0, hch, hcw - padding * canvas.width, hch,
                        axis1low, axis2low, axis1high, axis2high, rows, columns, false, errFn, errColors);
                    drawNClassGrid(ctx, hcw + padding * canvas.width, hch, hcw - padding * canvas.width, hch,
                        axis1low, axis2low, axis1high, axis2high, rowsV, columnsV, true, errFn, errColors);
                    break;
                }
            }
        }
    }
    // Show training data points overlay (for 2-input formats)
    if (nnl instanceof NeuralNetworkList && showTrainingData !== 'none' && inputSize >= 2) {
        const dpLeft = 0, dpTop = hch, dpWidth = hcw - padding * canvas.width, dpHeight = hch;
        if (networkFormat === 'Val2in1out') {
            switch(showTrainingData){
                case "error":
                    nnl.display2Input1OutputDataPoints((inputs) => [predict(inputs)[0] - test(inputs)[0]], ctx, dpLeft, dpTop, dpWidth, dpHeight, axis1low, axis2low, axis1high, axis2high, ouputMiddle, outputRange);
                    break;
                case "output":
                    nnl.display2Input1OutputDataPoints((inputs) => [predict(inputs)[0]], ctx, dpLeft, dpTop, dpWidth, dpHeight, axis1low, axis2low, axis1high, axis2high, ouputMiddle, outputRange);
                    break;
                case "test":
                    nnl.display2Input1OutputDataPoints(test, ctx, dpLeft, dpTop, dpWidth, dpHeight, axis1low, axis2low, axis1high, axis2high, ouputMiddle, outputRange);
                    break;
            }
        } else if (networkFormat === 'Cat2in2out') {
            switch(showTrainingData){
                case "error":
                    nnl.display2Input2OutputDataPoints((inputs) => {
                        let p = predict(inputs), t = test(inputs);
                        return [(Math.abs(p[0]-t[0]) + Math.abs(p[1]-t[1])) / 2, 0];
                    }, ctx, dpLeft, dpTop, dpWidth, dpHeight, axis1low, axis2low, axis1high, axis2high, {r:255,g:255,b:255}, {r:255,g:0,b:0});
                    break;
                case "output":
                    nnl.display2Input2OutputDataPoints((inputs) => predict(inputs), ctx, dpLeft, dpTop, dpWidth, dpHeight, axis1low, axis2low, axis1high, axis2high, color1, color2);
                    break;
                case "test":
                    nnl.display2Input2OutputDataPoints(test, ctx, dpLeft, dpTop, dpWidth, dpHeight, axis1low, axis2low, axis1high, axis2high, color1, color2);
                    break;
            }
        } else if (networkFormat === 'CatNout') {
            const catColors = getCategoryColors(numCategories);
            ctx.save();
            ctx.beginPath();
            ctx.rect(dpLeft, dpTop, dpWidth, dpHeight);
            ctx.clip();
            let dotFn: (inputs: number[]) => number[] = test;
            if (showTrainingData === 'output') dotFn = (inputs) => predict(inputs);
            else if (showTrainingData === 'error') dotFn = (inputs) => {
                let p = predict(inputs), t = test(inputs);
                let sum = 0;
                for (let k = 0; k < p.length; k++) sum += Math.abs(p[k] - t[k]);
                return [sum / p.length];
            };
            nnl.trialInputsList.forEach((input) => {
                const sx = dpLeft + ((input[0] - axis1low) / (axis1high - axis1low)) * dpWidth;
                const sy = dpTop + ((input[1] - axis2low) / (axis2high - axis2low)) * dpHeight;
                const out = dotFn(input);
                let maxIdx = 0;
                for (let k = 1; k < out.length; k++) if (out[k] > out[maxIdx]) maxIdx = k;
                const col = catColors[maxIdx % catColors.length];
                ctx.fillStyle = '#000';
                ctx.beginPath(); ctx.arc(sx, sy, 4, 0, 2 * Math.PI); ctx.fill();
                ctx.fillStyle = `rgb(${col.r},${col.g},${col.b})`;
                ctx.beginPath(); ctx.arc(sx, sy, 3, 0, 2 * Math.PI); ctx.fill();
            });
            ctx.restore();
        }
    }


}
let testInputsGrid: number[][] = [];
let xgbTrainInputs: number[][] = [];
let xgbTrainOutputs: number[][] = [];

function createTrials(){
    const inputs = createInputs(inputSize, 1, -1, 0.1);
    // Test inputs: offset grid that doesn't overlap training data
    testInputsGrid = createInputs(inputSize, 0.95, -0.95, 0.1);
    if (nnl instanceof NeuralNetworkList) {
        nnl.createTrials(inputs, test);
        nnl.testInputs = testInputsGrid;
        nnl.testFn = test;
    }
    // XGBoost training grid at configured resolution
    xgbTrainInputs = createInputs(inputSize, 1, -1, xgbResolution);
    xgbTrainOutputs = xgbTrainInputs.map(inp => test(inp));
}
function createInputs(numOfInputs: number, high: number, low: number, spacing: number): number[][] {
    const possibleValues: number[] = [];
    for (let v = low; v <= high; v += spacing) {
        possibleValues.push(v);
    }

    const result: number[][] = [];

    function generateCombinations(current: number[]) {
        if (current.length === numOfInputs) {
            result.push([...current]);
            return;
        }

        for (const value of possibleValues) {
            current.push(value);
            generateCombinations(current);
            current.pop();
        }
    }

    generateCombinations([]);
    return result;
}

function test(inputs: number[]): number[] {
    switch(networkFormat){
        case "Val1in1Out": {
            const x = inputs[0];
            switch(testFunctionVal1in1out){
                case "sine":
                    return [Math.sin(x * Math.PI)];
                case "square":
                    return [Math.sin(x * Math.PI * 2) > 0 ? 1 : -1];
                case "sawtooth":
                    return [((x + 1) % 0.5) * 4 - 1];
                case "triangle":
                    return [1 - 4 * Math.abs(Math.round(x * 0.5) - x * 0.5)];
                case "abs":
                    return [Math.abs(x) * 2 - 1];
                case "cubic":
                    return [Math.max(-1, Math.min(1, x ** 3 * 4))];
                case "polynomial":
                    return [Math.max(-1, Math.min(1, 2 * x * x * x - 3 * x * x + x + 0.5))];
                case "step":
                    return [x < -0.5 ? -1 : x < 0 ? -0.3 : x < 0.5 ? 0.3 : 1];
                case "gaussian":
                    return [Math.exp(-x * x * 5) * 2 - 1];
                case "tanh":
                    return [Math.tanh(x * 3)];
                case "sinc":
                    return [x === 0 ? 1 : Math.sin(x * 6) / (x * 6)];
                case "noise":
                    return [Math.sin(x * Math.PI) * 0.7 + Math.sin(x * 7) * 0.3];
            }
            break;
        }
        case "Val2in1out": {
            const x = inputs[0], y = inputs[1];
            switch(testFunctionVal2in1out){
                case "wave":
                    return [x > Math.sin(y * 2 * Math.PI) ? 1 : -1];
                case "radial":
                    return [Math.max(Math.min(1 - 2 * (x ** 2 + y ** 2), 1), -1)];
                case "xy":
                    return [x * y];
                case "checkerboard":
                    return [Math.floor(x * 4) % 2 === Math.floor(y * 4) % 2 ? 1 : -1];
                case "spiral": {
                    let a = Math.atan2(y, x);
                    let r = Math.sqrt(x ** 2 + y ** 2);
                    return [Math.sin(a * 3 + r * 5) > 0 ? 1 : -1];
                }
                case "diagonal":
                    return [x + y > 0 ? 1 : -1];
                case "gaussian2d":
                    return [Math.exp(-(x * x + y * y) * 3) * 2 - 1];
                case "saddle":
                    return [Math.max(-1, Math.min(1, x * x - y * y))];
                case "ripple": {
                    let d = Math.sqrt(x * x + y * y);
                    return [Math.sin(d * 8) * Math.exp(-d * 2)];
                }
                case "peaks":
                    return [Math.max(-1, Math.min(1,
                        Math.exp(-((x - 0.5) ** 2 + y ** 2) * 5) -
                        Math.exp(-((x + 0.5) ** 2 + y ** 2) * 5)))];
                case "step2d":
                    return [x > 0.3 ? 1 : x < -0.3 ? -1 : y > 0 ? 0.5 : -0.5];
                case "swiss":
                    return [Math.sin(x * 3 + y * 2) * Math.cos(x * 2 - y * 3) > 0 ? 1 : -1];
            }
            break;
        }
        case "Cat2in2out": {
            const x = inputs[0], y = inputs[1];
            switch(testFunctionCat2in2out){
                case "circle":
                    return x ** 2 + y ** 2 < 0.5 ? [1, 0] : [0, 1];
                case "square":
                    return Math.abs(x) < 0.5 && Math.abs(y) < 0.5 ? [1, 0] : [0, 1];
                case "quadrants":
                    return x * y > 0 ? [1, 0] : [0, 1];
                case "donut": {
                    let r = x ** 2 + y ** 2;
                    return r > 0.25 && r < 0.75 ? [1, 0] : [0, 1];
                }
                case "xor":
                    return (x > 0) !== (y > 0) ? [1, 0] : [0, 1];
                case "diagonal":
                    return x > y ? [1, 0] : [0, 1];
                case "stripes":
                    return Math.sin(x * 6) > 0 ? [1, 0] : [0, 1];
                case "checkerboard":
                    return (Math.floor((x + 1) * 3) + Math.floor((y + 1) * 3)) % 2 === 0 ? [1, 0] : [0, 1];
                case "spiral": {
                    let a = Math.atan2(y, x);
                    let r = Math.sqrt(x ** 2 + y ** 2);
                    return Math.sin(a * 2 + r * 6) > 0 ? [1, 0] : [0, 1];
                }
                case "moons":
                    return (x ** 2 + (y - 0.3) ** 2 < 0.6 && x ** 2 + (y + 0.3) ** 2 > 0.3) ? [1, 0] : [0, 1];
                case "diamond":
                    return Math.abs(x) + Math.abs(y) < 0.7 ? [1, 0] : [0, 1];
                case "cross":
                    return (Math.abs(x) < 0.2 || Math.abs(y) < 0.2) ? [1, 0] : [0, 1];
            }
            break;
        }
        case "CatNout": {
            const x = inputs[0], y = inputs[1];
            const n = numCategories;
            const oneHot = (idx: number) => {
                const arr = new Array(n).fill(0);
                arr[Math.min(Math.max(idx, 0), n - 1)] = 1;
                return arr;
            };
            switch(testFunctionCatNout){
                case "sectors": {
                    let a = Math.atan2(y, x);
                    return oneHot(Math.floor(((a + Math.PI) / (2 * Math.PI)) * n));
                }
                case "rings": {
                    let dist = Math.sqrt(x ** 2 + y ** 2);
                    return oneHot(Math.floor(dist * n / 1.5));
                }
                case "grid": {
                    let s = Math.ceil(Math.sqrt(n));
                    let gx = Math.floor((x + 1) / 2 * s);
                    let gy = Math.floor((y + 1) / 2 * s);
                    return oneHot((gy * s + gx) % n);
                }
                case "spiral": {
                    let a = Math.atan2(y, x);
                    let r = Math.sqrt(x ** 2 + y ** 2);
                    return oneHot(Math.floor(((a + Math.PI + r * 4) % (2 * Math.PI)) / (2 * Math.PI) * n));
                }
                case "stripes":
                    return oneHot(Math.floor(((x + 1) / 2) * n));
                case "checkerboard": {
                    let s = Math.ceil(Math.sqrt(n));
                    let cx = Math.floor((x + 1) / 2 * s);
                    let cy = Math.floor((y + 1) / 2 * s);
                    return oneHot((cx + cy) % n);
                }
                case "voronoi": {
                    // Fixed random-ish seed points per class
                    const centers = Array.from({length: n}, (_, i) => ({
                        x: Math.cos(i * 2.39996) * 0.6,
                        y: Math.sin(i * 2.39996) * 0.6
                    }));
                    let minDist = Infinity, minIdx = 0;
                    for (let i = 0; i < n; i++) {
                        let d = (x - centers[i].x) ** 2 + (y - centers[i].y) ** 2;
                        if (d < minDist) { minDist = d; minIdx = i; }
                    }
                    return oneHot(minIdx);
                }
                case "waves":
                    return oneHot(Math.floor(((Math.sin(x * 4) + Math.sin(y * 4) + 2) / 4) * n));
            }
            break;
        }
    }
    return new Array(outputSize).fill(0);
}

///////////////////////MAIN AREA//////////////////////////////////
///////////////////////UI AREA////////////////////////////////////
function startToggle(){
    const startButton = document.getElementById('startButton') as HTMLButtonElement;
    const startToggleText = document.getElementById('startToggleText') as HTMLParagraphElement;
    if (isStarted){
        startToggleText.innerHTML = 'Start';
    }else {
        startToggleText.innerHTML = 'Stop';
    }
    isStarted = !isStarted;
}
(window as any).startToggle = startToggle;
function networkChange(){
    networkFormat = (document.getElementById('networkFormat') as HTMLSelectElement).value as networkType;

    // Update test function dropdown options
    const testFunctionDropdown = document.getElementById('testFunction') as HTMLSelectElement;
    testFunctionDropdown.innerHTML = '';

    switch (networkFormat){
        case 'Val1in1Out':
            inputSize = 1;
            outputSize = 1;
            hiddenLayerSizes = [7, 10, 20, 20, 10, 7];
            activationFunction = 'relu';
            outputActivationFunction = 'tanh';

            testFunctionDropdown.innerHTML = `
                <option value="sine">Sine</option>
                <option value="square">Square Wave</option>
                <option value="sawtooth">Sawtooth</option>
                <option value="triangle">Triangle</option>
                <option value="abs">Abs</option>
                <option value="cubic">Cubic</option>
                <option value="polynomial">Polynomial</option>
                <option value="step">Step</option>
                <option value="gaussian">Gaussian</option>
                <option value="tanh">Tanh</option>
                <option value="sinc">Sinc</option>
                <option value="noise">Noisy Sine</option>
            `;
            testFunctionDropdown.value = testFunctionVal1in1out;
            break;
        case 'Val2in1out':
            inputSize = 2;
            outputSize = 1;
            hiddenLayerSizes = [7, 10, 20, 20, 10, 7];
            activationFunction = 'relu';
            outputActivationFunction = 'tanh';

            // Add Val2in1out options
            testFunctionDropdown.innerHTML = `
                <option value="wave">Wave</option>
                <option value="radial">Radial</option>
                <option value="xy">XY Product</option>
                <option value="checkerboard">Checkerboard</option>
                <option value="spiral">Spiral</option>
                <option value="diagonal">Diagonal</option>
                <option value="gaussian2d">Gaussian</option>
                <option value="saddle">Saddle</option>
                <option value="ripple">Ripple</option>
                <option value="peaks">Peaks</option>
                <option value="step2d">Step 2D</option>
                <option value="swiss">Swiss Roll</option>
            `;
            testFunctionDropdown.value = testFunctionVal2in1out;
            break;
        case 'Cat2in2out':
            inputSize = 2;
            outputSize = 2;
            hiddenLayerSizes = [7, 10, 20, 20, 10, 7];
            activationFunction = 'relu';
            outputActivationFunction = 'sigmoid';

            // Add Cat2in2out options
            testFunctionDropdown.innerHTML = `
                <option value="circle">Circle</option>
                <option value="square">Square</option>
                <option value="quadrants">Quadrants</option>
                <option value="donut">Donut</option>
                <option value="xor">XOR</option>
                <option value="diagonal">Diagonal</option>
                <option value="stripes">Stripes</option>
                <option value="checkerboard">Checkerboard</option>
                <option value="spiral">Spiral</option>
                <option value="moons">Moons</option>
                <option value="diamond">Diamond</option>
                <option value="cross">Cross</option>
            `;
            testFunctionDropdown.value = testFunctionCat2in2out;
            break;
        case 'CatNout':
            inputSize = 2;
            outputSize = numCategories;
            hiddenLayerSizes = [10, 20, 20, 10];
            activationFunction = 'relu';
            outputActivationFunction = 'sigmoid';

            testFunctionDropdown.innerHTML = `
                <option value="sectors">Sectors</option>
                <option value="rings">Rings</option>
                <option value="grid">Grid</option>
                <option value="spiral">Spiral</option>
                <option value="stripes">Stripes</option>
                <option value="checkerboard">Checkerboard</option>
                <option value="voronoi">Voronoi</option>
                <option value="waves">Waves</option>
            `;
            testFunctionDropdown.value = testFunctionCatNout;
            break;
    }

    // Show/hide categories slider and input2
    (document.getElementById('numCategoriesGroup') as HTMLDivElement).style.display =
        networkFormat === 'CatNout' ? '' : 'none';
    (document.getElementById('input2Group') as HTMLDivElement).style.display =
        inputSize >= 2 ? '' : 'none';

    // Update data format dropdown for 1in1out (has extra "Test vs Output" option)
    const dataFormatDropdown = document.getElementById('showDataFormat') as HTMLSelectElement;
    if (networkFormat === 'Val1in1Out') {
        dataFormatDropdown.innerHTML = `
            <option value="testvsoutput">Test vs Output</option>
            <option value="output">Output</option>
            <option value="test">Test</option>
            <option value="error">Error</option>
            <option value="none">None</option>
        `;
        showDataFormat = 'testvsoutput';
        dataFormatDropdown.value = 'testvsoutput';
    } else {
        dataFormatDropdown.innerHTML = `
            <option value="none">None</option>
            <option value="output" selected>Output</option>
            <option value="error">Error</option>
            <option value="test">Test</option>
        `;
        if (showDataFormat === 'testvsoutput') showDataFormat = 'output';
        dataFormatDropdown.value = showDataFormat;
    }

    // Set number of networks based on training method
    if (trainingMethod === 'backprop') {
        numOfNeuralNetworks = 1;
    } else if (trainingMethod === 'XGBoost') {
        numOfNeuralNetworks = 1; // XGBoost starts with 1 and grows
    } else {
        numOfNeuralNetworks = 16;
    }

    nnl = new NeuralNetworkList(numOfNeuralNetworks, inputSize, hiddenLayerSizes, outputSize, activationFunction, outputActivationFunction);
    if (trainingMethod === 'XGBoost') {
        xgboost = new XGBoostEnsemble(inputSize, outputSize, xgbShrinkage, xgbMaxDepth, xgbMaxTrees, networkFormat !== 'Val2in1out' && networkFormat !== 'Val1in1Out');
    }
    createTrials();
}
(window as any).networkChange = networkChange;
function showDataChange(){
    showDataFormat = (document.getElementById('showDataFormat') as HTMLSelectElement).value as ShowDataType;
}
(window as any).showDataChange = showDataChange;
function showNetworkChange(){
    showNetworkFormat = (document.getElementById('showNetworkFormat') as HTMLSelectElement).value as ShowNetworkType;
}
(window as any).showNetworkChange = showNetworkChange;
function testFunctionChange(){
    const value = (document.getElementById('testFunction') as HTMLSelectElement).value;
    if (networkFormat === 'Val1in1Out') {
        testFunctionVal1in1out = value as TestFunctionTypeVal1;
    } else if (networkFormat === 'Val2in1out') {
        testFunctionVal2in1out = value as TestFunctionTypeVal2;
    } else if (networkFormat === 'Cat2in2out') {
        testFunctionCat2in2out = value as TestFunctionTypeCat2in2Out;
    } else if (networkFormat === 'CatNout') {
        testFunctionCatNout = value as TestFunctionTypeCatN;
    }
    createTrials();
}
(window as any).testFunctionChange = testFunctionChange;

function numCategoriesChange(){
    const slider = document.getElementById('numCategoriesSlider') as HTMLInputElement;
    const input = document.getElementById('numCategoriesInput') as HTMLInputElement;
    const display = document.getElementById('numCategoriesDisplay') as HTMLSpanElement;
    // Sync slider and input
    if (document.activeElement === slider) {
        input.value = slider.value;
    } else {
        slider.value = input.value;
    }
    numCategories = parseInt(slider.value);
    display.textContent = numCategories.toString();
    if (networkFormat === 'CatNout') {
        networkChange();
    }
}
(window as any).numCategoriesChange = numCategoriesChange;
function inputChange(){
    generationsPerDrawCycle = parseFloat((document.getElementById('generationsPerDrawCycle') as HTMLInputElement).value);
    learningRate = parseFloat((document.getElementById('learningRate') as HTMLInputElement).value);
    momentum = parseFloat((document.getElementById('momentum') as HTMLInputElement).value);
    let i1 = parseFloat((document.getElementById('input1') as HTMLInputElement).value);
    input1 = Number(isNaN(i1)? 0: i1);
    let i2 = parseFloat((document.getElementById('input2') as HTMLInputElement).value);
    input2 = Number(isNaN(i2)? 0: i2);
    (document.getElementById('input1Slider') as HTMLInputElement).value = input1.toString();
    (document.getElementById('input2Slider') as HTMLInputElement).value = input2.toString();
    (document.getElementById('generationsPerDrawCycleSlider') as HTMLInputElement).value = generationsPerDrawCycle.toString();
    (document.getElementById('learningRateSlider') as HTMLInputElement).value = learningRate.toString();
    (document.getElementById('momentumSlider') as HTMLInputElement).value = momentum.toString();
    // Update display spans
    (document.getElementById('learningRateDisplay') as HTMLSpanElement).textContent = learningRate.toFixed(3);
    (document.getElementById('momentumDisplay') as HTMLSpanElement).textContent = momentum.toFixed(2);
}
(window as any).inputChange = inputChange;
function inputSliderChange(){
    generationsPerDrawCycle = parseFloat((document.getElementById('generationsPerDrawCycleSlider') as HTMLInputElement).value);
    learningRate = parseFloat((document.getElementById('learningRateSlider') as HTMLInputElement).value);
    momentum = parseFloat((document.getElementById('momentumSlider') as HTMLInputElement).value);
    input1 = parseFloat((document.getElementById('input1Slider') as HTMLInputElement).value);
    input2 = parseFloat((document.getElementById('input2Slider') as HTMLInputElement).value);
    (document.getElementById('input1') as HTMLInputElement).value = input1.toString();
    (document.getElementById('input2') as HTMLInputElement).value = input2.toString();
    (document.getElementById('generationsPerDrawCycle') as HTMLInputElement).value = generationsPerDrawCycle.toString();
    (document.getElementById('learningRate') as HTMLInputElement).value = learningRate.toString();
    (document.getElementById('momentum') as HTMLInputElement).value = momentum.toString();
    // Update display spans
    (document.getElementById('learningRateDisplay') as HTMLSpanElement).textContent = learningRate.toFixed(3);
    (document.getElementById('momentumDisplay') as HTMLSpanElement).textContent = momentum.toFixed(2);
}
(window as any).inputSliderChange = inputSliderChange;
function reset(){
    topPanelScroll = 0;
    if (trainingMethod === 'XGBoost') {
        nnl = new NeuralNetworkList(1, inputSize, hiddenLayerSizes, outputSize, activationFunction, outputActivationFunction);
        xgboost = new XGBoostEnsemble(inputSize, outputSize, xgbShrinkage, xgbMaxDepth, xgbMaxTrees, networkFormat !== 'Val2in1out' && networkFormat !== 'Val1in1Out');
    } else {
        nnl = new NeuralNetworkList(numOfNeuralNetworks, inputSize, hiddenLayerSizes, outputSize, activationFunction, outputActivationFunction);
        xgboost = null;
    }
    createTrials();
}
(window as any).reset = reset;
function showTrainingDataChange(){
    showTrainingData = (document.getElementById('showTrainingData') as HTMLSelectElement).value as ShowTrainingDataType;
}
(window as any).showTrainingDataChange = showTrainingDataChange;
function trainingMethodChange(){
    const newTrainingMethod = (document.getElementById('trainingMethod') as HTMLSelectElement).value as TrainingMethod;

    // Switching to or from XGBoost requires recreating the model
    if (newTrainingMethod === 'XGBoost' && trainingMethod !== 'XGBoost') {
        // Switching TO XGBoost: create XGBoost ensemble
        nnl = new NeuralNetworkList(1, inputSize, hiddenLayerSizes, outputSize, activationFunction, outputActivationFunction);
        xgboost = new XGBoostEnsemble(inputSize, outputSize, xgbShrinkage, xgbMaxDepth, xgbMaxTrees, networkFormat !== 'Val2in1out' && networkFormat !== 'Val1in1Out');
        createTrials();
        numOfNeuralNetworks = 1;
    } else if (newTrainingMethod !== 'XGBoost' && trainingMethod === 'XGBoost') {
        // Switching FROM XGBoost: recreate neural network list
        const newRequired = newTrainingMethod === 'genetic' ? 16 : 1;
        nnl = new NeuralNetworkList(newRequired, inputSize, hiddenLayerSizes, outputSize, activationFunction, outputActivationFunction);
        xgboost = null;
        createTrials();
        numOfNeuralNetworks = newRequired;
    } else if (nnl instanceof NeuralNetworkList) {
        // Switching between genetic and backprop
        const getRequiredNetworks = (method: TrainingMethod) => {
            if (method === 'genetic') return 16;
            if (method === 'backprop') return 1;
            return 16;
        };

        const oldRequired = getRequiredNetworks(trainingMethod);
        const newRequired = getRequiredNetworks(newTrainingMethod);

        if (newRequired !== oldRequired) {
            if (newRequired > oldRequired) {
                // Need more networks: clone the best one
                nnl.resetError();
                nnl.testErrorTrials(nnl.trialInputsList, nnl.trialOutputsList, nnl.trialPower);
                nnl.sort();

                const bestNetwork = nnl.neuralNetworks[0].clone();
                const newNetworks = [];
                for (let i = 0; i < newRequired; i++) {
                    newNetworks.push(bestNetwork.clone());
                }
                nnl.neuralNetworks = newNetworks;
            } else {
                // Need fewer networks: keep only the best ones
                nnl.resetError();
                nnl.testErrorTrials(nnl.trialInputsList, nnl.trialOutputsList, nnl.trialPower);
                nnl.sort();
                nnl.neuralNetworks = nnl.neuralNetworks.slice(0, newRequired);
            }

            nnl.numOfNeuralNetworks = newRequired;
            numOfNeuralNetworks = newRequired;
        }
    }

    trainingMethod = newTrainingMethod;
    topPanelScroll = 0;
    updateSettingsVisibility();
}
(window as any).trainingMethodChange = trainingMethodChange;

function updateSettingsVisibility() {
    const backpropSettings = document.getElementById('backpropSettings') as HTMLDivElement;
    const xgboostSettings = document.getElementById('xgboostSettings') as HTMLDivElement;
    const viewDropdown = document.getElementById('showNetworkFormat') as HTMLSelectElement;
    const viewLabel = viewDropdown.parentElement?.querySelector('label');

    if (trainingMethod === 'XGBoost') {
        backpropSettings.style.display = 'none';
        xgboostSettings.style.display = '';
        if (viewLabel) viewLabel.textContent = 'Tree View';
        viewDropdown.innerHTML = `
            <option value="base">Base Tree</option>
            <option value="latest">Latest Tree</option>
            <option value="all">All Trees</option>
        `;
        viewDropdown.value = 'base';
        showNetworkFormat = 'base';
    } else {
        backpropSettings.style.display = trainingMethod === 'backprop' ? '' : 'none';
        xgboostSettings.style.display = 'none';
        if (viewLabel) viewLabel.textContent = 'Network View';
        viewDropdown.innerHTML = `
            <option value="best">Best</option>
            <option value="all">All</option>
        `;
        viewDropdown.value = showNetworkFormat === 'all' ? 'all' : 'best';
        if (showNetworkFormat !== 'all') showNetworkFormat = 'best';
    }
    topPanelScroll = 0;
}

function limitTreesToggleChange() {
    xgbLimitTrees = (document.getElementById('limitTreesToggle') as HTMLInputElement).checked;
    const sliderGroup = document.getElementById('maxTreesSliderGroup') as HTMLDivElement;
    const display = document.getElementById('maxTreesDisplay') as HTMLSpanElement;
    if (xgbLimitTrees) {
        sliderGroup.style.display = '';
        xgbMaxTrees = parseInt((document.getElementById('maxTreesSlider') as HTMLInputElement).value);
        display.textContent = xgbMaxTrees.toString();
    } else {
        sliderGroup.style.display = 'none';
        xgbMaxTrees = Infinity;
        display.textContent = '\u221E';
    }
    if (xgboost) xgboost.maxTrees = xgbMaxTrees;
}
(window as any).limitTreesToggleChange = limitTreesToggleChange;

function xgboostSliderChange() {
    if (xgbLimitTrees) {
        xgbMaxTrees = parseInt((document.getElementById('maxTreesSlider') as HTMLInputElement).value);
        (document.getElementById('maxTrees') as HTMLInputElement).value = xgbMaxTrees.toString();
        (document.getElementById('maxTreesDisplay') as HTMLSpanElement).textContent = xgbMaxTrees.toString();
    }
    xgbShrinkage = parseFloat((document.getElementById('shrinkageSlider') as HTMLInputElement).value);
    xgbMaxDepth = parseInt((document.getElementById('maxDepthSlider') as HTMLInputElement).value);
    (document.getElementById('shrinkage') as HTMLInputElement).value = xgbShrinkage.toString();
    (document.getElementById('maxDepth') as HTMLInputElement).value = xgbMaxDepth.toString();
    (document.getElementById('shrinkageDisplay') as HTMLSpanElement).textContent = xgbShrinkage.toFixed(2);
    (document.getElementById('maxDepthDisplay') as HTMLSpanElement).textContent = xgbMaxDepth.toString();
    if (xgboost) {
        xgboost.maxTrees = xgbMaxTrees;
        xgboost.shrinkage = xgbShrinkage;
        xgboost.maxDepth = xgbMaxDepth;
    }
}
(window as any).xgboostSliderChange = xgboostSliderChange;

function xgboostInputChange() {
    if (xgbLimitTrees) {
        xgbMaxTrees = parseInt((document.getElementById('maxTrees') as HTMLInputElement).value);
        (document.getElementById('maxTreesSlider') as HTMLInputElement).value = xgbMaxTrees.toString();
        (document.getElementById('maxTreesDisplay') as HTMLSpanElement).textContent = xgbMaxTrees.toString();
    }
    xgbShrinkage = parseFloat((document.getElementById('shrinkage') as HTMLInputElement).value);
    xgbMaxDepth = parseInt((document.getElementById('maxDepth') as HTMLInputElement).value);
    (document.getElementById('shrinkageSlider') as HTMLInputElement).value = xgbShrinkage.toString();
    (document.getElementById('maxDepthSlider') as HTMLInputElement).value = xgbMaxDepth.toString();
    (document.getElementById('shrinkageDisplay') as HTMLSpanElement).textContent = xgbShrinkage.toFixed(2);
    (document.getElementById('maxDepthDisplay') as HTMLSpanElement).textContent = xgbMaxDepth.toString();
    if (xgboost) {
        xgboost.maxTrees = xgbMaxTrees;
        xgboost.shrinkage = xgbShrinkage;
        xgboost.maxDepth = xgbMaxDepth;
    }
}
(window as any).xgboostInputChange = xgboostInputChange;

function xgbResolutionChange() {
    xgbResolution = parseFloat((document.getElementById('xgbResolution') as HTMLSelectElement).value);
    (document.getElementById('xgbResolutionDisplay') as HTMLSpanElement).textContent = xgbResolution.toString();
    // Rebuild training data and reset XGBoost
    xgbTrainInputs = createInputs(inputSize, 1, -1, xgbResolution);
    xgbTrainOutputs = xgbTrainInputs.map(inp => test(inp));
    if (xgboost) {
        xgboost = new XGBoostEnsemble(inputSize, outputSize, xgbShrinkage, xgbMaxDepth, xgbMaxTrees, networkFormat !== 'Val2in1out' && networkFormat !== 'Val1in1Out');
    }
}
(window as any).xgbResolutionChange = xgbResolutionChange;
///////////////////////UI AREA////////////////////////////////////
