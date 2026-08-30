import { drawCircle, Point } from './graphics.js';
////////////////////////////////////////Neural Network List///////////////////////////////////////
export class NeuralNetworkList {
    numOfNeuralNetworks = 0;
    neuralNetworks: NeuralNetwork[] = [];
    generation: number = 0;
    inputSize: number;
    outputSize: number;
    trainRMSE: number = 0;
    trainMAE: number = 0;
    testRMSE: number = 0;
    testMAE: number = 0;

    trialInputsList: number[][] = [];
    trialOutputsList: number[][] = [];
    trialPower: number = 2;
    trainingData = [];
    testInputs: number[][] = [];
    testFn: ((inputs: number[]) => number[]) | null = null;
    constructor(numOfNeuralNetworks:number, inputSize: number, hiddenLayerSizes: number[], outputSize: number, activationFunction: ActivationFunction, outputActivationFunction:ActivationFunction){
        this.numOfNeuralNetworks = numOfNeuralNetworks;
        for(let i = 0; i < numOfNeuralNetworks; i++){
            const nn = new NeuralNetwork(inputSize, hiddenLayerSizes, outputSize, activationFunction, outputActivationFunction);
            this.neuralNetworks.push(nn);
        }
        this.inputSize = inputSize;
        this.outputSize = outputSize;

        
    }
    public run(inputs: number []){
        this.neuralNetworks.forEach(nn => nn.run(inputs));
    }
    public mutate(numOfWeights:number, weightStrength:number, numOfBiases:number, biasesStrength:number){
        this.neuralNetworks.forEach(nn => nn.mutate(numOfWeights,weightStrength, numOfBiases, biasesStrength));
        return this;
    }
    static MIN_CELL_HEIGHT = 200;

    public getContentHeight(rowSize: number, displayHeaderHeight: number, displayHeader: boolean, panelHeight: number): number {
        const headerH = displayHeader ? displayHeaderHeight : 0;
        const availableHeight = panelHeight - headerH;
        const numRows = Math.ceil(this.neuralNetworks.length / rowSize);
        const cellHeight = Math.max(NeuralNetworkList.MIN_CELL_HEIGHT, availableHeight / numRows);
        return headerH + numRows * cellHeight;
    }

    public draw(ctx: CanvasRenderingContext2D, left: number, top: number, width: number, height: number, rowSize: number, displayHeaderHeight?:number, displayHeader?:boolean, displayErrorDigits?: number, displayMeanError?: boolean, scrollY: number = 0){
        let xSpace = width / rowSize;
        const headerH = displayHeader ? displayHeaderHeight! : 0;
        const availableHeight = height - headerH;
        const numRows = Math.ceil(this.neuralNetworks.length / rowSize);
        let ySpace = Math.max(NeuralNetworkList.MIN_CELL_HEIGHT, availableHeight / numRows);

        ctx.save();

        // Clip to content area below header
        ctx.beginPath();
        ctx.rect(left, top + headerH, width, height - headerH);
        ctx.clip();

        //draw networks (scrolled)
        let col = 0;
        let row = 0;
        this.neuralNetworks.forEach((nn: NeuralNetwork) => {
            const y = top + ySpace * row + headerH - scrollY;
            // Skip if entirely outside visible area
            if (y + ySpace >= top + headerH && y < top + height) {
                nn.draw(ctx, left + xSpace * col, y, xSpace, ySpace, displayErrorDigits, displayMeanError);
            }
            col++;
            if(col >= rowSize){
                col = 0;
                row++;
            }
        });

        // Draw scroll indicator if content overflows
        const totalContentHeight = numRows * ySpace;
        if (totalContentHeight > availableHeight) {
            const scrollBarHeight = Math.max(20, availableHeight * (availableHeight / totalContentHeight));
            const scrollBarY = top + headerH + (scrollY / (totalContentHeight - availableHeight)) * (availableHeight - scrollBarHeight);
            ctx.fillStyle = "rgba(0, 0, 0, 0.3)";
            ctx.fillRect(left + width - 6, scrollBarY, 4, scrollBarHeight);
        }

        ctx.restore();
    }
    public drawBest(ctx: CanvasRenderingContext2D, left: number, top: number, width: number, height: number, rowSize: number, displayHeaderHeight?:number, displayHeader?:boolean, displayErrorDigits?: number, displayMeanError?: boolean){
        this.sort();
        //draw header
        if (displayHeader){
            this.drawHeader(ctx, left, top, width, displayHeaderHeight);
        }

        this.neuralNetworks[0].draw(ctx, left, top + (displayHeader? 20: 0), width, height - (displayHeader? 20: 0), displayErrorDigits, displayMeanError);
    }
    public computeTestErr(predict: (inputs: number[]) => number[]) {
        if (!this.testInputs.length || !this.testFn) { this.testRMSE = 0; this.testMAE = 0; return; }
        let sqSum = 0;
        let absSum = 0;
        for (let i = 0; i < this.testInputs.length; i++) {
            const pred = predict(this.testInputs[i]);
            const expected = this.testFn(this.testInputs[i]);
            for (let j = 0; j < pred.length; j++) {
                const diff = Math.abs(pred[j] - expected[j]);
                sqSum += diff ** 2;
                absSum += diff;
            }
        }
        this.testRMSE = Math.sqrt(sqSum / this.testInputs.length);
        this.testMAE = absSum / this.testInputs.length;
    }

    public drawHeader(ctx, left, top , width, height){
        ctx.save();
            ctx.fillStyle = "#e0e0e0"
            ctx.fillRect(left, top, width, height);
            ctx.font = '12px sans-serif';
            ctx.textBaseline = 'middle';
            ctx.fillStyle = "#000000"
            const text = `Gen: ${this.generation} | Train ϵ RMSE: ${this.trainRMSE.toFixed(4)} μ MAE: ${this.trainMAE.toFixed(4)} | Test ϵ RMSE: ${this.testRMSE.toFixed(4)} μ MAE: ${this.testMAE.toFixed(4)}`;
            ctx.fillText(text, left + 5, top + height / 2);

            ctx.restore();

    }

    // Draw XGBoost ensemble as a grid of small networks
    public drawXGBoostEnsemble(ctx: CanvasRenderingContext2D, left: number, top: number, width: number, height: number, displayHeaderHeight: number){
        ctx.save();

        // Draw header
        ctx.fillStyle = "#e0e0e0";
        ctx.fillRect(left, top, width, displayHeaderHeight);
        ctx.fillStyle = "#000000";
        ctx.font = '12px sans-serif';
        ctx.textBaseline = 'middle';
        ctx.fillText(`Gen: ${this.generation} | Ensemble: ${this.neuralNetworks.length} networks`, left + 5, top + displayHeaderHeight / 2);

        // Calculate grid layout
        const contentHeight = height - displayHeaderHeight;
        const cols = Math.min(4, this.neuralNetworks.length);
        const rows = Math.ceil(this.neuralNetworks.length / cols);
        const cellWidth = width / cols;
        const cellHeight = contentHeight / rows;

        // Draw each network in the ensemble
        for (let i = 0; i < this.neuralNetworks.length; i++) {
            const col = i % cols;
            const row = Math.floor(i / cols);
            const x = left + col * cellWidth;
            const y = top + displayHeaderHeight + row * cellHeight;

            // Draw network background
            ctx.fillStyle = i === 0 ? "#f0f0ff" : "#f8f8f8";
            ctx.fillRect(x + 2, y + 2, cellWidth - 4, cellHeight - 4);

            // Draw network index
            ctx.fillStyle = "#000000";
            ctx.font = '10px sans-serif';
            ctx.fillText(`#${i}${i === 0 ? ' (base)' : ''}`, x + 5, y + 12);

            // Draw mini network visualization
            this.neuralNetworks[i].draw(ctx, x + 5, y + 20, cellWidth - 10, cellHeight - 25);
        }

        ctx.restore();
    }
    public testError(outputs: number[], power?:number): number{

        if(outputs.length !== this.outputSize){
            console.error("Wrong number of outputs, expected: " + this.outputSize + " | received: " + outputs.length)
            return 0
        }
        this.neuralNetworks.forEach((nn: NeuralNetwork) => {
            nn.testError(outputs, false, power);
        });

    }
    public display2Input1OutputDataPoints(test: (inputs: number[]) => number[], ctx, left, top, width, height, axis1low: number, axis2low: number, axis1high: number, axis2high: number, outputMiddle:number, outputRange: number){

        ctx.save();
        ctx.rect(left, top, width, height);
        ctx.clip();

        this.trialInputsList.forEach((input) => {
            const x = left + ((input[0] - axis1low) / (axis1high - axis1low)) * width;
            const y = top + ((input[1] - axis2low) / (axis2high - axis2low)) * height;

            ctx.fillStyle = "black";
            drawCircle(ctx, {x, y}, Math.min(width, height) / 100);
            ctx.fillStyle = this.numToColorWhite((test(input)[0] - outputMiddle) / outputRange);
            drawCircle(ctx, {x, y}, Math.min(width, height) / 120);
        })

        ctx.restore();

    }

    public display2Input2OutputDataPoints(test: (inputs: number[]) => number[], ctx, left, top, width, height, axis1low: number, axis2low: number, axis1high: number, axis2high: number, color1: {r: number, g: number, b: number}, color2: {r: number, g: number, b: number}){

        ctx.save();
        ctx.rect(left, top, width, height);
        ctx.clip();

        this.trialInputsList.forEach((input) => {
            const x = left + ((input[0] - axis1low) / (axis1high - axis1low)) * width;
            const y = top + ((input[1] - axis2low) / (axis2high - axis2low)) * height;

            const outputs = test(input);
            ctx.fillStyle = "black";
            drawCircle(ctx, {x, y}, Math.min(width, height) / 100);
            ctx.fillStyle = this.neuralNetworks[0].interpolateColorPublic(color1, color2, outputs[0]);
            drawCircle(ctx, {x, y}, Math.min(width, height) / 120);
        })

        ctx.restore();

    }
    public testErrorTrials(inputsList: number[][], outputsList: number[][], power?: number){
        if (inputsList.length !== outputsList.length){
            console.error("Wrong number of inputs or outputs, expected: " + inputsList.length + " | received: " + outputsList.length)
            return 0
        }
        const p = power || 2;
        this.neuralNetworks.forEach((nn: NeuralNetwork) => {
            for (let i = 0; i < inputsList.length; i++){
                nn.run(inputsList[i]);
                nn.testError(outputsList[i], true, p);
            }
            nn.error = (nn.error / inputsList.length) ** (1 / p);
            nn.meanError /= inputsList.length;
        });
        
    }
    public resetError(){
        this.neuralNetworks.forEach(nn => {
            nn.error = 0;
            nn.meanError = 0;
        });
    }
    public createTrials(inputsList: number[][], outputFunction: (inputs: number[]) => number[]){
        this.trialInputsList = inputsList;
        this.trialOutputsList = [];

        this.trialInputsList.forEach(i => {
            this.trialOutputsList.push(outputFunction(i));
        });
    }
    public killLowestError(numOfNeuralNetworks: number){
        this.sort();
        this.neuralNetworks = this.neuralNetworks.slice(0, numOfNeuralNetworks);
    }
    public reproduceSurvivors(numOfWeights:number, weightStrength:number, numOfBiases:number, biasesStrength:number){
        let numOfSurvivors = this.neuralNetworks.length;

        // Keep all survivors unchanged - they earned their survival!
        // Clone them and mutate only the clones to fill the population
        let index = 0;
        while (this.neuralNetworks.length < this.numOfNeuralNetworks - 1){
            let nn = this.neuralNetworks[index].clone()
            nn.mutate(numOfWeights, weightStrength, numOfBiases, biasesStrength);
            this.neuralNetworks.push(nn);
            index++;
            if(index >= numOfSurvivors){index = 0;}
        }
        this.neuralNetworks.push(new NeuralNetwork(this.neuralNetworks[0].inputSize, this.neuralNetworks[0].hiddenLayerSizes, this.neuralNetworks[0].outputSize, this.neuralNetworks[0].activationFunction, this.neuralNetworks[0].outputActivationFunction));
    }
    public runGeneration(numOfWeights:number, weightStrength:number, numOfBiases:number, biasesStrength:number): number {
        this.generation++;
        this.resetError(); // Reset at START so previous error is visible for drawing
        this.testErrorTrials(this.trialInputsList, this.trialOutputsList, this.trialPower);
        this.sort();
        const bestError = this.neuralNetworks[0].error;
        this.trainRMSE = bestError;
        this.trainMAE = this.neuralNetworks[0].meanError;
        this.killLowestError(Math.floor(this.numOfNeuralNetworks / 2));
        this.reproduceSurvivors(numOfWeights, weightStrength, numOfBiases, biasesStrength);
        return bestError;
    }

    public setLearningRate(learningRate: number) {
        this.neuralNetworks.forEach(nn => {
            nn.learningRate = learningRate;
        });
    }

    public setMomentum(momentum: number) {
        this.neuralNetworks.forEach(nn => {
            nn.momentum = momentum;
        });
    }

    private _backpropWorker: NeuralNetwork | null = null;
    private _backpropBestError: number = Infinity;

    public trainBackpropagation(epochs: number = 1): number {
        this.generation++;

        // Lazy-init: the worker is the actual training network, index 0 holds the best for display
        if (!this._backpropWorker) {
            this._backpropWorker = this.neuralNetworks[0].clone();
            this._backpropBestError = Infinity;
        }

        // Train the worker
        for (let epoch = 0; epoch < epochs; epoch++) {
            this._backpropWorker.trainBatch(this.trialInputsList, this.trialOutputsList);
        }

        // Recalculate worker error
        this._backpropWorker.error = 0;
        this._backpropWorker.meanError = 0;
        for (let i = 0; i < this.trialInputsList.length; i++) {
            this._backpropWorker.run(this.trialInputsList[i]);
            this._backpropWorker.testError(this.trialOutputsList[i], true, this.trialPower);
        }
        this._backpropWorker.error = (this._backpropWorker.error / this.trialInputsList.length) ** (1 / this.trialPower);
        this._backpropWorker.meanError /= this.trialInputsList.length;

        // If worker improved, update the display network (index 0)
        if (this._backpropWorker.error < this._backpropBestError) {
            this._backpropBestError = this._backpropWorker.error;
            this.neuralNetworks[0] = this._backpropWorker.clone();
        }

        this.trainRMSE = this._backpropBestError;
        this.trainMAE = this.neuralNetworks[0].meanError;
        return this._backpropBestError;
    }

    public resetBackpropWorker() {
        this._backpropWorker = null;
        this._backpropBestError = Infinity;
    }

    // XGBoost-style training: Add one new network to ensemble, trained on residuals
    public trainXGBoost(epochs: number = 10, shrinkage: number = 0.1): number {
        this.generation++;

        // Calculate residuals (errors of current ensemble)
        const residuals: number[][] = [];

        for (let i = 0; i < this.trialInputsList.length; i++) {
            const ensemblePred = this.runEnsemble(this.trialInputsList[i]);
            const target = this.trialOutputsList[i];

            // Residual = what the ensemble got wrong
            const residual = target.map((t, idx) => t - ensemblePred[idx]);
            residuals.push(residual);
        }

        // Create a new network (clone the architecture of the first one)
        const newNetwork = this.neuralNetworks[0].clone();
        newNetwork.randomize(); // Randomize weights for the new network

        // Train the new network to predict the residuals
        for (let epoch = 0; epoch < epochs; epoch++) {
            newNetwork.trainBatch(this.trialInputsList, residuals);
        }

        // Apply shrinkage to the new network (prevents overfitting)
        newNetwork.shrinkWeights(shrinkage);

        // Add the new network to the ensemble
        this.neuralNetworks.push(newNetwork);
        this.numOfNeuralNetworks = this.neuralNetworks.length;

        // Calculate final ensemble error
        this.resetError();
        for (let i = 0; i < this.trialInputsList.length; i++) {
            const ensemblePred = this.runEnsemble(this.trialInputsList[i]);
            const target = this.trialOutputsList[i];

            let error = 0;
            for (let j = 0; j < this.outputSize; j++) {
                error += Math.abs(ensemblePred[j] - target[j]) ** this.trialPower;
            }
            this.neuralNetworks[0].error += error;
        }

        this.neuralNetworks[0].error = (this.neuralNetworks[0].error / this.trialInputsList.length) ** (1 / this.trialPower);

        return this.neuralNetworks[0].error;
    }

    // Run ensemble prediction (sum of all network outputs)
    private runEnsemble(inputs: number[]): number[] {
        const output = new Array(this.outputSize).fill(0);

        for (const nn of this.neuralNetworks) {
            const result = nn.run(inputs);
            for (let i = 0; i < this.outputSize; i++) {
                output[i] += result.neurons[i].value;
            }
        }

        return output;
    }

    // Get a virtual network representing the ensemble for display purposes
    public getEnsembleNetwork(): NeuralNetwork {
        // Create a copy of the first network structure
        const ensemble = this.neuralNetworks[0].clone();

        // Override the run method to use ensemble prediction
        const originalRun = ensemble.run.bind(ensemble);
        ensemble.run = (inputs: number[]) => {
            // Get ensemble prediction
            const ensembleOutput = this.runEnsemble(inputs);

            // Run the network normally first to set up the structure
            originalRun(inputs);

            // Override output values with ensemble prediction
            for (let i = 0; i < ensembleOutput.length; i++) {
                ensemble.getOutputLayer().neurons[i].value = ensembleOutput[i];
            }

            return ensemble.getOutputLayer();
        };

        return ensemble;
    }

    public sort(){
        this.neuralNetworks.sort((a, b) => a.error - b.error);
    }
    
    private numToColorBlack(num: number): string {
        if (num >= 0) {
            return "rgb(0, " + (Math.tanh(num)) * 255 + ",0)"
        }
        if (num < 0) {
            return "rgb(" + (-Math.tanh(num)) * 255 + ", 0 ,0)"

        }
        return ""
    }
    private numToColorWhite(num: number): string {
        if (num >= 0) {
            return "rgb(" + (1 - Math.tanh(num)) * 255 + ", 255," + (1 - Math.tanh(num)) * 255 + ")"
        }
        if (num < 0) {
            return "rgb(255, " + (1 + Math.tanh(num)) * 255 + "," + (1 + Math.tanh(num)) * 255 + ")"
        }
        return ""
    }
}
///////////////////////////////////Neural Network////////////////////////////////////////////////////////
export class NeuralNetwork {
    error:number = 0;
    meanError:number = 0;

    numOfLayers: number;
    inputSize: number;
    hiddenLayerSizes: number[];
    outputSize: number;
    startRandomBias: boolean = true;
    startRandomWeights: boolean = true;
    clampBiases: boolean = false;
    clampWeights: boolean = false;

    changeWeightAlpha: boolean = false;
    linePower: number = 4;

    layers: Layer[] = [];
    activationFunction: ActivationFunction = 'tanh';
    outputActivationFunction: ActivationFunction = 'tanh';

    // For backpropagation
    learningRate: number = 0.01; // Default learning rate (adjustable via UI)
    momentum: number = 0.9; // Default momentum (adjustable via UI)
    constructor(inputSize: number, hiddenLayerSizes: number[], outputSize: number, activationFunction: ActivationFunction, outputActivationFunction:ActivationFunction) {
        this.numOfLayers = hiddenLayerSizes.length + 2;
        this.inputSize = inputSize
        this.hiddenLayerSizes = hiddenLayerSizes;
        this.outputSize = outputSize;
        this.initLayers();
        this.activationFunction = activationFunction;
        this.outputActivationFunction = outputActivationFunction;
    }
    public run(inputs: number[]) {
        if (inputs.length != this.inputSize) {
            throw new Error('Wrong Number of INPUTs, Expected: ' + this.inputSize + "| Received: " + inputs.length)
        }
        for (let i = 0; i < this.inputSize; i++){
            this.layers[0].neurons[i].value = inputs[i];
        }
        for (let l = 1; l < this.numOfLayers; l++){
            let neurons = this.layers[l].neurons;
            for (let n = 0; n < neurons.length; n++){
                let neuron = neurons[n];
                neuron.value = 0;
                let weights = neuron.weights;
                for (let w = 0; w < weights.length; w++){
                    let weight: Weight = weights[w];
                    neuron.value += weight.value * this.getNodeValue(weight.to);
                }
                neuron.value += neuron.bias;
                neuron.activate();
                
            }
        }
        return this.layers[this.layers.length-1];
    }
    public mutate(numOfWeights:number, weightStrength:number, numOfBiases:number, biasesStrength:number){
        for (let i = 0; i < numOfWeights; i++){
            let randLayerIndex = Math.floor(Math.random() * (this.numOfLayers - 1)) + 1;
            let neurons = this.layers[randLayerIndex].neurons
            let randNeuronIndex = Math.floor(Math.random() * neurons.length)
            let weights = this.layers[randLayerIndex].neurons[randNeuronIndex].weights
            let randWeightsIndex = Math.floor(Math.random() * weights.length);
            weights[randWeightsIndex].value += (Math.random() * 2 - 1) * weightStrength;
            if (this.clampWeights){
                weights[randWeightsIndex].value = Math.min(Math.max(weights[randWeightsIndex].value, -1), 1)
            }
        }
        for (let i = 0; i < numOfBiases; i++){
            let randLayerIndex = Math.floor(Math.random() * (this.numOfLayers - 1)) + 1;
            let neurons = this.layers[randLayerIndex].neurons
            let randNeuronIndex = Math.floor(Math.random() * neurons.length)
            neurons[randNeuronIndex].bias += (Math.random() * 2 - 1) * biasesStrength;
            if (this.clampBiases){
                neurons[randNeuronIndex].bias = Math.min(Math.max(neurons[randNeuronIndex].bias, -1), 1)
            }
        }
    }

    // Randomize all weights and biases (for creating new networks in XGBoost)
    public randomize(){
        for (let l = 1; l < this.numOfLayers; l++) {
            for (let n = 0; n < this.layers[l].neurons.length; n++) {
                const neuron = this.layers[l].neurons[n];
                neuron.bias = Math.random() * 2 - 1;

                for (let w = 0; w < neuron.weights.length; w++) {
                    neuron.weights[w].value = Math.random() * 2 - 1;
                }
            }
        }
    }
    public testError(outputs: number[], additive: boolean, power?:number){
        let sum = 0;
        let meanSum = 0;
        const p = power || 2;
        for (let i = 0; i < this.outputSize; i++){
            const diff = Math.abs(this.getOutputLayer().neurons[i].value - outputs[i]);
            sum += diff ** p;
            meanSum += diff;
        }
        if (additive) {
            this.error += sum;
            this.meanError += meanSum;
        } else {
            this.error = sum;
            this.meanError = meanSum;
        }
    }
    // Backpropagation: calculate gradients for all weights and biases
    public backPropogate(targetOutputs: number[], resetGradients: boolean = true){
        if (targetOutputs.length !== this.outputSize) {
            throw new Error('Wrong number of target outputs');
        }

        // Reset all gradients to 0 (only if specified)
        if (resetGradients) {
            for (let l = 1; l < this.numOfLayers; l++) {
                for (let n = 0; n < this.layers[l].neurons.length; n++) {
                    this.layers[l].neurons[n].gradient = 0;
                    for (let w = 0; w < this.layers[l].neurons[n].weights.length; w++) {
                        this.layers[l].neurons[n].weights[w].gradient = 0;
                    }
                }
            }
        }

        // Calculate output layer gradients
        const outputLayer = this.layers[this.numOfLayers - 1];
        for (let i = 0; i < outputLayer.neurons.length; i++) {
            const neuron = outputLayer.neurons[i];
            const error = neuron.value - targetOutputs[i];
            neuron.gradient = 2 * error * neuron.activationDerivative();
        }

        // Backpropagate through hidden layers (down to 0 to accumulate weight gradients on layer 1)
        for (let l = this.numOfLayers - 2; l >= 0; l--) {
            const currentLayer = this.layers[l];
            const nextLayer = this.layers[l + 1];

            for (let i = 0; i < currentLayer.neurons.length; i++) {
                const neuron = currentLayer.neurons[i];
                let sum = 0;

                // Sum up gradients from all neurons in next layer that this neuron connects to
                for (let j = 0; j < nextLayer.neurons.length; j++) {
                    const nextNeuron = nextLayer.neurons[j];
                    // Find the weight connecting this neuron to nextNeuron
                    for (let w = 0; w < nextNeuron.weights.length; w++) {
                        const weight = nextNeuron.weights[w];
                        if (weight.to.layer === l && weight.to.index === i) {
                            sum += nextNeuron.gradient * weight.value;
                            // Also accumulate weight gradient
                            weight.gradient += nextNeuron.gradient * neuron.value;
                        }
                    }
                }

                // Apply activation derivative
                neuron.gradient = sum * neuron.activationDerivative();
            }
        }
    }

    // Apply gradients to update weights and biases with momentum
    public applyGradients() {
        for (let l = 1; l < this.numOfLayers; l++) {
            for (let n = 0; n < this.layers[l].neurons.length; n++) {
                const neuron = this.layers[l].neurons[n];

                // Update bias with momentum
                neuron.biasVelocity = this.momentum * neuron.biasVelocity - this.learningRate * neuron.gradient;
                neuron.bias += neuron.biasVelocity;

                if (this.clampBiases) {
                    neuron.bias = Math.max(-1, Math.min(1, neuron.bias));
                }

                // Update weights with momentum
                for (let w = 0; w < neuron.weights.length; w++) {
                    const weight = neuron.weights[w];
                    weight.velocity = this.momentum * weight.velocity - this.learningRate * weight.gradient;
                    weight.value += weight.velocity;

                    if (this.clampWeights) {
                        weight.value = Math.max(-1, Math.min(1, weight.value));
                    }
                }
            }
        }
    }

    // Shrink all weights and biases by a factor (for XGBoost shrinkage/regularization)
    public shrinkWeights(shrinkage: number) {
        for (let l = 1; l < this.numOfLayers; l++) {
            for (let n = 0; n < this.layers[l].neurons.length; n++) {
                const neuron = this.layers[l].neurons[n];
                neuron.bias *= shrinkage;

                for (let w = 0; w < neuron.weights.length; w++) {
                    neuron.weights[w].value *= shrinkage;
                }
            }
        }
    }

    // Train on a batch of data using backpropagation
    public trainBatch(inputs: number[][], targetOutputs: number[][]) {
        if (inputs.length !== targetOutputs.length) {
            throw new Error('Inputs and target outputs must have same length');
        }

        for (let i = 0; i < inputs.length; i++) {
            this.run(inputs[i]);
            this.backPropogate(targetOutputs[i], i === 0);
        }

        const batchSize = inputs.length;
        for (let l = 1; l < this.numOfLayers; l++) {
            for (let n = 0; n < this.layers[l].neurons.length; n++) {
                const neuron = this.layers[l].neurons[n];
                neuron.gradient /= batchSize;
                for (let w = 0; w < neuron.weights.length; w++) {
                    neuron.weights[w].gradient /= batchSize;
                }
            }
        }

        // Apply accumulated and normalized gradients
        this.applyGradients();
    }
    public getNodeValue(np: NeuronPosition): number {
        return this.layers[np.layer].neurons[np.index].value;
    }
    public getInputLayer(): Layer {
        return this.layers[0]
    }
    public getOutputLayer(): Layer {
        return this.layers[this.numOfLayers - 1]
    }
    public clone(): NeuralNetwork {
        let nn = new NeuralNetwork(this.inputSize, this.hiddenLayerSizes, this.outputSize, this.activationFunction, this.outputActivationFunction);

        nn.startRandomBias = this.startRandomBias;
        nn.startRandomWeights = this.startRandomWeights;
        nn.clampBiases = this.clampBiases;
        nn.clampWeights = this.clampWeights;
        nn.changeWeightAlpha = this.changeWeightAlpha;
        nn.linePower = this.linePower;
        nn.error = this.error;
        nn.meanError = this.meanError;

        for (let l = 0; l < this.layers.length; l++) {
            const sourceLayer = this.layers[l];
            const targetLayer = nn.layers[l];

            for (let n = 0; n < sourceLayer.neurons.length; n++) {
                const sourceNeuron = sourceLayer.neurons[n];
                const targetNeuron = targetLayer.neurons[n];

                targetNeuron.value = sourceNeuron.value;
                targetNeuron.bias = sourceNeuron.bias;
                targetNeuron.biasVelocity = sourceNeuron.biasVelocity;

                for (let w = 0; w < sourceNeuron.weights.length; w++) {
                    targetNeuron.weights[w].value = sourceNeuron.weights[w].value;
                    targetNeuron.weights[w].velocity = sourceNeuron.weights[w].velocity;
                }
            }
        }

        return nn;
    }
    private initLayers(): void {
        this.initNextLayer(this.inputSize, false, this.activationFunction);
        for (let i = 0; i < this.hiddenLayerSizes.length; i++) {
            this.initNextLayer(this.hiddenLayerSizes[i], this.startRandomBias, this.activationFunction)
        }
        this.initNextLayer(this.outputSize, this.startRandomBias, this.outputActivationFunction);
    }
    private initNextLayer(numOfNeurons: number, randomBias: boolean, activationFunction:ActivationFunction) {
        const layerNeurons: Neuron[] = [];
        for (let j = 0; j < numOfNeurons; j++) {
            const neuronWeights: Weight[] = this.initWeights();

            layerNeurons.push(new Neuron(neuronWeights, (randomBias ? Math.random() * 2 - 1 : 0), activationFunction));
        }
        this.layers.push(new Layer(layerNeurons));
    }
    private initWeights(): Weight[] {
        if (this.layers.length == 0) { return []; }
        let w: Weight[] = [];
        let lastLayerIndex = this.layers.length - 1;
        for (let i = 0; i < this.layers[lastLayerIndex].neurons.length; i++) {
            w.push(new Weight((this.startRandomWeights ? Math.random() * 2 - 1 : 0), new NeuronPosition(lastLayerIndex, i)));
        }
        return w;
    }
    public write(): void {
        console.log(this.layers);
        for (let i = 0; i < this.layers.length; i++) {
            let str = ""
            for (let j = 0; j < this.layers[i].neurons.length; j++) {
                let neuron = this.layers[i].neurons[j];
                str += (neuron.bias)
                str += " "
            }
        }
    }
    public draw(ctx: CanvasRenderingContext2D, left: number, top: number, width: number, height: number, displayErrorDigits?: number, displayMeanError?: boolean) {
        let spaceX = width / (this.numOfLayers + 1);
        let lastYs: number[] = [];

        const bottomPadding = (displayErrorDigits ? 15 : 0) + (displayMeanError ? 15 : 0);

        for (let i = 0; i < this.numOfLayers; i++) {
            let x: number = (i + 1) * spaceX + left;

            let lastX: number = i * spaceX + left;

            let neurons: Neuron[] = this.layers[i].neurons;
            let spaceY: number = (height - bottomPadding) / (neurons.length + 1);
            let newYs: number[] = []

            for (let j = 0; j < neurons.length; j++) {
                let neuron: Neuron = neurons[j];
                let y: number = (j + 1) * spaceY + top;
                newYs.push(y);

                for (let w = 0; w < lastYs.length; w++) {
                    ctx.save()
                    ctx.strokeStyle = this.numToColorBlack(neuron.weights[w].value);
                    ctx.lineWidth = 2;
                    if (this.changeWeightAlpha) ctx.globalAlpha = Math.abs(neuron.weights[w].value)**this.linePower;
                    ctx.beginPath();
                    ctx.moveTo(x, y);
                    ctx.lineTo(lastX, lastYs[w]);
                    ctx.stroke();

                    ctx.restore();
                }

            }
            lastYs = newYs;


        }
        for (let i = 0; i < this.numOfLayers; i++) {
            let neurons: Neuron[] = this.layers[i].neurons;
            let spaceY: number = (height - bottomPadding) / (neurons.length + 1);
            let x: number = (i + 1) * spaceX + left;

            for (let j = 0; j < neurons.length; j++) {
                let neuron: Neuron = neurons[j];
                let y: number = (j + 1) * spaceY + top;

                let r1 = Math.min(spaceX, spaceY) / 2.1
                ctx.save();
                ctx.fillStyle = this.numToColorBlack(neuron.bias);
                drawCircle(ctx, { x, y }, r1);
                ctx.restore();

                let r2 = Math.min(spaceX, spaceY) / 2.3
                ctx.save();
                ctx.fillStyle = this.numToColorWhite(neuron.value);
                drawCircle(ctx, { x, y }, r2);
                ctx.restore();

 

                ctx.save();
                let val = Math.floor(neuron.value * 1000) / 1000;
                let textStr = val.toString();
                let drawn: boolean = false;
                ctx.font = '12px sans-serif';
                ctx.textBaseline = 'middle';
                ctx.fillStyle = '#000000';
                while (textStr.length > 0 && textStr != "-" && !drawn){
                    if (textStr.at(-1) == "."){
                        textStr = textStr.slice(0, -1);
                        continue;
                    }
                    let measure = ctx.measureText(textStr);
                    let w = measure.width;

                    if(w >= r2 * 1.8) {
                        textStr = textStr.slice(0, -1);
                        continue;
                    }

                    drawn = true;
                    ctx.fillText(textStr, x - w / 2, y);
                    ctx.restore();

                }

            }
        }
        if (displayErrorDigits){
            ctx.font = '12px sans-serif';
            ctx.fillStyle = '#000000';
            let textStr = "ϵ RMSE: " + (Math.floor(this.error * 10**displayErrorDigits) / 10**displayErrorDigits).toString()
            let drawn = false;
            while (textStr.length > 0 && textStr != "-" && !drawn){
                let measure = ctx.measureText(textStr);
                let w = measure.width;
                if(w >= width) {
                    textStr = textStr.slice(0, -1);
                    continue;
                }
                drawn = true;
                const yOffset = displayMeanError ? 20 : 10;
                ctx.fillText(textStr, left + width / 2 - w / 2,  top + height - yOffset);
                ctx.restore();
            }
        }
        if (displayMeanError){
            ctx.font = '12px sans-serif';
            ctx.fillStyle = '#000000';
            let textStr = "μ MAE: " + (Math.floor(this.meanError * 10**displayErrorDigits) / 10**displayErrorDigits).toString()
            let drawn = false;
            while (textStr.length > 0 && textStr != "-" && !drawn){
                let measure = ctx.measureText(textStr);
                let w = measure.width;
                if(w >= width) {
                    textStr = textStr.slice(0, -1);
                    continue;
                }
                drawn = true;
                ctx.fillText(textStr, left + width / 2 - w / 2,  top + height - 10);
                ctx.restore();
            }
        }
    }
    
    public displayGrid2Input1Output(ctx, left: number, top: number, width: number, height: number, axis1low: number, axis2low: number, axis1high: number, axis2high: number, rows: number, columns: number, decimals: number, showText: boolean, showHeaders: boolean, getCellValue: (input1: number, input2: number) => number, getColorValue: (cellValue: number) => number){
        const headerOffset = showHeaders ? 1 : 0;
        let spaceX = width / (columns + headerOffset);
        let spaceY = height / (rows + headerOffset);

        ctx.save();
        ctx.font = `${Math.min(spaceX, spaceY) * 0.3}px sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = '#e0e0e0';
        ctx.fillRect(left, top, width, height);

        if (showHeaders) {
            for (let i = 0; i < columns; i++){
                let x = left + (i + 1) * spaceX;
                let y = top;
                let headerValue = axis1low + i * (axis1high - axis1low) / (columns - 1);

                ctx.fillStyle = '#000000';
                let text = headerValue.toFixed(2);
                if (ctx.measureText(text).width > spaceX * 0.9) {
                    text = headerValue.toFixed(1);
                }
                if (ctx.measureText(text).width > spaceX * 0.9) {
                    text = headerValue.toFixed(0);
                }
                ctx.fillText(text, x + spaceX / 2, y + spaceY / 2);
            }

            for (let j = 0; j < rows; j++){
                let x = left;
                let y = top + (j + 1) * spaceY;
                let headerValue = axis2low + j * (axis2high - axis2low) / (rows - 1);

                ctx.fillStyle = '#e0e0e0';
                ctx.fillRect(x, y, spaceX, spaceY);

                ctx.fillStyle = '#000000';
                let text = headerValue.toFixed(2);
                if (ctx.measureText(text).width > spaceX * 0.9) {
                    text = headerValue.toFixed(1);
                }
                if (ctx.measureText(text).width > spaceX * 0.9) {
                    text = headerValue.toFixed(0);
                }
                ctx.fillText(text, x + spaceX / 2, y + spaceY / 2);
            }

            ctx.fillStyle = '#c0c0c0';
            ctx.fillRect(left, top, spaceX, spaceY);
        }

        for (let i = 0; i < columns; i++){
            for (let j = 0; j < rows; j++){
                let x = left + (i + headerOffset) * spaceX;
                let y = top + (j + headerOffset) * spaceY;
                let input1 = axis1low + i * (axis1high - axis1low) / (columns - 1);
                let input2 = axis2low + j * (axis2high - axis2low) / (rows - 1);

                let cellValue = getCellValue(input1, input2);
                ctx.fillStyle = this.numToColorWhite(getColorValue(cellValue));
                ctx.fillRect(x, y, spaceX + (showText? -1: 1), spaceY + (showText? -1: 1));

                if(showText){
                    ctx.fillStyle = '#000000';
                    let text = cellValue.toFixed(decimals);
                    let textWidth = ctx.measureText(text).width;
                    if (textWidth > spaceX * 0.9) {
                        text = cellValue.toFixed(Math.max(0, decimals - 1));
                        textWidth = ctx.measureText(text).width;
                        if (textWidth > spaceX * 0.9) {
                            text = cellValue.toFixed(0);
                        }
                    }
                    ctx.fillText(text, x + spaceX / 2, y + spaceY / 2);
                }
            }
        }
        ctx.restore();
        
    }
    
    public display2Input1OutputError (ctx, left: number, top: number, width:number, height:number, axis1low: number, axis2low: number, axis1high: number, axis2high: number, errorRange: number, rows:number, columns: number, decimals: number, showText: boolean, showHeaders: boolean, test: (inputs: number[]) => number[]){
        this.displayGrid2Input1Output(ctx, left, top, width, height, axis1low, axis2low, axis1high, axis2high, rows, columns, decimals, showText, showHeaders,
            (input1, input2) => {
                let val = this.run([input1, input2]);
                return val.neurons[0].value - test([input1, input2])[0];
            },
            (cellValue) => cellValue * errorRange
        );
    }

    public display2Input1OutputTest(ctx, left: number, top: number, width:number, height:number, axis1low: number, axis2low: number, axis1high: number, axis2high: number, ouputMiddle:number, outputRange: number, rows:number, columns: number, decimals: number, showText: boolean, showHeaders: boolean, test: (inputs: number[]) => number[]){
        this.displayGrid2Input1Output(ctx, left, top, width, height, axis1low, axis2low, axis1high, axis2high, rows, columns, decimals, showText, showHeaders,
            (input1, input2) => test([input1, input2])[0],
            (cellValue) => cellValue * outputRange - ouputMiddle
        );
    }

    public display2Input1Output(ctx, left: number, top: number, width:number, height:number, axis1low: number, axis2low: number, axis1high: number, axis2high: number, ouputMiddle:number, outputRange: number, rows:number, columns: number, decimals: number, showText: boolean, showHeaders: boolean){
        this.displayGrid2Input1Output(ctx, left, top, width, height, axis1low, axis2low, axis1high, axis2high, rows, columns, decimals, showText, showHeaders,
            (input1, input2) => this.run([input1, input2]).neurons[0].value,
            (cellValue) => cellValue * outputRange - ouputMiddle
        );
    }

    private numToColorBlack(num: number): string {
        if (num >= 0) {
            return "rgb(0, " + (Math.tanh(num)) * 255 + ",0)"
        }
        if (num < 0) {
            return "rgb(" + (-Math.tanh(num)) * 255 + ", 0 ,0)"

        }
        return ""
    }
    private numToColorWhite(num: number): string {
        if (num >= 0) {
            return "rgb(" + (1 - Math.tanh(num)) * 255 + ", 255," + (1 - Math.tanh(num)) * 255 + ")"
        }
        if (num < 0) {
            return "rgb(255, " + (1 + Math.tanh(num)) * 255 + "," + (1 + Math.tanh(num)) * 255 + ")"
        }
        return ""
    }

    private interpolateColor(color1: {r: number, g: number, b: number}, color2: {r: number, g: number, b: number}, weight: number): string {
        // weight should be between 0 and 1
        // weight = 0 means 100% color2, weight = 1 means 100% color1
        const clampedWeight = Math.max(0, Math.min(1, weight));
        const r = Math.round(color1.r * clampedWeight + color2.r * (1 - clampedWeight));
        const g = Math.round(color1.g * clampedWeight + color2.g * (1 - clampedWeight));
        const b = Math.round(color1.b * clampedWeight + color2.b * (1 - clampedWeight));
        return `rgb(${r}, ${g}, ${b})`;
    }

    public interpolateColorPublic(color1: {r: number, g: number, b: number}, color2: {r: number, g: number, b: number}, weight: number): string {
        return this.interpolateColor(color1, color2, weight);
    }

    public displayGrid2Input2Output(ctx, left: number, top: number, width: number, height: number, axis1low: number, axis2low: number, axis1high: number, axis2high: number, rows: number, columns: number, decimals: number, showText: boolean, showHeaders: boolean, color1: {r: number, g: number, b: number}, color2: {r: number, g: number, b: number}, getOutputs: (input1: number, input2: number) => number[]){
        const headerOffset = showHeaders ? 1 : 0;
        let spaceX = width / (columns + headerOffset);
        let spaceY = height / (rows + headerOffset);

        ctx.save();
        ctx.font = `${Math.min(spaceX, spaceY) * 0.3}px sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = '#e0e0e0';
        ctx.fillRect(left, top, width, height);

        if (showHeaders) {
            for (let i = 0; i < columns; i++){
                let x = left + (i + 1) * spaceX;
                let y = top;
                let headerValue = axis1low + i * (axis1high - axis1low) / (columns - 1);

                ctx.fillStyle = '#000000';
                let text = headerValue.toFixed(2);
                if (ctx.measureText(text).width > spaceX * 0.9) {
                    text = headerValue.toFixed(1);
                }
                if (ctx.measureText(text).width > spaceX * 0.9) {
                    text = headerValue.toFixed(0);
                }
                ctx.fillText(text, x + spaceX / 2, y + spaceY / 2);
            }

            for (let j = 0; j < rows; j++){
                let x = left;
                let y = top + (j + 1) * spaceY;
                let headerValue = axis2low + j * (axis2high - axis2low) / (rows - 1);

                ctx.fillStyle = '#e0e0e0';
                ctx.fillRect(x, y, spaceX, spaceY);

                ctx.fillStyle = '#000000';
                let text = headerValue.toFixed(2);
                if (ctx.measureText(text).width > spaceX * 0.9) {
                    text = headerValue.toFixed(1);
                }
                if (ctx.measureText(text).width > spaceX * 0.9) {
                    text = headerValue.toFixed(0);
                }
                ctx.fillText(text, x + spaceX / 2, y + spaceY / 2);
            }

            ctx.fillStyle = '#c0c0c0';
            ctx.fillRect(left, top, spaceX, spaceY);
        }

        for (let i = 0; i < columns; i++){
            for (let j = 0; j < rows; j++){
                let x = left + (i + headerOffset) * spaceX;
                let y = top + (j + headerOffset) * spaceY;
                let input1 = axis1low + i * (axis1high - axis1low) / (columns - 1);
                let input2 = axis2low + j * (axis2high - axis2low) / (rows - 1);

                let outputs = getOutputs(input1, input2);
                // outputs[0] is probability of class 1, outputs[1] is probability of class 2
                // Use outputs[0] as the weight for interpolation
                ctx.fillStyle = this.interpolateColor(color1, color2, outputs[0]);
                ctx.fillRect(x, y, spaceX + (showText? -1: 1), spaceY + (showText? -1: 1));

                if(showText){
                    ctx.fillStyle = '#000000';
                    let text = outputs[0].toFixed(decimals) + ',' + outputs[1].toFixed(decimals);
                    let textWidth = ctx.measureText(text).width;
                    let shrink = 0
                    while (textWidth > spaceX * 0.9 && shrink < decimals) {
                        text = outputs[0].toFixed(decimals - shrink) + ',' + outputs[1].toFixed(decimals - shrink);
                        textWidth = ctx.measureText(text).width;
                        if (textWidth > spaceX * 0.9) {
                            text = outputs[0].toFixed(decimals - shrink);
                        }
                        shrink++;
                    }
                    ctx.fillText(text, x + spaceX / 2, y + spaceY / 2);
                }
            }
        }
        ctx.restore();

    }

    public display2Input2Output(ctx, left: number, top: number, width:number, height:number, axis1low: number, axis2low: number, axis1high: number, axis2high: number, rows:number, columns: number, decimals: number, showText: boolean, showHeaders: boolean, color1: {r: number, g: number, b: number}, color2: {r: number, g: number, b: number}){
        this.displayGrid2Input2Output(ctx, left, top, width, height, axis1low, axis2low, axis1high, axis2high, rows, columns, decimals, showText, showHeaders, color1, color2,
            (input1, input2) => {
                let result = this.run([input1, input2]);
                return [result.neurons[0].value, result.neurons[1].value];
            }
        );
    }

    public display2Input2OutputTest(ctx, left: number, top: number, width:number, height:number, axis1low: number, axis2low: number, axis1high: number, axis2high: number, rows:number, columns: number, decimals: number, showText: boolean, showHeaders: boolean, color1: {r: number, g: number, b: number}, color2: {r: number, g: number, b: number}, test: (inputs: number[]) => number[]){
        this.displayGrid2Input2Output(ctx, left, top, width, height, axis1low, axis2low, axis1high, axis2high, rows, columns, decimals, showText, showHeaders, color1, color2,
            (input1, input2) => test([input1, input2])
        );
    }

    public display2Input2OutputError(ctx, left: number, top: number, width:number, height:number, axis1low: number, axis2low: number, axis1high: number, axis2high: number, rows:number, columns: number, decimals: number, showText: boolean, showHeaders: boolean, color1: {r: number, g: number, b: number}, color2: {r: number, g: number, b: number}, test: (inputs: number[]) => number[]){
        // For error visualization, use white (no error) to red (high error) gradient
        const errorColor1 = {r: 255, g: 255, b: 255}; // White for no error
        const errorColor2 = {r: 255, g: 0, b: 0}; // Red for high error

        this.displayGrid2Input2Output(ctx, left, top, width, height, axis1low, axis2low, axis1high, axis2high, rows, columns, decimals, showText, showHeaders, errorColor1, errorColor2,
            (input1, input2) => {
                let nnOutput = this.run([input1, input2]);
                let testOutput = test([input1, input2]);
                let error0 = Math.abs(nnOutput.neurons[0].value - testOutput[0]);
                let error1 = Math.abs(nnOutput.neurons[1].value - testOutput[1]);
                // Calculate total error (sum of both output errors)
                let totalError = (error0 + error1) / 2; // Average error
                // Return total error as both outputs for color interpolation
                // This will create a gradient from white (0 error) to red (1 error)
                return [totalError, 0]; // Using totalError as weight
            }
        );
    }

}
export type ActivationFunction = 'sigmoid' | 'relu' | 'tanh';

class Layer {
    neurons: Neuron[];
    constructor(neurons: Neuron[]) {
        this.neurons = neurons;
    }
}
class Neuron {
    value: number;
    weights: Weight[];
    bias: number;
    activationFunction: ActivationFunction;
    // For backpropagation
    gradient: number = 0; // Gradient of the loss with respect to this neuron's output
    preActivation: number = 0; // Value before activation function
    biasVelocity: number = 0; // Momentum for bias

    constructor(weights: Weight[], bias: number, activationFunction: ActivationFunction) {
        this.value = 0;
        this.weights = weights;
        this.bias = bias;
        this.activationFunction = activationFunction;
    }
    public activate(){
        this.preActivation = this.value; // Store pre-activation value for backprop
        switch (this.activationFunction){
            case "relu":
                this.value = (this.value <= 0)? 0: this.value;
                break;
            case "sigmoid":
                this.value = 1 / (1 + Math.pow(Math.E, -this.value));
                break;
            case "tanh":
               this.value = Math.tanh(this.value);
               break;
        }
    }

    // Calculate derivative of activation function
    public activationDerivative(): number {
        switch (this.activationFunction){
            case "relu":
                return this.preActivation > 0 ? 1 : 0;
            case "sigmoid":
                return this.value * (1 - this.value);
            case "tanh":
                return 1 - this.value * this.value;
        }
        return 0;
    }
}
class Weight {
    value: number;
    to: NeuronPosition;
    gradient: number = 0; 
    velocity: number = 0; // Momentum for weight

    constructor(value: number, to: NeuronPosition) {
        this.value = value;
        this.to = to;
    }
}
class NeuronPosition {
    layer: number;
    index: number;
    constructor(layer: number, index: number) {
        this.layer = layer;
        this.index = index;
    }
}


