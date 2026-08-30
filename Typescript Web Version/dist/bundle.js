(() => {
  // graphics.js
  function drawCircle(ctx2, position, radius, opts = {}) {
    if (radius <= 0) {
      return;
    }
    const { fillStyle, strokeStyle, lineWidth, fill = true, stroke = false } = opts;
    ctx2.beginPath();
    ctx2.arc(position.x, position.y, radius, 0, Math.PI * 2);
    if (lineWidth !== void 0)
      ctx2.lineWidth = lineWidth;
    if (fillStyle !== void 0)
      ctx2.fillStyle = fillStyle;
    if (strokeStyle !== void 0)
      ctx2.strokeStyle = strokeStyle;
    if (fill)
      ctx2.fill();
    if (stroke)
      ctx2.stroke();
  }

  // neuralnetwork.js
  var NeuralNetworkList = class _NeuralNetworkList {
    numOfNeuralNetworks = 0;
    neuralNetworks = [];
    generation = 0;
    inputSize;
    outputSize;
    trainRMSE = 0;
    trainMAE = 0;
    testRMSE = 0;
    testMAE = 0;
    trialInputsList = [];
    trialOutputsList = [];
    trialPower = 2;
    trainingData = [];
    testInputs = [];
    testFn = null;
    constructor(numOfNeuralNetworks2, inputSize2, hiddenLayerSizes2, outputSize2, activationFunction2, outputActivationFunction2) {
      this.numOfNeuralNetworks = numOfNeuralNetworks2;
      for (let i = 0; i < numOfNeuralNetworks2; i++) {
        const nn = new NeuralNetwork(inputSize2, hiddenLayerSizes2, outputSize2, activationFunction2, outputActivationFunction2);
        this.neuralNetworks.push(nn);
      }
      this.inputSize = inputSize2;
      this.outputSize = outputSize2;
    }
    run(inputs) {
      this.neuralNetworks.forEach((nn) => nn.run(inputs));
    }
    mutate(numOfWeights, weightStrength, numOfBiases, biasesStrength) {
      this.neuralNetworks.forEach((nn) => nn.mutate(numOfWeights, weightStrength, numOfBiases, biasesStrength));
      return this;
    }
    static MIN_CELL_HEIGHT = 200;
    getContentHeight(rowSize, displayHeaderHeight, displayHeader, panelHeight) {
      const headerH = displayHeader ? displayHeaderHeight : 0;
      const availableHeight = panelHeight - headerH;
      const numRows = Math.ceil(this.neuralNetworks.length / rowSize);
      const cellHeight = Math.max(_NeuralNetworkList.MIN_CELL_HEIGHT, availableHeight / numRows);
      return headerH + numRows * cellHeight;
    }
    draw(ctx2, left, top, width, height, rowSize, displayHeaderHeight, displayHeader, displayErrorDigits, displayMeanError, scrollY = 0) {
      let xSpace = width / rowSize;
      const headerH = displayHeader ? displayHeaderHeight : 0;
      const availableHeight = height - headerH;
      const numRows = Math.ceil(this.neuralNetworks.length / rowSize);
      let ySpace = Math.max(_NeuralNetworkList.MIN_CELL_HEIGHT, availableHeight / numRows);
      ctx2.save();
      ctx2.beginPath();
      ctx2.rect(left, top + headerH, width, height - headerH);
      ctx2.clip();
      let col = 0;
      let row = 0;
      this.neuralNetworks.forEach((nn) => {
        const y = top + ySpace * row + headerH - scrollY;
        if (y + ySpace >= top + headerH && y < top + height) {
          nn.draw(ctx2, left + xSpace * col, y, xSpace, ySpace, displayErrorDigits, displayMeanError);
        }
        col++;
        if (col >= rowSize) {
          col = 0;
          row++;
        }
      });
      const totalContentHeight = numRows * ySpace;
      if (totalContentHeight > availableHeight) {
        const scrollBarHeight = Math.max(20, availableHeight * (availableHeight / totalContentHeight));
        const scrollBarY = top + headerH + scrollY / (totalContentHeight - availableHeight) * (availableHeight - scrollBarHeight);
        ctx2.fillStyle = "rgba(0, 0, 0, 0.3)";
        ctx2.fillRect(left + width - 6, scrollBarY, 4, scrollBarHeight);
      }
      ctx2.restore();
    }
    drawBest(ctx2, left, top, width, height, rowSize, displayHeaderHeight, displayHeader, displayErrorDigits, displayMeanError) {
      this.sort();
      if (displayHeader) {
        this.drawHeader(ctx2, left, top, width, displayHeaderHeight);
      }
      this.neuralNetworks[0].draw(ctx2, left, top + (displayHeader ? 20 : 0), width, height - (displayHeader ? 20 : 0), displayErrorDigits, displayMeanError);
    }
    computeTestErr(predict) {
      if (!this.testInputs.length || !this.testFn) {
        this.testRMSE = 0;
        this.testMAE = 0;
        return;
      }
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
    drawHeader(ctx2, left, top, width, height) {
      ctx2.save();
      ctx2.fillStyle = "#e0e0e0";
      ctx2.fillRect(left, top, width, height);
      ctx2.font = "12px sans-serif";
      ctx2.textBaseline = "middle";
      ctx2.fillStyle = "#000000";
      const text = `Gen: ${this.generation} | Train \u03F5 RMSE: ${this.trainRMSE.toFixed(4)} \u03BC MAE: ${this.trainMAE.toFixed(4)} | Test \u03F5 RMSE: ${this.testRMSE.toFixed(4)} \u03BC MAE: ${this.testMAE.toFixed(4)}`;
      ctx2.fillText(text, left + 5, top + height / 2);
      ctx2.restore();
    }
    // Draw XGBoost ensemble as a grid of small networks
    drawXGBoostEnsemble(ctx2, left, top, width, height, displayHeaderHeight) {
      ctx2.save();
      ctx2.fillStyle = "#e0e0e0";
      ctx2.fillRect(left, top, width, displayHeaderHeight);
      ctx2.fillStyle = "#000000";
      ctx2.font = "12px sans-serif";
      ctx2.textBaseline = "middle";
      ctx2.fillText(`Gen: ${this.generation} | Ensemble: ${this.neuralNetworks.length} networks`, left + 5, top + displayHeaderHeight / 2);
      const contentHeight = height - displayHeaderHeight;
      const cols = Math.min(4, this.neuralNetworks.length);
      const rows = Math.ceil(this.neuralNetworks.length / cols);
      const cellWidth = width / cols;
      const cellHeight = contentHeight / rows;
      for (let i = 0; i < this.neuralNetworks.length; i++) {
        const col = i % cols;
        const row = Math.floor(i / cols);
        const x = left + col * cellWidth;
        const y = top + displayHeaderHeight + row * cellHeight;
        ctx2.fillStyle = i === 0 ? "#f0f0ff" : "#f8f8f8";
        ctx2.fillRect(x + 2, y + 2, cellWidth - 4, cellHeight - 4);
        ctx2.fillStyle = "#000000";
        ctx2.font = "10px sans-serif";
        ctx2.fillText(`#${i}${i === 0 ? " (base)" : ""}`, x + 5, y + 12);
        this.neuralNetworks[i].draw(ctx2, x + 5, y + 20, cellWidth - 10, cellHeight - 25);
      }
      ctx2.restore();
    }
    testError(outputs, power) {
      if (outputs.length !== this.outputSize) {
        console.error("Wrong number of outputs, expected: " + this.outputSize + " | received: " + outputs.length);
        return 0;
      }
      this.neuralNetworks.forEach((nn) => {
        nn.testError(outputs, false, power);
      });
    }
    display2Input1OutputDataPoints(test2, ctx2, left, top, width, height, axis1low, axis2low, axis1high, axis2high, outputMiddle, outputRange) {
      ctx2.save();
      ctx2.rect(left, top, width, height);
      ctx2.clip();
      this.trialInputsList.forEach((input) => {
        const x = left + (input[0] - axis1low) / (axis1high - axis1low) * width;
        const y = top + (input[1] - axis2low) / (axis2high - axis2low) * height;
        ctx2.fillStyle = "black";
        drawCircle(ctx2, { x, y }, Math.min(width, height) / 100);
        ctx2.fillStyle = this.numToColorWhite((test2(input)[0] - outputMiddle) / outputRange);
        drawCircle(ctx2, { x, y }, Math.min(width, height) / 120);
      });
      ctx2.restore();
    }
    display2Input2OutputDataPoints(test2, ctx2, left, top, width, height, axis1low, axis2low, axis1high, axis2high, color12, color22) {
      ctx2.save();
      ctx2.rect(left, top, width, height);
      ctx2.clip();
      this.trialInputsList.forEach((input) => {
        const x = left + (input[0] - axis1low) / (axis1high - axis1low) * width;
        const y = top + (input[1] - axis2low) / (axis2high - axis2low) * height;
        const outputs = test2(input);
        ctx2.fillStyle = "black";
        drawCircle(ctx2, { x, y }, Math.min(width, height) / 100);
        ctx2.fillStyle = this.neuralNetworks[0].interpolateColorPublic(color12, color22, outputs[0]);
        drawCircle(ctx2, { x, y }, Math.min(width, height) / 120);
      });
      ctx2.restore();
    }
    testErrorTrials(inputsList, outputsList, power) {
      if (inputsList.length !== outputsList.length) {
        console.error("Wrong number of inputs or outputs, expected: " + inputsList.length + " | received: " + outputsList.length);
        return 0;
      }
      const p = power || 2;
      this.neuralNetworks.forEach((nn) => {
        for (let i = 0; i < inputsList.length; i++) {
          nn.run(inputsList[i]);
          nn.testError(outputsList[i], true, p);
        }
        nn.error = (nn.error / inputsList.length) ** (1 / p);
        nn.meanError /= inputsList.length;
      });
    }
    resetError() {
      this.neuralNetworks.forEach((nn) => {
        nn.error = 0;
        nn.meanError = 0;
      });
    }
    createTrials(inputsList, outputFunction) {
      this.trialInputsList = inputsList;
      this.trialOutputsList = [];
      this.trialInputsList.forEach((i) => {
        this.trialOutputsList.push(outputFunction(i));
      });
    }
    killLowestError(numOfNeuralNetworks2) {
      this.sort();
      this.neuralNetworks = this.neuralNetworks.slice(0, numOfNeuralNetworks2);
    }
    reproduceSurvivors(numOfWeights, weightStrength, numOfBiases, biasesStrength) {
      let numOfSurvivors = this.neuralNetworks.length;
      let index = 0;
      while (this.neuralNetworks.length < this.numOfNeuralNetworks - 1) {
        let nn = this.neuralNetworks[index].clone();
        nn.mutate(numOfWeights, weightStrength, numOfBiases, biasesStrength);
        this.neuralNetworks.push(nn);
        index++;
        if (index >= numOfSurvivors) {
          index = 0;
        }
      }
      this.neuralNetworks.push(new NeuralNetwork(this.neuralNetworks[0].inputSize, this.neuralNetworks[0].hiddenLayerSizes, this.neuralNetworks[0].outputSize, this.neuralNetworks[0].activationFunction, this.neuralNetworks[0].outputActivationFunction));
    }
    runGeneration(numOfWeights, weightStrength, numOfBiases, biasesStrength) {
      this.generation++;
      this.resetError();
      this.testErrorTrials(this.trialInputsList, this.trialOutputsList, this.trialPower);
      this.sort();
      const bestError = this.neuralNetworks[0].error;
      this.trainRMSE = bestError;
      this.trainMAE = this.neuralNetworks[0].meanError;
      this.killLowestError(Math.floor(this.numOfNeuralNetworks / 2));
      this.reproduceSurvivors(numOfWeights, weightStrength, numOfBiases, biasesStrength);
      return bestError;
    }
    setLearningRate(learningRate2) {
      this.neuralNetworks.forEach((nn) => {
        nn.learningRate = learningRate2;
      });
    }
    setMomentum(momentum2) {
      this.neuralNetworks.forEach((nn) => {
        nn.momentum = momentum2;
      });
    }
    _backpropWorker = null;
    _backpropBestError = Infinity;
    trainBackpropagation(epochs = 1) {
      this.generation++;
      if (!this._backpropWorker) {
        this._backpropWorker = this.neuralNetworks[0].clone();
        this._backpropBestError = Infinity;
      }
      for (let epoch = 0; epoch < epochs; epoch++) {
        this._backpropWorker.trainBatch(this.trialInputsList, this.trialOutputsList);
      }
      this._backpropWorker.error = 0;
      this._backpropWorker.meanError = 0;
      for (let i = 0; i < this.trialInputsList.length; i++) {
        this._backpropWorker.run(this.trialInputsList[i]);
        this._backpropWorker.testError(this.trialOutputsList[i], true, this.trialPower);
      }
      this._backpropWorker.error = (this._backpropWorker.error / this.trialInputsList.length) ** (1 / this.trialPower);
      this._backpropWorker.meanError /= this.trialInputsList.length;
      if (this._backpropWorker.error < this._backpropBestError) {
        this._backpropBestError = this._backpropWorker.error;
        this.neuralNetworks[0] = this._backpropWorker.clone();
      }
      this.trainRMSE = this._backpropBestError;
      this.trainMAE = this.neuralNetworks[0].meanError;
      return this._backpropBestError;
    }
    resetBackpropWorker() {
      this._backpropWorker = null;
      this._backpropBestError = Infinity;
    }
    // XGBoost-style training: Add one new network to ensemble, trained on residuals
    trainXGBoost(epochs = 10, shrinkage = 0.1) {
      this.generation++;
      const residuals = [];
      for (let i = 0; i < this.trialInputsList.length; i++) {
        const ensemblePred = this.runEnsemble(this.trialInputsList[i]);
        const target = this.trialOutputsList[i];
        const residual = target.map((t, idx) => t - ensemblePred[idx]);
        residuals.push(residual);
      }
      const newNetwork = this.neuralNetworks[0].clone();
      newNetwork.randomize();
      for (let epoch = 0; epoch < epochs; epoch++) {
        newNetwork.trainBatch(this.trialInputsList, residuals);
      }
      newNetwork.shrinkWeights(shrinkage);
      this.neuralNetworks.push(newNetwork);
      this.numOfNeuralNetworks = this.neuralNetworks.length;
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
    runEnsemble(inputs) {
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
    getEnsembleNetwork() {
      const ensemble = this.neuralNetworks[0].clone();
      const originalRun = ensemble.run.bind(ensemble);
      ensemble.run = (inputs) => {
        const ensembleOutput = this.runEnsemble(inputs);
        originalRun(inputs);
        for (let i = 0; i < ensembleOutput.length; i++) {
          ensemble.getOutputLayer().neurons[i].value = ensembleOutput[i];
        }
        return ensemble.getOutputLayer();
      };
      return ensemble;
    }
    sort() {
      this.neuralNetworks.sort((a, b) => a.error - b.error);
    }
    numToColorBlack(num) {
      if (num >= 0) {
        return "rgb(0, " + Math.tanh(num) * 255 + ",0)";
      }
      if (num < 0) {
        return "rgb(" + -Math.tanh(num) * 255 + ", 0 ,0)";
      }
      return "";
    }
    numToColorWhite(num) {
      if (num >= 0) {
        return "rgb(" + (1 - Math.tanh(num)) * 255 + ", 255," + (1 - Math.tanh(num)) * 255 + ")";
      }
      if (num < 0) {
        return "rgb(255, " + (1 + Math.tanh(num)) * 255 + "," + (1 + Math.tanh(num)) * 255 + ")";
      }
      return "";
    }
  };
  var NeuralNetwork = class _NeuralNetwork {
    error = 0;
    meanError = 0;
    numOfLayers;
    inputSize;
    hiddenLayerSizes;
    outputSize;
    startRandomBias = true;
    startRandomWeights = true;
    clampBiases = false;
    clampWeights = false;
    changeWeightAlpha = false;
    linePower = 4;
    layers = [];
    activationFunction = "tanh";
    outputActivationFunction = "tanh";
    // For backpropagation
    learningRate = 0.01;
    // Default learning rate (adjustable via UI)
    momentum = 0.9;
    // Default momentum (adjustable via UI)
    constructor(inputSize2, hiddenLayerSizes2, outputSize2, activationFunction2, outputActivationFunction2) {
      this.numOfLayers = hiddenLayerSizes2.length + 2;
      this.inputSize = inputSize2;
      this.hiddenLayerSizes = hiddenLayerSizes2;
      this.outputSize = outputSize2;
      this.initLayers();
      this.activationFunction = activationFunction2;
      this.outputActivationFunction = outputActivationFunction2;
    }
    run(inputs) {
      if (inputs.length != this.inputSize) {
        throw new Error("Wrong Number of INPUTs, Expected: " + this.inputSize + "| Received: " + inputs.length);
      }
      for (let i = 0; i < this.inputSize; i++) {
        this.layers[0].neurons[i].value = inputs[i];
      }
      for (let l = 1; l < this.numOfLayers; l++) {
        let neurons = this.layers[l].neurons;
        for (let n = 0; n < neurons.length; n++) {
          let neuron = neurons[n];
          neuron.value = 0;
          let weights = neuron.weights;
          for (let w = 0; w < weights.length; w++) {
            let weight = weights[w];
            neuron.value += weight.value * this.getNodeValue(weight.to);
          }
          neuron.value += neuron.bias;
          neuron.activate();
        }
      }
      return this.layers[this.layers.length - 1];
    }
    mutate(numOfWeights, weightStrength, numOfBiases, biasesStrength) {
      for (let i = 0; i < numOfWeights; i++) {
        let randLayerIndex = Math.floor(Math.random() * (this.numOfLayers - 1)) + 1;
        let neurons = this.layers[randLayerIndex].neurons;
        let randNeuronIndex = Math.floor(Math.random() * neurons.length);
        let weights = this.layers[randLayerIndex].neurons[randNeuronIndex].weights;
        let randWeightsIndex = Math.floor(Math.random() * weights.length);
        weights[randWeightsIndex].value += (Math.random() * 2 - 1) * weightStrength;
        if (this.clampWeights) {
          weights[randWeightsIndex].value = Math.min(Math.max(weights[randWeightsIndex].value, -1), 1);
        }
      }
      for (let i = 0; i < numOfBiases; i++) {
        let randLayerIndex = Math.floor(Math.random() * (this.numOfLayers - 1)) + 1;
        let neurons = this.layers[randLayerIndex].neurons;
        let randNeuronIndex = Math.floor(Math.random() * neurons.length);
        neurons[randNeuronIndex].bias += (Math.random() * 2 - 1) * biasesStrength;
        if (this.clampBiases) {
          neurons[randNeuronIndex].bias = Math.min(Math.max(neurons[randNeuronIndex].bias, -1), 1);
        }
      }
    }
    // Randomize all weights and biases (for creating new networks in XGBoost)
    randomize() {
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
    testError(outputs, additive, power) {
      let sum = 0;
      let meanSum = 0;
      const p = power || 2;
      for (let i = 0; i < this.outputSize; i++) {
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
    backPropogate(targetOutputs, resetGradients = true) {
      if (targetOutputs.length !== this.outputSize) {
        throw new Error("Wrong number of target outputs");
      }
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
      const outputLayer = this.layers[this.numOfLayers - 1];
      for (let i = 0; i < outputLayer.neurons.length; i++) {
        const neuron = outputLayer.neurons[i];
        const error = neuron.value - targetOutputs[i];
        neuron.gradient = 2 * error * neuron.activationDerivative();
      }
      for (let l = this.numOfLayers - 2; l >= 0; l--) {
        const currentLayer = this.layers[l];
        const nextLayer = this.layers[l + 1];
        for (let i = 0; i < currentLayer.neurons.length; i++) {
          const neuron = currentLayer.neurons[i];
          let sum = 0;
          for (let j = 0; j < nextLayer.neurons.length; j++) {
            const nextNeuron = nextLayer.neurons[j];
            for (let w = 0; w < nextNeuron.weights.length; w++) {
              const weight = nextNeuron.weights[w];
              if (weight.to.layer === l && weight.to.index === i) {
                sum += nextNeuron.gradient * weight.value;
                weight.gradient += nextNeuron.gradient * neuron.value;
              }
            }
          }
          neuron.gradient = sum * neuron.activationDerivative();
        }
      }
    }
    // Apply gradients to update weights and biases with momentum
    applyGradients() {
      for (let l = 1; l < this.numOfLayers; l++) {
        for (let n = 0; n < this.layers[l].neurons.length; n++) {
          const neuron = this.layers[l].neurons[n];
          neuron.biasVelocity = this.momentum * neuron.biasVelocity - this.learningRate * neuron.gradient;
          neuron.bias += neuron.biasVelocity;
          if (this.clampBiases) {
            neuron.bias = Math.max(-1, Math.min(1, neuron.bias));
          }
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
    shrinkWeights(shrinkage) {
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
    trainBatch(inputs, targetOutputs) {
      if (inputs.length !== targetOutputs.length) {
        throw new Error("Inputs and target outputs must have same length");
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
      this.applyGradients();
    }
    getNodeValue(np) {
      return this.layers[np.layer].neurons[np.index].value;
    }
    getInputLayer() {
      return this.layers[0];
    }
    getOutputLayer() {
      return this.layers[this.numOfLayers - 1];
    }
    clone() {
      let nn = new _NeuralNetwork(this.inputSize, this.hiddenLayerSizes, this.outputSize, this.activationFunction, this.outputActivationFunction);
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
    initLayers() {
      this.initNextLayer(this.inputSize, false, this.activationFunction);
      for (let i = 0; i < this.hiddenLayerSizes.length; i++) {
        this.initNextLayer(this.hiddenLayerSizes[i], this.startRandomBias, this.activationFunction);
      }
      this.initNextLayer(this.outputSize, this.startRandomBias, this.outputActivationFunction);
    }
    initNextLayer(numOfNeurons, randomBias, activationFunction2) {
      const layerNeurons = [];
      for (let j = 0; j < numOfNeurons; j++) {
        const neuronWeights = this.initWeights();
        layerNeurons.push(new Neuron(neuronWeights, randomBias ? Math.random() * 2 - 1 : 0, activationFunction2));
      }
      this.layers.push(new Layer(layerNeurons));
    }
    initWeights() {
      if (this.layers.length == 0) {
        return [];
      }
      let w = [];
      let lastLayerIndex = this.layers.length - 1;
      for (let i = 0; i < this.layers[lastLayerIndex].neurons.length; i++) {
        w.push(new Weight(this.startRandomWeights ? Math.random() * 2 - 1 : 0, new NeuronPosition(lastLayerIndex, i)));
      }
      return w;
    }
    write() {
      console.log(this.layers);
      for (let i = 0; i < this.layers.length; i++) {
        let str = "";
        for (let j = 0; j < this.layers[i].neurons.length; j++) {
          let neuron = this.layers[i].neurons[j];
          str += neuron.bias;
          str += " ";
        }
      }
    }
    draw(ctx2, left, top, width, height, displayErrorDigits, displayMeanError) {
      let spaceX = width / (this.numOfLayers + 1);
      let lastYs = [];
      const bottomPadding = (displayErrorDigits ? 15 : 0) + (displayMeanError ? 15 : 0);
      for (let i = 0; i < this.numOfLayers; i++) {
        let x = (i + 1) * spaceX + left;
        let lastX = i * spaceX + left;
        let neurons = this.layers[i].neurons;
        let spaceY = (height - bottomPadding) / (neurons.length + 1);
        let newYs = [];
        for (let j = 0; j < neurons.length; j++) {
          let neuron = neurons[j];
          let y = (j + 1) * spaceY + top;
          newYs.push(y);
          for (let w = 0; w < lastYs.length; w++) {
            ctx2.save();
            ctx2.strokeStyle = this.numToColorBlack(neuron.weights[w].value);
            ctx2.lineWidth = 2;
            if (this.changeWeightAlpha)
              ctx2.globalAlpha = Math.abs(neuron.weights[w].value) ** this.linePower;
            ctx2.beginPath();
            ctx2.moveTo(x, y);
            ctx2.lineTo(lastX, lastYs[w]);
            ctx2.stroke();
            ctx2.restore();
          }
        }
        lastYs = newYs;
      }
      for (let i = 0; i < this.numOfLayers; i++) {
        let neurons = this.layers[i].neurons;
        let spaceY = (height - bottomPadding) / (neurons.length + 1);
        let x = (i + 1) * spaceX + left;
        for (let j = 0; j < neurons.length; j++) {
          let neuron = neurons[j];
          let y = (j + 1) * spaceY + top;
          let r1 = Math.min(spaceX, spaceY) / 2.1;
          ctx2.save();
          ctx2.fillStyle = this.numToColorBlack(neuron.bias);
          drawCircle(ctx2, { x, y }, r1);
          ctx2.restore();
          let r2 = Math.min(spaceX, spaceY) / 2.3;
          ctx2.save();
          ctx2.fillStyle = this.numToColorWhite(neuron.value);
          drawCircle(ctx2, { x, y }, r2);
          ctx2.restore();
          ctx2.save();
          let val = Math.floor(neuron.value * 1e3) / 1e3;
          let textStr = val.toString();
          let drawn = false;
          ctx2.font = "12px sans-serif";
          ctx2.textBaseline = "middle";
          ctx2.fillStyle = "#000000";
          while (textStr.length > 0 && textStr != "-" && !drawn) {
            if (textStr.at(-1) == ".") {
              textStr = textStr.slice(0, -1);
              continue;
            }
            let measure = ctx2.measureText(textStr);
            let w = measure.width;
            if (w >= r2 * 1.8) {
              textStr = textStr.slice(0, -1);
              continue;
            }
            drawn = true;
            ctx2.fillText(textStr, x - w / 2, y);
            ctx2.restore();
          }
        }
      }
      if (displayErrorDigits) {
        ctx2.font = "12px sans-serif";
        ctx2.fillStyle = "#000000";
        let textStr = "\u03F5 RMSE: " + (Math.floor(this.error * 10 ** displayErrorDigits) / 10 ** displayErrorDigits).toString();
        let drawn = false;
        while (textStr.length > 0 && textStr != "-" && !drawn) {
          let measure = ctx2.measureText(textStr);
          let w = measure.width;
          if (w >= width) {
            textStr = textStr.slice(0, -1);
            continue;
          }
          drawn = true;
          const yOffset = displayMeanError ? 20 : 10;
          ctx2.fillText(textStr, left + width / 2 - w / 2, top + height - yOffset);
          ctx2.restore();
        }
      }
      if (displayMeanError) {
        ctx2.font = "12px sans-serif";
        ctx2.fillStyle = "#000000";
        let textStr = "\u03BC MAE: " + (Math.floor(this.meanError * 10 ** displayErrorDigits) / 10 ** displayErrorDigits).toString();
        let drawn = false;
        while (textStr.length > 0 && textStr != "-" && !drawn) {
          let measure = ctx2.measureText(textStr);
          let w = measure.width;
          if (w >= width) {
            textStr = textStr.slice(0, -1);
            continue;
          }
          drawn = true;
          ctx2.fillText(textStr, left + width / 2 - w / 2, top + height - 10);
          ctx2.restore();
        }
      }
    }
    displayGrid2Input1Output(ctx2, left, top, width, height, axis1low, axis2low, axis1high, axis2high, rows, columns, decimals, showText, showHeaders, getCellValue, getColorValue) {
      const headerOffset = showHeaders ? 1 : 0;
      let spaceX = width / (columns + headerOffset);
      let spaceY = height / (rows + headerOffset);
      ctx2.save();
      ctx2.font = `${Math.min(spaceX, spaceY) * 0.3}px sans-serif`;
      ctx2.textAlign = "center";
      ctx2.textBaseline = "middle";
      ctx2.fillStyle = "#e0e0e0";
      ctx2.fillRect(left, top, width, height);
      if (showHeaders) {
        for (let i = 0; i < columns; i++) {
          let x = left + (i + 1) * spaceX;
          let y = top;
          let headerValue = axis1low + i * (axis1high - axis1low) / (columns - 1);
          ctx2.fillStyle = "#000000";
          let text = headerValue.toFixed(2);
          if (ctx2.measureText(text).width > spaceX * 0.9) {
            text = headerValue.toFixed(1);
          }
          if (ctx2.measureText(text).width > spaceX * 0.9) {
            text = headerValue.toFixed(0);
          }
          ctx2.fillText(text, x + spaceX / 2, y + spaceY / 2);
        }
        for (let j = 0; j < rows; j++) {
          let x = left;
          let y = top + (j + 1) * spaceY;
          let headerValue = axis2low + j * (axis2high - axis2low) / (rows - 1);
          ctx2.fillStyle = "#e0e0e0";
          ctx2.fillRect(x, y, spaceX, spaceY);
          ctx2.fillStyle = "#000000";
          let text = headerValue.toFixed(2);
          if (ctx2.measureText(text).width > spaceX * 0.9) {
            text = headerValue.toFixed(1);
          }
          if (ctx2.measureText(text).width > spaceX * 0.9) {
            text = headerValue.toFixed(0);
          }
          ctx2.fillText(text, x + spaceX / 2, y + spaceY / 2);
        }
        ctx2.fillStyle = "#c0c0c0";
        ctx2.fillRect(left, top, spaceX, spaceY);
      }
      for (let i = 0; i < columns; i++) {
        for (let j = 0; j < rows; j++) {
          let x = left + (i + headerOffset) * spaceX;
          let y = top + (j + headerOffset) * spaceY;
          let input12 = axis1low + i * (axis1high - axis1low) / (columns - 1);
          let input22 = axis2low + j * (axis2high - axis2low) / (rows - 1);
          let cellValue = getCellValue(input12, input22);
          ctx2.fillStyle = this.numToColorWhite(getColorValue(cellValue));
          ctx2.fillRect(x, y, spaceX + (showText ? -1 : 1), spaceY + (showText ? -1 : 1));
          if (showText) {
            ctx2.fillStyle = "#000000";
            let text = cellValue.toFixed(decimals);
            let textWidth = ctx2.measureText(text).width;
            if (textWidth > spaceX * 0.9) {
              text = cellValue.toFixed(Math.max(0, decimals - 1));
              textWidth = ctx2.measureText(text).width;
              if (textWidth > spaceX * 0.9) {
                text = cellValue.toFixed(0);
              }
            }
            ctx2.fillText(text, x + spaceX / 2, y + spaceY / 2);
          }
        }
      }
      ctx2.restore();
    }
    display2Input1OutputError(ctx2, left, top, width, height, axis1low, axis2low, axis1high, axis2high, errorRange, rows, columns, decimals, showText, showHeaders, test2) {
      this.displayGrid2Input1Output(ctx2, left, top, width, height, axis1low, axis2low, axis1high, axis2high, rows, columns, decimals, showText, showHeaders, (input12, input22) => {
        let val = this.run([input12, input22]);
        return val.neurons[0].value - test2([input12, input22])[0];
      }, (cellValue) => cellValue * errorRange);
    }
    display2Input1OutputTest(ctx2, left, top, width, height, axis1low, axis2low, axis1high, axis2high, ouputMiddle, outputRange, rows, columns, decimals, showText, showHeaders, test2) {
      this.displayGrid2Input1Output(ctx2, left, top, width, height, axis1low, axis2low, axis1high, axis2high, rows, columns, decimals, showText, showHeaders, (input12, input22) => test2([input12, input22])[0], (cellValue) => cellValue * outputRange - ouputMiddle);
    }
    display2Input1Output(ctx2, left, top, width, height, axis1low, axis2low, axis1high, axis2high, ouputMiddle, outputRange, rows, columns, decimals, showText, showHeaders) {
      this.displayGrid2Input1Output(ctx2, left, top, width, height, axis1low, axis2low, axis1high, axis2high, rows, columns, decimals, showText, showHeaders, (input12, input22) => this.run([input12, input22]).neurons[0].value, (cellValue) => cellValue * outputRange - ouputMiddle);
    }
    numToColorBlack(num) {
      if (num >= 0) {
        return "rgb(0, " + Math.tanh(num) * 255 + ",0)";
      }
      if (num < 0) {
        return "rgb(" + -Math.tanh(num) * 255 + ", 0 ,0)";
      }
      return "";
    }
    numToColorWhite(num) {
      if (num >= 0) {
        return "rgb(" + (1 - Math.tanh(num)) * 255 + ", 255," + (1 - Math.tanh(num)) * 255 + ")";
      }
      if (num < 0) {
        return "rgb(255, " + (1 + Math.tanh(num)) * 255 + "," + (1 + Math.tanh(num)) * 255 + ")";
      }
      return "";
    }
    interpolateColor(color12, color22, weight) {
      const clampedWeight = Math.max(0, Math.min(1, weight));
      const r = Math.round(color12.r * clampedWeight + color22.r * (1 - clampedWeight));
      const g = Math.round(color12.g * clampedWeight + color22.g * (1 - clampedWeight));
      const b = Math.round(color12.b * clampedWeight + color22.b * (1 - clampedWeight));
      return `rgb(${r}, ${g}, ${b})`;
    }
    interpolateColorPublic(color12, color22, weight) {
      return this.interpolateColor(color12, color22, weight);
    }
    displayGrid2Input2Output(ctx2, left, top, width, height, axis1low, axis2low, axis1high, axis2high, rows, columns, decimals, showText, showHeaders, color12, color22, getOutputs) {
      const headerOffset = showHeaders ? 1 : 0;
      let spaceX = width / (columns + headerOffset);
      let spaceY = height / (rows + headerOffset);
      ctx2.save();
      ctx2.font = `${Math.min(spaceX, spaceY) * 0.3}px sans-serif`;
      ctx2.textAlign = "center";
      ctx2.textBaseline = "middle";
      ctx2.fillStyle = "#e0e0e0";
      ctx2.fillRect(left, top, width, height);
      if (showHeaders) {
        for (let i = 0; i < columns; i++) {
          let x = left + (i + 1) * spaceX;
          let y = top;
          let headerValue = axis1low + i * (axis1high - axis1low) / (columns - 1);
          ctx2.fillStyle = "#000000";
          let text = headerValue.toFixed(2);
          if (ctx2.measureText(text).width > spaceX * 0.9) {
            text = headerValue.toFixed(1);
          }
          if (ctx2.measureText(text).width > spaceX * 0.9) {
            text = headerValue.toFixed(0);
          }
          ctx2.fillText(text, x + spaceX / 2, y + spaceY / 2);
        }
        for (let j = 0; j < rows; j++) {
          let x = left;
          let y = top + (j + 1) * spaceY;
          let headerValue = axis2low + j * (axis2high - axis2low) / (rows - 1);
          ctx2.fillStyle = "#e0e0e0";
          ctx2.fillRect(x, y, spaceX, spaceY);
          ctx2.fillStyle = "#000000";
          let text = headerValue.toFixed(2);
          if (ctx2.measureText(text).width > spaceX * 0.9) {
            text = headerValue.toFixed(1);
          }
          if (ctx2.measureText(text).width > spaceX * 0.9) {
            text = headerValue.toFixed(0);
          }
          ctx2.fillText(text, x + spaceX / 2, y + spaceY / 2);
        }
        ctx2.fillStyle = "#c0c0c0";
        ctx2.fillRect(left, top, spaceX, spaceY);
      }
      for (let i = 0; i < columns; i++) {
        for (let j = 0; j < rows; j++) {
          let x = left + (i + headerOffset) * spaceX;
          let y = top + (j + headerOffset) * spaceY;
          let input12 = axis1low + i * (axis1high - axis1low) / (columns - 1);
          let input22 = axis2low + j * (axis2high - axis2low) / (rows - 1);
          let outputs = getOutputs(input12, input22);
          ctx2.fillStyle = this.interpolateColor(color12, color22, outputs[0]);
          ctx2.fillRect(x, y, spaceX + (showText ? -1 : 1), spaceY + (showText ? -1 : 1));
          if (showText) {
            ctx2.fillStyle = "#000000";
            let text = outputs[0].toFixed(decimals) + "," + outputs[1].toFixed(decimals);
            let textWidth = ctx2.measureText(text).width;
            let shrink = 0;
            while (textWidth > spaceX * 0.9 && shrink < decimals) {
              text = outputs[0].toFixed(decimals - shrink) + "," + outputs[1].toFixed(decimals - shrink);
              textWidth = ctx2.measureText(text).width;
              if (textWidth > spaceX * 0.9) {
                text = outputs[0].toFixed(decimals - shrink);
              }
              shrink++;
            }
            ctx2.fillText(text, x + spaceX / 2, y + spaceY / 2);
          }
        }
      }
      ctx2.restore();
    }
    display2Input2Output(ctx2, left, top, width, height, axis1low, axis2low, axis1high, axis2high, rows, columns, decimals, showText, showHeaders, color12, color22) {
      this.displayGrid2Input2Output(ctx2, left, top, width, height, axis1low, axis2low, axis1high, axis2high, rows, columns, decimals, showText, showHeaders, color12, color22, (input12, input22) => {
        let result = this.run([input12, input22]);
        return [result.neurons[0].value, result.neurons[1].value];
      });
    }
    display2Input2OutputTest(ctx2, left, top, width, height, axis1low, axis2low, axis1high, axis2high, rows, columns, decimals, showText, showHeaders, color12, color22, test2) {
      this.displayGrid2Input2Output(ctx2, left, top, width, height, axis1low, axis2low, axis1high, axis2high, rows, columns, decimals, showText, showHeaders, color12, color22, (input12, input22) => test2([input12, input22]));
    }
    display2Input2OutputError(ctx2, left, top, width, height, axis1low, axis2low, axis1high, axis2high, rows, columns, decimals, showText, showHeaders, color12, color22, test2) {
      const errorColor1 = { r: 255, g: 255, b: 255 };
      const errorColor2 = { r: 255, g: 0, b: 0 };
      this.displayGrid2Input2Output(ctx2, left, top, width, height, axis1low, axis2low, axis1high, axis2high, rows, columns, decimals, showText, showHeaders, errorColor1, errorColor2, (input12, input22) => {
        let nnOutput = this.run([input12, input22]);
        let testOutput = test2([input12, input22]);
        let error0 = Math.abs(nnOutput.neurons[0].value - testOutput[0]);
        let error1 = Math.abs(nnOutput.neurons[1].value - testOutput[1]);
        let totalError = (error0 + error1) / 2;
        return [totalError, 0];
      });
    }
  };
  var Layer = class {
    neurons;
    constructor(neurons) {
      this.neurons = neurons;
    }
  };
  var Neuron = class {
    value;
    weights;
    bias;
    activationFunction;
    // For backpropagation
    gradient = 0;
    // Gradient of the loss with respect to this neuron's output
    preActivation = 0;
    // Value before activation function
    biasVelocity = 0;
    // Momentum for bias
    constructor(weights, bias, activationFunction2) {
      this.value = 0;
      this.weights = weights;
      this.bias = bias;
      this.activationFunction = activationFunction2;
    }
    activate() {
      this.preActivation = this.value;
      switch (this.activationFunction) {
        case "relu":
          this.value = this.value <= 0 ? 0 : this.value;
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
    activationDerivative() {
      switch (this.activationFunction) {
        case "relu":
          return this.preActivation > 0 ? 1 : 0;
        case "sigmoid":
          return this.value * (1 - this.value);
        case "tanh":
          return 1 - this.value * this.value;
      }
      return 0;
    }
  };
  var Weight = class {
    value;
    to;
    gradient = 0;
    velocity = 0;
    // Momentum for weight
    constructor(value, to) {
      this.value = value;
      this.to = to;
    }
  };
  var NeuronPosition = class {
    layer;
    index;
    constructor(layer, index) {
      this.layer = layer;
      this.index = index;
    }
  };

  // decisiontree.js
  var TreeNode = class {
    feature = null;
    // Which input feature to split on (0 or 1)
    threshold = null;
    // Split threshold
    left = null;
    right = null;
    value = null;
    // Leaf value (for each output)
    isLeaf() {
      return this.value !== null;
    }
  };
  var DecisionTree = class {
    root = new TreeNode();
    maxDepth = 4;
    minSamplesLeaf = 2;
    inputSize;
    outputSize;
    isClassification = false;
    _drawLeft = 0;
    _drawRight = 0;
    constructor(inputSize2, outputSize2, maxDepth = 4) {
      this.inputSize = inputSize2;
      this.outputSize = outputSize2;
      this.maxDepth = maxDepth;
    }
    // Train the decision tree on input-output pairs
    train(inputs, outputs) {
      this.root = this.buildTree(inputs, outputs, 0);
    }
    // Recursively build the tree
    buildTree(inputs, outputs, depth) {
      const node = new TreeNode();
      if (depth >= this.maxDepth || inputs.length < this.minSamplesLeaf) {
        node.value = this.calculateMean(outputs);
        return node;
      }
      const bestSplit = this.findBestSplit(inputs, outputs);
      if (bestSplit === null) {
        node.value = this.calculateMean(outputs);
        return node;
      }
      node.feature = bestSplit.feature;
      node.threshold = bestSplit.threshold;
      const [leftInputs, leftOutputs, rightInputs, rightOutputs] = this.splitData(inputs, outputs, bestSplit.feature, bestSplit.threshold);
      node.left = this.buildTree(leftInputs, leftOutputs, depth + 1);
      node.right = this.buildTree(rightInputs, rightOutputs, depth + 1);
      return node;
    }
    // Find the best feature and threshold to split on
    findBestSplit(inputs, outputs) {
      let bestVariance = Infinity;
      let bestSplit = null;
      for (let feature = 0; feature < this.inputSize; feature++) {
        const values = inputs.map((inp) => inp[feature]);
        const uniqueValues = [...new Set(values)].sort((a, b) => a - b);
        for (let i = 0; i < uniqueValues.length - 1; i++) {
          const threshold = (uniqueValues[i] + uniqueValues[i + 1]) / 2;
          const [leftInputs, leftOutputs, rightInputs, rightOutputs] = this.splitData(inputs, outputs, feature, threshold);
          if (leftOutputs.length === 0 || rightOutputs.length === 0)
            continue;
          const leftVar = this.calculateVariance(leftOutputs);
          const rightVar = this.calculateVariance(rightOutputs);
          const totalVar = (leftOutputs.length * leftVar + rightOutputs.length * rightVar) / outputs.length;
          if (totalVar < bestVariance) {
            bestVariance = totalVar;
            bestSplit = { feature, threshold };
          }
        }
      }
      return bestSplit;
    }
    // Split data based on feature and threshold
    splitData(inputs, outputs, feature, threshold) {
      const leftInputs = [];
      const leftOutputs = [];
      const rightInputs = [];
      const rightOutputs = [];
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
    calculateMean(outputs) {
      if (outputs.length === 0)
        return new Array(this.outputSize).fill(0);
      const sum = new Array(this.outputSize).fill(0);
      for (const output of outputs) {
        for (let i = 0; i < this.outputSize; i++) {
          sum[i] += output[i];
        }
      }
      return sum.map((s) => s / outputs.length);
    }
    // Calculate variance of outputs
    calculateVariance(outputs) {
      if (outputs.length === 0)
        return 0;
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
    predict(input) {
      let node = this.root;
      while (!node.isLeaf()) {
        if (input[node.feature] < node.threshold) {
          node = node.left;
        } else {
          node = node.right;
        }
      }
      return node.value;
    }
    // Get the set of nodes on the prediction path for a given input
    getPath(input) {
      const path = /* @__PURE__ */ new Set();
      let node = this.root;
      while (node) {
        path.add(node);
        if (node.isLeaf())
          break;
        if (input[node.feature] < node.threshold) {
          node = node.left;
        } else {
          node = node.right;
        }
      }
      return path;
    }
    // Draw the tree, optionally highlighting a prediction path
    draw(ctx2, left, top, width, height, highlightPath) {
      ctx2.save();
      this._drawLeft = left;
      this._drawRight = left + width;
      ctx2.fillStyle = "#f8f8f8";
      ctx2.fillRect(left, top, width, height);
      if (this.root) {
        this.drawNode(ctx2, this.root, left + width / 2, top + 15, width * 0.8, height - 20, 0, highlightPath);
      }
      ctx2.restore();
    }
    // Recursively draw tree nodes
    drawNode(ctx2, node, x, y, width, height, depth, highlightPath) {
      const onPath = highlightPath ? highlightPath.has(node) : false;
      const nodeRadius = onPath ? 10 : 8;
      const verticalGap = 25;
      const clampX = (textX, text, font) => {
        ctx2.font = font;
        const tw = ctx2.measureText(text).width;
        const minX = this._drawLeft + tw / 2 + 2;
        const maxX = this._drawRight - tw / 2 - 2;
        return Math.max(minX, Math.min(maxX, textX));
      };
      if (node.isLeaf()) {
        ctx2.fillStyle = onPath ? "#E53935" : "#4CAF50";
        ctx2.beginPath();
        ctx2.arc(x, y, nodeRadius, 0, 2 * Math.PI);
        ctx2.fill();
        const leafFont = onPath ? "bold 10px sans-serif" : "9px sans-serif";
        ctx2.fillStyle = onPath ? "#E53935" : "#000";
        ctx2.font = leafFont;
        ctx2.textAlign = "center";
        let valStr;
        if (this.isClassification && node.value.length > 1) {
          let maxIdx = 0;
          for (let k = 1; k < node.value.length; k++) {
            if (node.value[k] > node.value[maxIdx])
              maxIdx = k;
          }
          valStr = (maxIdx + 1).toString();
        } else {
          valStr = node.value.map((v) => v.toFixed(2)).join(", ");
        }
        ctx2.fillText(valStr, clampX(x, valStr, leafFont), y + nodeRadius + 10);
      } else {
        ctx2.fillStyle = onPath ? "#E53935" : "#2196F3";
        ctx2.beginPath();
        ctx2.arc(x, y, nodeRadius, 0, 2 * Math.PI);
        ctx2.fill();
        const splitFont = onPath ? "bold 9px sans-serif" : "8px sans-serif";
        ctx2.fillStyle = onPath ? "#E53935" : "#000";
        ctx2.font = splitFont;
        ctx2.textAlign = "center";
        const featureName = node.feature === 0 ? "x" : "y";
        const splitText = `${featureName}<${node.threshold.toFixed(1)}`;
        ctx2.fillText(splitText, clampX(x, splitText, splitFont), y - nodeRadius - 3);
        if (node.left && node.right) {
          const childWidth = width / 2;
          const leftX = x - width / 4;
          const rightX = x + width / 4;
          const childY = y + verticalGap;
          const leftOnPath = highlightPath ? highlightPath.has(node.left) : false;
          ctx2.strokeStyle = leftOnPath ? "#E53935" : "#666";
          ctx2.lineWidth = leftOnPath ? 2.5 : 1;
          ctx2.beginPath();
          ctx2.moveTo(x, y + nodeRadius);
          ctx2.lineTo(leftX, childY - (leftOnPath ? 10 : 8));
          ctx2.stroke();
          const rightOnPath = highlightPath ? highlightPath.has(node.right) : false;
          ctx2.strokeStyle = rightOnPath ? "#E53935" : "#666";
          ctx2.lineWidth = rightOnPath ? 2.5 : 1;
          ctx2.beginPath();
          ctx2.moveTo(x, y + nodeRadius);
          ctx2.lineTo(rightX, childY - (rightOnPath ? 10 : 8));
          ctx2.stroke();
          this.drawNode(ctx2, node.left, leftX, childY, childWidth, height - verticalGap, depth + 1, highlightPath);
          this.drawNode(ctx2, node.right, rightX, childY, childWidth, height - verticalGap, depth + 1, highlightPath);
        }
      }
    }
  };

  // xgboost.js
  var XGBoostEnsemble = class _XGBoostEnsemble {
    trees = [];
    generation = 0;
    inputSize;
    outputSize;
    shrinkage = 0.1;
    maxDepth = 4;
    maxTrees = Infinity;
    isClassification = false;
    trainRMSE = 0;
    trainMAE = 0;
    testRMSE = 0;
    testMAE = 0;
    constructor(inputSize2, outputSize2, shrinkage = 0.1, maxDepth = 4, maxTrees = Infinity, isClassification = false) {
      this.inputSize = inputSize2;
      this.outputSize = outputSize2;
      this.shrinkage = shrinkage;
      this.maxDepth = maxDepth;
      this.maxTrees = maxTrees;
      this.isClassification = isClassification;
    }
    // Train: Add one new tree that learns the residuals
    train(inputs, outputs, testInputs, testFn) {
      this.generation++;
      if (this.trees.length === 0) {
        const baseTree = new DecisionTree(this.inputSize, this.outputSize, this.maxDepth);
        baseTree.isClassification = this.isClassification;
        baseTree.train(inputs, outputs);
        this.trees.push(baseTree);
        this.computeErrors(inputs, outputs, testInputs, testFn);
        return;
      }
      if (this.trees.length < this.maxTrees) {
        const residuals = [];
        for (let i = 0; i < inputs.length; i++) {
          const pred = this.predict(inputs[i]);
          const residual = outputs[i].map((val, idx) => val - pred[idx]);
          residuals.push(residual);
        }
        const newTree = new DecisionTree(this.inputSize, this.outputSize, this.maxDepth);
        newTree.isClassification = this.isClassification;
        newTree.train(inputs, residuals);
        this.trees.push(newTree);
      }
      this.computeErrors(inputs, outputs, testInputs, testFn);
    }
    computeErrors(inputs, outputs, testInputs, testFn) {
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
      if (!testInputs || !testFn) {
        this.testRMSE = 0;
        this.testMAE = 0;
        return;
      }
      sqSum = 0;
      absSum = 0;
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
    predict(input) {
      if (this.trees.length === 0) {
        return new Array(this.outputSize).fill(0);
      }
      const result = new Array(this.outputSize).fill(0);
      for (let i = 0; i < this.trees.length; i++) {
        const treePred = this.trees[i].predict(input);
        const weight = i === 0 ? 1 : this.shrinkage;
        for (let j = 0; j < this.outputSize; j++) {
          result[j] += treePred[j] * weight;
        }
      }
      return result;
    }
    static MIN_CELL_HEIGHT = 200;
    // Returns the total content height (for scroll calculations)
    getContentHeight(displayHeaderHeight, panelHeight) {
      if (this.trees.length === 0)
        return displayHeaderHeight;
      const cols = Math.min(4, this.trees.length);
      const rows = Math.ceil(this.trees.length / cols);
      const availableHeight = panelHeight - displayHeaderHeight;
      const cellHeight = Math.max(_XGBoostEnsemble.MIN_CELL_HEIGHT, availableHeight / rows);
      return displayHeaderHeight + rows * cellHeight;
    }
    drawHeader(ctx2, left, top, width, displayHeaderHeight) {
      ctx2.fillStyle = "#e0e0e0";
      ctx2.fillRect(left, top, width, displayHeaderHeight);
      ctx2.fillStyle = "#000000";
      ctx2.font = "12px sans-serif";
      ctx2.textBaseline = "middle";
      const text = `Gen: ${this.generation} | Trees: ${this.trees.length} | Train \u03F5 RMSE: ${this.trainRMSE.toFixed(4)} \u03BC MAE: ${this.trainMAE.toFixed(4)} | Test \u03F5 RMSE: ${this.testRMSE.toFixed(4)} \u03BC MAE: ${this.testMAE.toFixed(4)}`;
      ctx2.fillText(text, left + 5, top + displayHeaderHeight / 2);
    }
    // Draw a single tree (base or latest) filling the full content area (no header — caller draws it)
    drawSingleTree(ctx2, left, top, width, height, treeIndex, input) {
      ctx2.save();
      if (this.trees.length === 0 || treeIndex < 0 || treeIndex >= this.trees.length) {
        ctx2.restore();
        return;
      }
      const tree = this.trees[treeIndex];
      ctx2.fillStyle = "#000000";
      ctx2.font = "11px sans-serif";
      ctx2.textAlign = "left";
      ctx2.textBaseline = "top";
      const label = treeIndex === 0 ? `Tree 0 (base)` : `Tree ${treeIndex}`;
      ctx2.fillText(label, left + 5, top + 2);
      const highlightPath = input ? tree.getPath(input) : void 0;
      tree.draw(ctx2, left, top + 15, width, height - 15, highlightPath);
      if (input && highlightPath) {
        const pred = tree.predict(input);
        ctx2.fillStyle = "#E53935";
        ctx2.font = "bold 12px sans-serif";
        ctx2.textAlign = "right";
        ctx2.textBaseline = "top";
        ctx2.fillText(`Prediction: ${pred.map((v) => v.toFixed(3)).join(", ")}`, left + width - 5, top + 2);
      }
      ctx2.restore();
    }
    // Draw all trees in a scrollable grid, with input path highlighting
    draw(ctx2, left, top, width, height, displayHeaderHeight, scrollY = 0, input) {
      ctx2.save();
      ctx2.beginPath();
      ctx2.rect(left, top + displayHeaderHeight, width, height - displayHeaderHeight);
      ctx2.clip();
      if (this.trees.length === 0) {
        ctx2.restore();
        return;
      }
      const cols = Math.min(4, this.trees.length);
      const rows = Math.ceil(this.trees.length / cols);
      const cellWidth = width / cols;
      const availableHeight = height - displayHeaderHeight;
      const cellHeight = Math.max(_XGBoostEnsemble.MIN_CELL_HEIGHT, availableHeight / rows);
      for (let i = 0; i < this.trees.length; i++) {
        const col = i % cols;
        const row = Math.floor(i / cols);
        const x = left + col * cellWidth;
        const y = top + displayHeaderHeight + row * cellHeight - scrollY;
        if (y + cellHeight < top + displayHeaderHeight || y > top + height)
          continue;
        ctx2.fillStyle = i === 0 ? "#f0f0ff" : "#f8f8f8";
        ctx2.fillRect(x + 2, y + 2, cellWidth - 4, cellHeight - 4);
        ctx2.fillStyle = "#000000";
        ctx2.font = "10px sans-serif";
        ctx2.textAlign = "left";
        ctx2.fillText(`Tree ${i}${i === 0 ? " (base)" : ""}`, x + 5, y + 12);
        const highlightPath = input ? this.trees[i].getPath(input) : void 0;
        this.trees[i].draw(ctx2, x + 5, y + 20, cellWidth - 10, cellHeight - 25, highlightPath);
      }
      const totalContentHeight = rows * cellHeight;
      if (totalContentHeight > availableHeight) {
        const scrollBarHeight = Math.max(20, availableHeight * (availableHeight / totalContentHeight));
        const scrollBarY = top + displayHeaderHeight + scrollY / (totalContentHeight - availableHeight) * (availableHeight - scrollBarHeight);
        ctx2.fillStyle = "rgba(0, 0, 0, 0.3)";
        ctx2.fillRect(left + width - 6, scrollBarY, 4, scrollBarHeight);
      }
      ctx2.restore();
    }
  };

  // index.js
  var canvas = document.getElementById("canvas");
  var ctx = canvas?.getContext("2d");
  if (!canvas || !ctx) {
    console.error("Canvas or context undefined");
    throw new Error("Canvas initialization failed");
  }
  var frame = 0;
  window.addEventListener("resize", resizeCanvas);
  window.onload = function() {
    document.querySelectorAll("select").forEach((sel) => {
      const opts = sel.options;
      let found = false;
      for (let i = 0; i < opts.length; i++) {
        if (opts[i].defaultSelected) {
          sel.selectedIndex = i;
          found = true;
          break;
        }
      }
      if (!found)
        sel.selectedIndex = 0;
    });
    document.querySelectorAll('input[type="range"]').forEach((el) => {
      el.value = el.defaultValue;
    });
    document.querySelectorAll('input[type="number"]').forEach((el) => {
      el.value = el.defaultValue;
    });
    document.querySelectorAll('input[type="checkbox"]').forEach((el) => {
      el.checked = el.defaultChecked;
    });
    start();
    resizeCanvas();
    setInterval(() => {
      update();
      frame++;
    }, 1);
  };
  var isStarted = false;
  var networkFormat = "Val1in1Out";
  var numCategories = 3;
  var showDataFormat = "output";
  var showNetworkFormat = "best";
  var testFunctionVal1in1out = "sine";
  var testFunctionVal2in1out = "wave";
  var testFunctionCat2in2out = "circle";
  var testFunctionCatNout = "sectors";
  var showTrainingData = "none";
  var trainingMethod = "genetic";
  var input1 = 0;
  var input2 = 0;
  var generationsPerDrawCycle = 1;
  var learningRate = 0.01;
  var momentum = 0.9;
  var geneticMutationWeights = 100;
  var geneticMutationWeightStrength = 0.01;
  var geneticMutationBiases = 100;
  var geneticMutationBiasStrength = 0.01;
  var xgbMaxTrees = Infinity;
  var xgbLimitTrees = false;
  var xgbShrinkage = 0.1;
  var xgbMaxDepth = 4;
  var xgbResolution = 0.02;
  var color1 = { r: 100, g: 150, b: 255 };
  var color2 = { r: 255, g: 100, b: 100 };
  function getCategoryColors(n) {
    const colors = [];
    for (let i = 0; i < n; i++) {
      const hue = i / n * 360;
      const s = 0.7, l = 0.55;
      const c = (1 - Math.abs(2 * l - 1)) * s;
      const x = c * (1 - Math.abs(hue / 60 % 2 - 1));
      const m = l - c / 2;
      let r1 = 0, g1 = 0, b1 = 0;
      if (hue < 60) {
        r1 = c;
        g1 = x;
      } else if (hue < 120) {
        r1 = x;
        g1 = c;
      } else if (hue < 180) {
        g1 = c;
        b1 = x;
      } else if (hue < 240) {
        g1 = x;
        b1 = c;
      } else if (hue < 300) {
        r1 = x;
        b1 = c;
      } else {
        r1 = c;
        b1 = x;
      }
      colors.push({
        r: Math.round((r1 + m) * 255),
        g: Math.round((g1 + m) * 255),
        b: Math.round((b1 + m) * 255)
      });
    }
    return colors;
  }
  function drawNClassGrid(ctx2, left, top, width, height, axis1low, axis2low, axis1high, axis2high, rows, columns, showText, getOutputs, colors) {
    const spaceX = width / columns;
    const spaceY = height / rows;
    ctx2.save();
    ctx2.fillStyle = "#e0e0e0";
    ctx2.fillRect(left, top, width, height);
    for (let i = 0; i < columns; i++) {
      for (let j = 0; j < rows; j++) {
        const x = left + i * spaceX;
        const y = top + j * spaceY;
        const in1 = axis1low + i * (axis1high - axis1low) / (columns - 1);
        const in2 = axis2low + j * (axis2high - axis2low) / (rows - 1);
        const outputs = getOutputs(in1, in2);
        let maxIdx = 0;
        for (let k = 1; k < outputs.length; k++) {
          if (outputs[k] > outputs[maxIdx])
            maxIdx = k;
        }
        const col = colors[maxIdx % colors.length];
        const confidence = Math.max(0, Math.min(1, outputs[maxIdx]));
        const r = Math.round(col.r * confidence + 240 * (1 - confidence));
        const g = Math.round(col.g * confidence + 240 * (1 - confidence));
        const b = Math.round(col.b * confidence + 240 * (1 - confidence));
        ctx2.fillStyle = `rgb(${r},${g},${b})`;
        ctx2.fillRect(x, y, spaceX + (showText ? -1 : 1), spaceY + (showText ? -1 : 1));
        if (showText) {
          ctx2.fillStyle = "#000";
          ctx2.font = `${Math.min(spaceX, spaceY) * 0.3}px sans-serif`;
          ctx2.textAlign = "center";
          ctx2.textBaseline = "middle";
          ctx2.fillText(maxIdx.toString(), x + spaceX / 2, y + spaceY / 2);
        }
      }
    }
    ctx2.restore();
  }
  function drawLineGraph(ctx2, left, top, width, height, xLow, xHigh, yLow, yHigh, lines, numPoints = 200, showAxes = true) {
    ctx2.save();
    ctx2.fillStyle = "#f8f8f8";
    ctx2.fillRect(left, top, width, height);
    const toScreenX = (x) => left + (x - xLow) / (xHigh - xLow) * width;
    const toScreenY = (y) => top + height - (y - yLow) / (yHigh - yLow) * height;
    if (showAxes) {
      ctx2.strokeStyle = "#ddd";
      ctx2.lineWidth = 0.5;
      for (let v = Math.ceil(xLow * 5) / 5; v <= xHigh; v += 0.2) {
        const sx = toScreenX(v);
        ctx2.beginPath();
        ctx2.moveTo(sx, top);
        ctx2.lineTo(sx, top + height);
        ctx2.stroke();
      }
      for (let v = Math.ceil(yLow * 5) / 5; v <= yHigh; v += 0.2) {
        const sy = toScreenY(v);
        ctx2.beginPath();
        ctx2.moveTo(left, sy);
        ctx2.lineTo(left + width, sy);
        ctx2.stroke();
      }
      ctx2.strokeStyle = "#999";
      ctx2.lineWidth = 1;
      if (yLow <= 0 && yHigh >= 0) {
        const y0 = toScreenY(0);
        ctx2.beginPath();
        ctx2.moveTo(left, y0);
        ctx2.lineTo(left + width, y0);
        ctx2.stroke();
      }
      if (xLow <= 0 && xHigh >= 0) {
        const x0 = toScreenX(0);
        ctx2.beginPath();
        ctx2.moveTo(x0, top);
        ctx2.lineTo(x0, top + height);
        ctx2.stroke();
      }
      ctx2.fillStyle = "#666";
      ctx2.font = "9px sans-serif";
      ctx2.textAlign = "center";
      ctx2.textBaseline = "top";
      for (let v = Math.ceil(xLow * 2) / 2; v <= xHigh; v += 0.5) {
        ctx2.fillText(v.toFixed(1), toScreenX(v), top + height + 1);
      }
      ctx2.textAlign = "right";
      ctx2.textBaseline = "middle";
      for (let v = Math.ceil(yLow * 2) / 2; v <= yHigh; v += 0.5) {
        ctx2.fillText(v.toFixed(1), left - 2, toScreenY(v));
      }
    }
    for (const line of lines) {
      ctx2.strokeStyle = line.color;
      ctx2.lineWidth = line.lineWidth ?? 2;
      ctx2.beginPath();
      for (let i = 0; i <= numPoints; i++) {
        const x = xLow + i / numPoints * (xHigh - xLow);
        const y = line.fn(x);
        const sx = toScreenX(x);
        const sy = toScreenY(Math.max(yLow, Math.min(yHigh, y)));
        if (i === 0)
          ctx2.moveTo(sx, sy);
        else
          ctx2.lineTo(sx, sy);
      }
      ctx2.stroke();
    }
    ctx2.restore();
  }
  var numOfNeuralNetworks = 16;
  var inputSize = 2;
  var outputSize = 1;
  var hiddenLayerSizes = [7, 10, 20, 20, 10, 7];
  var activationFunction = "relu";
  var outputActivationFunction = "tanh";
  var nnl = new NeuralNetworkList(numOfNeuralNetworks, inputSize, hiddenLayerSizes, outputSize, activationFunction, outputActivationFunction);
  var xgboost = null;
  var lastError = Infinity;
  var topPanelScroll = 0;
  var topPanelMaxScroll = 0;
  canvas.addEventListener("wheel", (e) => {
    const rect = canvas.getBoundingClientRect();
    const mouseY = e.clientY - rect.top;
    const hch = canvas.height / 2;
    if (mouseY < hch && topPanelMaxScroll > 0) {
      e.preventDefault();
      let delta = e.deltaY;
      if (e.deltaMode === 1)
        delta *= 30;
      else if (e.deltaMode === 2)
        delta *= canvas.height;
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
    showDataFormat = document.getElementById("showDataFormat").value;
    showNetworkFormat = document.getElementById("showNetworkFormat").value;
    showTrainingData = document.getElementById("showTrainingData").value;
    trainingMethod = document.getElementById("trainingMethod").value;
    const resEl = document.getElementById("xgbResolution");
    if (resEl)
      xgbResolution = parseFloat(resEl.value);
    networkChange();
    updateSettingsVisibility();
  }
  function update() {
    if (isStarted) {
      let bestFitness = 0;
      if (trainingMethod === "genetic") {
        if (nnl instanceof NeuralNetworkList) {
          for (let i = 0; i < generationsPerDrawCycle; i++) {
            bestFitness = nnl.runGeneration(geneticMutationWeights, geneticMutationWeightStrength, geneticMutationBiases, geneticMutationBiasStrength);
            if (bestFitness === lastError) {
              geneticMutationBiasStrength = Math.max(1e-4, geneticMutationBiasStrength / 1.001);
              geneticMutationWeightStrength = Math.max(1e-4, geneticMutationWeightStrength / 1.001);
            } else {
              geneticMutationBiasStrength = Math.min(1, geneticMutationBiasStrength * 1.001);
              geneticMutationWeightStrength = Math.min(1, geneticMutationWeightStrength * 1.001);
            }
          }
        }
      } else if (trainingMethod === "backprop") {
        if (nnl instanceof NeuralNetworkList) {
          nnl.setLearningRate(learningRate);
          nnl.setMomentum(momentum);
          for (let i = 0; i < generationsPerDrawCycle; i++) {
            bestFitness = nnl.trainBackpropagation(1);
          }
        }
      } else if (trainingMethod === "XGBoost") {
        if (xgboost && nnl instanceof NeuralNetworkList) {
          for (let i = 0; i < generationsPerDrawCycle; i++) {
            xgboost.train(xgbTrainInputs, xgbTrainOutputs, testInputsGrid, test);
          }
          bestFitness = xgboost.trainRMSE;
        }
      }
      if (trainingMethod !== "XGBoost" && nnl instanceof NeuralNetworkList) {
        nnl.computeTestErr((inputs) => {
          const result = nnl instanceof NeuralNetworkList ? nnl.neuralNetworks[0].run(inputs) : { neurons: [{ value: 0 }] };
          return result.neurons.map((n) => n.value);
        });
      }
      lastError = bestFitness;
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const padding = 1e-3;
    const hcw = canvas.width / 2;
    const hch = canvas.height / 2;
    ctx.fillStyle = "#000000";
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
    if (trainingMethod === "XGBoost" && xgboost) {
      xgboost.drawHeader(ctx, 0, 0, canvas.width, displayHeaderHeight);
    } else if (nnl instanceof NeuralNetworkList) {
      nnl.drawHeader(ctx, 0, 0, canvas.width, displayHeaderHeight);
    }
    switch (showNetworkFormat) {
      case "all":
        if (trainingMethod === "XGBoost" && xgboost) {
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
    const decimalsV = 2;
    const rowsV = 11;
    const columnsV = 11;
    const predict = (inputs) => {
      if (trainingMethod === "XGBoost" && xgboost) {
        return xgboost.predict(inputs);
      } else if (nnl instanceof NeuralNetworkList) {
        const result = nnl.neuralNetworks[0].run(inputs);
        return result.neurons.map((n) => n.value);
      }
      return [0];
    };
    if (nnl instanceof NeuralNetworkList) {
      const displayNetwork = nnl.neuralNetworks[0];
      const isXGB = trainingMethod === "XGBoost" && xgboost;
      if (networkFormat === "Val1in1Out") {
        const graphMargin = 15;
        const graphLeft = graphMargin;
        const graphTop = hch + 5;
        const graphWidth = canvas.width - graphMargin * 2;
        const graphHeight = hch - 10;
        const predictLine = (x) => predict([x])[0];
        const testLine = (x) => test([x])[0];
        const errorLine = (x) => predictLine(x) - testLine(x);
        switch (showDataFormat) {
          case "none":
            drawLineGraph(ctx, graphLeft, graphTop, graphWidth, graphHeight, axis1low, axis1high, -1.2, 1.2, []);
            break;
          case "testvsoutput":
            drawLineGraph(ctx, graphLeft, graphTop, graphWidth, graphHeight, axis1low, axis1high, -1.2, 1.2, [
              { fn: testLine, color: "#4CAF50", lineWidth: 1.5 },
              { fn: predictLine, color: "#2196F3", lineWidth: 2 }
            ]);
            break;
          case "output":
            drawLineGraph(ctx, graphLeft, graphTop, graphWidth, graphHeight, axis1low, axis1high, -1.2, 1.2, [
              { fn: predictLine, color: "#2196F3", lineWidth: 2 }
            ]);
            break;
          case "test":
            drawLineGraph(ctx, graphLeft, graphTop, graphWidth, graphHeight, axis1low, axis1high, -1.2, 1.2, [
              { fn: testLine, color: "#4CAF50", lineWidth: 2 }
            ]);
            break;
          case "error":
            drawLineGraph(ctx, graphLeft, graphTop, graphWidth, graphHeight, axis1low, axis1high, -1.2, 1.2, [
              { fn: errorLine, color: "#E53935", lineWidth: 2 }
            ]);
            break;
        }
        if (showTrainingData !== "none" && nnl instanceof NeuralNetworkList) {
          const toSX = (v) => graphLeft + (v - axis1low) / (axis1high - axis1low) * graphWidth;
          const toSY = (v) => graphTop + graphHeight - (v - -1.2) / (1.2 - -1.2) * graphHeight;
          let dotFn = null;
          let dotColor = "#000";
          if (showTrainingData === "output") {
            dotFn = (inp) => predict(inp)[0];
            dotColor = "#2196F3";
          } else if (showTrainingData === "test") {
            dotFn = (inp) => test(inp)[0];
            dotColor = "#4CAF50";
          } else if (showTrainingData === "error") {
            dotFn = (inp) => predict(inp)[0] - test(inp)[0];
            dotColor = "#E53935";
          }
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
        const markerX = graphLeft + (input1 - axis1low) / (axis1high - axis1low) * graphWidth;
        ctx.strokeStyle = "rgba(0,0,0,0.4)";
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(markerX, graphTop);
        ctx.lineTo(markerX, graphTop + graphHeight);
        ctx.stroke();
        ctx.setLineDash([]);
      } else if (networkFormat === "Val2in1out") {
        switch (showDataFormat) {
          case "none":
            ctx.fillRect(0, hch, hcw - padding * canvas.width, hch);
            if (isXGB) {
              displayNetwork.displayGrid2Input1Output(ctx, hcw + padding * canvas.width, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, rowsV, columnsV, decimalsV, true, true, (in1, in2) => predict([in1, in2])[0], (val) => val * outputRange - ouputMiddle);
            } else {
              displayNetwork.display2Input1Output(ctx, hcw + padding * canvas.width, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, ouputMiddle, outputRange, rowsV, columnsV, decimalsV, true, true);
            }
            break;
          case "error":
            let errorRange = 1;
            if (isXGB) {
              const errorFn = (in1, in2) => predict([in1, in2])[0] - test([in1, in2])[0];
              displayNetwork.displayGrid2Input1Output(ctx, 0, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, rows, columns, decimals, false, false, errorFn, (val) => val * errorRange);
              displayNetwork.displayGrid2Input1Output(ctx, hcw + padding * canvas.width, hch, hcw, hch, axis1low, axis2low, axis1high, axis2high, rowsV, columnsV, decimalsV, true, true, errorFn, (val) => val * errorRange);
            } else {
              displayNetwork.display2Input1OutputError(ctx, 0, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, errorRange, rows, columns, decimals, false, false, test);
              displayNetwork.display2Input1OutputError(ctx, hcw + padding * canvas.width, hch, hcw, hch, axis1low, axis2low, axis1high, axis2high, errorRange, rowsV, columnsV, decimalsV, true, true, test);
            }
            break;
          case "output":
            if (isXGB) {
              const outputFn = (in1, in2) => predict([in1, in2])[0];
              const colorFn = (val) => val * outputRange - ouputMiddle;
              displayNetwork.displayGrid2Input1Output(ctx, 0, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, rows, columns, decimals, false, false, outputFn, colorFn);
              displayNetwork.displayGrid2Input1Output(ctx, hcw + padding * canvas.width, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, rowsV, columnsV, decimalsV, true, true, outputFn, colorFn);
            } else {
              displayNetwork.display2Input1Output(ctx, 0, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, ouputMiddle, outputRange, rows, columns, decimals, false, false);
              displayNetwork.display2Input1Output(ctx, hcw + padding * canvas.width, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, ouputMiddle, outputRange, rowsV, columnsV, decimalsV, true, true);
            }
            break;
          case "test":
            displayNetwork.display2Input1OutputTest(ctx, 0, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, ouputMiddle, outputRange, rows, columns, decimals, false, false, test);
            displayNetwork.display2Input1OutputTest(ctx, hcw + padding * canvas.width, hch, hcw, hch, axis1low, axis2low, axis1high, axis2high, ouputMiddle, outputRange, rowsV, columnsV, decimalsV, true, true, test);
            break;
        }
      } else if (networkFormat === "Cat2in2out") {
        switch (showDataFormat) {
          case "none":
            ctx.fillRect(0, hch, hcw - padding * canvas.width, hch);
            if (isXGB) {
              displayNetwork.displayGrid2Input2Output(ctx, hcw + padding * canvas.width, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, rowsV, columnsV, decimalsV, true, true, color1, color2, (in1, in2) => predict([in1, in2]));
            } else {
              displayNetwork.display2Input2Output(ctx, hcw + padding * canvas.width, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, rowsV, columnsV, decimalsV, true, true, color1, color2);
            }
            break;
          case "error":
            if (isXGB) {
              const errorColor1 = { r: 255, g: 255, b: 255 };
              const errorColor2 = { r: 255, g: 0, b: 0 };
              const errorFn = (in1, in2) => {
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
              const outputFn = (in1, in2) => predict([in1, in2]);
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
      } else if (networkFormat === "CatNout") {
        const catColors = getCategoryColors(numCategories);
        const outputFn = (in1, in2) => predict([in1, in2]);
        const testFn = (in1, in2) => test([in1, in2]);
        switch (showDataFormat) {
          case "none":
            ctx.fillRect(0, hch, hcw - padding * canvas.width, hch);
            drawNClassGrid(ctx, hcw + padding * canvas.width, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, rowsV, columnsV, true, outputFn, catColors);
            break;
          case "output":
            drawNClassGrid(ctx, 0, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, rows, columns, false, outputFn, catColors);
            drawNClassGrid(ctx, hcw + padding * canvas.width, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, rowsV, columnsV, true, outputFn, catColors);
            break;
          case "test":
            drawNClassGrid(ctx, 0, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, rows, columns, false, testFn, catColors);
            drawNClassGrid(ctx, hcw + padding * canvas.width, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, rowsV, columnsV, true, testFn, catColors);
            break;
          case "error": {
            const errFn = (in1, in2) => {
              const pred = predict([in1, in2]);
              const expected = test([in1, in2]);
              let sum = 0;
              for (let k = 0; k < pred.length; k++)
                sum += Math.abs(pred[k] - expected[k]);
              const err = sum / pred.length;
              return [err];
            };
            const errColors = [{ r: 255, g: 0, b: 0 }];
            drawNClassGrid(ctx, 0, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, rows, columns, false, errFn, errColors);
            drawNClassGrid(ctx, hcw + padding * canvas.width, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, rowsV, columnsV, true, errFn, errColors);
            break;
          }
        }
      }
    }
    if (nnl instanceof NeuralNetworkList && showTrainingData !== "none" && inputSize >= 2) {
      const dpLeft = 0, dpTop = hch, dpWidth = hcw - padding * canvas.width, dpHeight = hch;
      if (networkFormat === "Val2in1out") {
        switch (showTrainingData) {
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
      } else if (networkFormat === "Cat2in2out") {
        switch (showTrainingData) {
          case "error":
            nnl.display2Input2OutputDataPoints((inputs) => {
              let p = predict(inputs), t = test(inputs);
              return [(Math.abs(p[0] - t[0]) + Math.abs(p[1] - t[1])) / 2, 0];
            }, ctx, dpLeft, dpTop, dpWidth, dpHeight, axis1low, axis2low, axis1high, axis2high, { r: 255, g: 255, b: 255 }, { r: 255, g: 0, b: 0 });
            break;
          case "output":
            nnl.display2Input2OutputDataPoints((inputs) => predict(inputs), ctx, dpLeft, dpTop, dpWidth, dpHeight, axis1low, axis2low, axis1high, axis2high, color1, color2);
            break;
          case "test":
            nnl.display2Input2OutputDataPoints(test, ctx, dpLeft, dpTop, dpWidth, dpHeight, axis1low, axis2low, axis1high, axis2high, color1, color2);
            break;
        }
      } else if (networkFormat === "CatNout") {
        const catColors = getCategoryColors(numCategories);
        ctx.save();
        ctx.beginPath();
        ctx.rect(dpLeft, dpTop, dpWidth, dpHeight);
        ctx.clip();
        let dotFn = test;
        if (showTrainingData === "output")
          dotFn = (inputs) => predict(inputs);
        else if (showTrainingData === "error")
          dotFn = (inputs) => {
            let p = predict(inputs), t = test(inputs);
            let sum = 0;
            for (let k = 0; k < p.length; k++)
              sum += Math.abs(p[k] - t[k]);
            return [sum / p.length];
          };
        nnl.trialInputsList.forEach((input) => {
          const sx = dpLeft + (input[0] - axis1low) / (axis1high - axis1low) * dpWidth;
          const sy = dpTop + (input[1] - axis2low) / (axis2high - axis2low) * dpHeight;
          const out = dotFn(input);
          let maxIdx = 0;
          for (let k = 1; k < out.length; k++)
            if (out[k] > out[maxIdx])
              maxIdx = k;
          const col = catColors[maxIdx % catColors.length];
          ctx.fillStyle = "#000";
          ctx.beginPath();
          ctx.arc(sx, sy, 4, 0, 2 * Math.PI);
          ctx.fill();
          ctx.fillStyle = `rgb(${col.r},${col.g},${col.b})`;
          ctx.beginPath();
          ctx.arc(sx, sy, 3, 0, 2 * Math.PI);
          ctx.fill();
        });
        ctx.restore();
      }
    }
  }
  var testInputsGrid = [];
  var xgbTrainInputs = [];
  var xgbTrainOutputs = [];
  function createTrials() {
    const inputs = createInputs(inputSize, 1, -1, 0.1);
    testInputsGrid = createInputs(inputSize, 0.95, -0.95, 0.1);
    if (nnl instanceof NeuralNetworkList) {
      nnl.createTrials(inputs, test);
      nnl.testInputs = testInputsGrid;
      nnl.testFn = test;
    }
    xgbTrainInputs = createInputs(inputSize, 1, -1, xgbResolution);
    xgbTrainOutputs = xgbTrainInputs.map((inp) => test(inp));
  }
  function createInputs(numOfInputs, high, low, spacing) {
    const possibleValues = [];
    for (let v = low; v <= high; v += spacing) {
      possibleValues.push(v);
    }
    const result = [];
    function generateCombinations(current) {
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
  function test(inputs) {
    switch (networkFormat) {
      case "Val1in1Out": {
        const x = inputs[0];
        switch (testFunctionVal1in1out) {
          case "sine":
            return [Math.sin(x * Math.PI)];
          case "square":
            return [Math.sin(x * Math.PI * 2) > 0 ? 1 : -1];
          case "sawtooth":
            return [(x + 1) % 0.5 * 4 - 1];
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
        switch (testFunctionVal2in1out) {
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
            return [Math.max(-1, Math.min(1, Math.exp(-((x - 0.5) ** 2 + y ** 2) * 5) - Math.exp(-((x + 0.5) ** 2 + y ** 2) * 5)))];
          case "step2d":
            return [x > 0.3 ? 1 : x < -0.3 ? -1 : y > 0 ? 0.5 : -0.5];
          case "swiss":
            return [Math.sin(x * 3 + y * 2) * Math.cos(x * 2 - y * 3) > 0 ? 1 : -1];
        }
        break;
      }
      case "Cat2in2out": {
        const x = inputs[0], y = inputs[1];
        switch (testFunctionCat2in2out) {
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
            return x > 0 !== y > 0 ? [1, 0] : [0, 1];
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
            return x ** 2 + (y - 0.3) ** 2 < 0.6 && x ** 2 + (y + 0.3) ** 2 > 0.3 ? [1, 0] : [0, 1];
          case "diamond":
            return Math.abs(x) + Math.abs(y) < 0.7 ? [1, 0] : [0, 1];
          case "cross":
            return Math.abs(x) < 0.2 || Math.abs(y) < 0.2 ? [1, 0] : [0, 1];
        }
        break;
      }
      case "CatNout": {
        const x = inputs[0], y = inputs[1];
        const n = numCategories;
        const oneHot = (idx) => {
          const arr = new Array(n).fill(0);
          arr[Math.min(Math.max(idx, 0), n - 1)] = 1;
          return arr;
        };
        switch (testFunctionCatNout) {
          case "sectors": {
            let a = Math.atan2(y, x);
            return oneHot(Math.floor((a + Math.PI) / (2 * Math.PI) * n));
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
            return oneHot(Math.floor((a + Math.PI + r * 4) % (2 * Math.PI) / (2 * Math.PI) * n));
          }
          case "stripes":
            return oneHot(Math.floor((x + 1) / 2 * n));
          case "checkerboard": {
            let s = Math.ceil(Math.sqrt(n));
            let cx = Math.floor((x + 1) / 2 * s);
            let cy = Math.floor((y + 1) / 2 * s);
            return oneHot((cx + cy) % n);
          }
          case "voronoi": {
            const centers = Array.from({ length: n }, (_, i) => ({
              x: Math.cos(i * 2.39996) * 0.6,
              y: Math.sin(i * 2.39996) * 0.6
            }));
            let minDist = Infinity, minIdx = 0;
            for (let i = 0; i < n; i++) {
              let d = (x - centers[i].x) ** 2 + (y - centers[i].y) ** 2;
              if (d < minDist) {
                minDist = d;
                minIdx = i;
              }
            }
            return oneHot(minIdx);
          }
          case "waves":
            return oneHot(Math.floor((Math.sin(x * 4) + Math.sin(y * 4) + 2) / 4 * n));
        }
        break;
      }
    }
    return new Array(outputSize).fill(0);
  }
  function startToggle() {
    const startButton = document.getElementById("startButton");
    const startToggleText = document.getElementById("startToggleText");
    if (isStarted) {
      startToggleText.innerHTML = "Start";
    } else {
      startToggleText.innerHTML = "Stop";
    }
    isStarted = !isStarted;
  }
  window.startToggle = startToggle;
  function networkChange() {
    networkFormat = document.getElementById("networkFormat").value;
    const testFunctionDropdown = document.getElementById("testFunction");
    testFunctionDropdown.innerHTML = "";
    switch (networkFormat) {
      case "Val1in1Out":
        inputSize = 1;
        outputSize = 1;
        hiddenLayerSizes = [7, 10, 20, 20, 10, 7];
        activationFunction = "relu";
        outputActivationFunction = "tanh";
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
      case "Val2in1out":
        inputSize = 2;
        outputSize = 1;
        hiddenLayerSizes = [7, 10, 20, 20, 10, 7];
        activationFunction = "relu";
        outputActivationFunction = "tanh";
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
      case "Cat2in2out":
        inputSize = 2;
        outputSize = 2;
        hiddenLayerSizes = [7, 10, 20, 20, 10, 7];
        activationFunction = "relu";
        outputActivationFunction = "sigmoid";
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
      case "CatNout":
        inputSize = 2;
        outputSize = numCategories;
        hiddenLayerSizes = [10, 20, 20, 10];
        activationFunction = "relu";
        outputActivationFunction = "sigmoid";
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
    document.getElementById("numCategoriesGroup").style.display = networkFormat === "CatNout" ? "" : "none";
    document.getElementById("input2Group").style.display = inputSize >= 2 ? "" : "none";
    const dataFormatDropdown = document.getElementById("showDataFormat");
    if (networkFormat === "Val1in1Out") {
      dataFormatDropdown.innerHTML = `
            <option value="testvsoutput">Test vs Output</option>
            <option value="output">Output</option>
            <option value="test">Test</option>
            <option value="error">Error</option>
            <option value="none">None</option>
        `;
      showDataFormat = "testvsoutput";
      dataFormatDropdown.value = "testvsoutput";
    } else {
      dataFormatDropdown.innerHTML = `
            <option value="none">None</option>
            <option value="output" selected>Output</option>
            <option value="error">Error</option>
            <option value="test">Test</option>
        `;
      if (showDataFormat === "testvsoutput")
        showDataFormat = "output";
      dataFormatDropdown.value = showDataFormat;
    }
    if (trainingMethod === "backprop") {
      numOfNeuralNetworks = 1;
    } else if (trainingMethod === "XGBoost") {
      numOfNeuralNetworks = 1;
    } else {
      numOfNeuralNetworks = 16;
    }
    nnl = new NeuralNetworkList(numOfNeuralNetworks, inputSize, hiddenLayerSizes, outputSize, activationFunction, outputActivationFunction);
    if (trainingMethod === "XGBoost") {
      xgboost = new XGBoostEnsemble(inputSize, outputSize, xgbShrinkage, xgbMaxDepth, xgbMaxTrees, networkFormat !== "Val2in1out" && networkFormat !== "Val1in1Out");
    }
    createTrials();
  }
  window.networkChange = networkChange;
  function showDataChange() {
    showDataFormat = document.getElementById("showDataFormat").value;
  }
  window.showDataChange = showDataChange;
  function showNetworkChange() {
    showNetworkFormat = document.getElementById("showNetworkFormat").value;
  }
  window.showNetworkChange = showNetworkChange;
  function testFunctionChange() {
    const value = document.getElementById("testFunction").value;
    if (networkFormat === "Val1in1Out") {
      testFunctionVal1in1out = value;
    } else if (networkFormat === "Val2in1out") {
      testFunctionVal2in1out = value;
    } else if (networkFormat === "Cat2in2out") {
      testFunctionCat2in2out = value;
    } else if (networkFormat === "CatNout") {
      testFunctionCatNout = value;
    }
    createTrials();
  }
  window.testFunctionChange = testFunctionChange;
  function numCategoriesChange() {
    const slider = document.getElementById("numCategoriesSlider");
    const input = document.getElementById("numCategoriesInput");
    const display = document.getElementById("numCategoriesDisplay");
    if (document.activeElement === slider) {
      input.value = slider.value;
    } else {
      slider.value = input.value;
    }
    numCategories = parseInt(slider.value);
    display.textContent = numCategories.toString();
    if (networkFormat === "CatNout") {
      networkChange();
    }
  }
  window.numCategoriesChange = numCategoriesChange;
  function inputChange() {
    generationsPerDrawCycle = parseFloat(document.getElementById("generationsPerDrawCycle").value);
    learningRate = parseFloat(document.getElementById("learningRate").value);
    momentum = parseFloat(document.getElementById("momentum").value);
    let i1 = parseFloat(document.getElementById("input1").value);
    input1 = Number(isNaN(i1) ? 0 : i1);
    let i2 = parseFloat(document.getElementById("input2").value);
    input2 = Number(isNaN(i2) ? 0 : i2);
    document.getElementById("input1Slider").value = input1.toString();
    document.getElementById("input2Slider").value = input2.toString();
    document.getElementById("generationsPerDrawCycleSlider").value = generationsPerDrawCycle.toString();
    document.getElementById("learningRateSlider").value = learningRate.toString();
    document.getElementById("momentumSlider").value = momentum.toString();
    document.getElementById("learningRateDisplay").textContent = learningRate.toFixed(3);
    document.getElementById("momentumDisplay").textContent = momentum.toFixed(2);
  }
  window.inputChange = inputChange;
  function inputSliderChange() {
    generationsPerDrawCycle = parseFloat(document.getElementById("generationsPerDrawCycleSlider").value);
    learningRate = parseFloat(document.getElementById("learningRateSlider").value);
    momentum = parseFloat(document.getElementById("momentumSlider").value);
    input1 = parseFloat(document.getElementById("input1Slider").value);
    input2 = parseFloat(document.getElementById("input2Slider").value);
    document.getElementById("input1").value = input1.toString();
    document.getElementById("input2").value = input2.toString();
    document.getElementById("generationsPerDrawCycle").value = generationsPerDrawCycle.toString();
    document.getElementById("learningRate").value = learningRate.toString();
    document.getElementById("momentum").value = momentum.toString();
    document.getElementById("learningRateDisplay").textContent = learningRate.toFixed(3);
    document.getElementById("momentumDisplay").textContent = momentum.toFixed(2);
  }
  window.inputSliderChange = inputSliderChange;
  function reset() {
    topPanelScroll = 0;
    if (trainingMethod === "XGBoost") {
      nnl = new NeuralNetworkList(1, inputSize, hiddenLayerSizes, outputSize, activationFunction, outputActivationFunction);
      xgboost = new XGBoostEnsemble(inputSize, outputSize, xgbShrinkage, xgbMaxDepth, xgbMaxTrees, networkFormat !== "Val2in1out" && networkFormat !== "Val1in1Out");
    } else {
      nnl = new NeuralNetworkList(numOfNeuralNetworks, inputSize, hiddenLayerSizes, outputSize, activationFunction, outputActivationFunction);
      xgboost = null;
    }
    createTrials();
  }
  window.reset = reset;
  function showTrainingDataChange() {
    showTrainingData = document.getElementById("showTrainingData").value;
  }
  window.showTrainingDataChange = showTrainingDataChange;
  function trainingMethodChange() {
    const newTrainingMethod = document.getElementById("trainingMethod").value;
    if (newTrainingMethod === "XGBoost" && trainingMethod !== "XGBoost") {
      nnl = new NeuralNetworkList(1, inputSize, hiddenLayerSizes, outputSize, activationFunction, outputActivationFunction);
      xgboost = new XGBoostEnsemble(inputSize, outputSize, xgbShrinkage, xgbMaxDepth, xgbMaxTrees, networkFormat !== "Val2in1out" && networkFormat !== "Val1in1Out");
      createTrials();
      numOfNeuralNetworks = 1;
    } else if (newTrainingMethod !== "XGBoost" && trainingMethod === "XGBoost") {
      const newRequired = newTrainingMethod === "genetic" ? 16 : 1;
      nnl = new NeuralNetworkList(newRequired, inputSize, hiddenLayerSizes, outputSize, activationFunction, outputActivationFunction);
      xgboost = null;
      createTrials();
      numOfNeuralNetworks = newRequired;
    } else if (nnl instanceof NeuralNetworkList) {
      const getRequiredNetworks = (method) => {
        if (method === "genetic")
          return 16;
        if (method === "backprop")
          return 1;
        return 16;
      };
      const oldRequired = getRequiredNetworks(trainingMethod);
      const newRequired = getRequiredNetworks(newTrainingMethod);
      if (newRequired !== oldRequired) {
        if (newRequired > oldRequired) {
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
  window.trainingMethodChange = trainingMethodChange;
  function updateSettingsVisibility() {
    const backpropSettings = document.getElementById("backpropSettings");
    const xgboostSettings = document.getElementById("xgboostSettings");
    const viewDropdown = document.getElementById("showNetworkFormat");
    const viewLabel = viewDropdown.parentElement?.querySelector("label");
    if (trainingMethod === "XGBoost") {
      backpropSettings.style.display = "none";
      xgboostSettings.style.display = "";
      if (viewLabel)
        viewLabel.textContent = "Tree View";
      viewDropdown.innerHTML = `
            <option value="base">Base Tree</option>
            <option value="latest">Latest Tree</option>
            <option value="all">All Trees</option>
        `;
      viewDropdown.value = "base";
      showNetworkFormat = "base";
    } else {
      backpropSettings.style.display = trainingMethod === "backprop" ? "" : "none";
      xgboostSettings.style.display = "none";
      if (viewLabel)
        viewLabel.textContent = "Network View";
      viewDropdown.innerHTML = `
            <option value="best">Best</option>
            <option value="all">All</option>
        `;
      viewDropdown.value = showNetworkFormat === "all" ? "all" : "best";
      if (showNetworkFormat !== "all")
        showNetworkFormat = "best";
    }
    topPanelScroll = 0;
  }
  function limitTreesToggleChange() {
    xgbLimitTrees = document.getElementById("limitTreesToggle").checked;
    const sliderGroup = document.getElementById("maxTreesSliderGroup");
    const display = document.getElementById("maxTreesDisplay");
    if (xgbLimitTrees) {
      sliderGroup.style.display = "";
      xgbMaxTrees = parseInt(document.getElementById("maxTreesSlider").value);
      display.textContent = xgbMaxTrees.toString();
    } else {
      sliderGroup.style.display = "none";
      xgbMaxTrees = Infinity;
      display.textContent = "\u221E";
    }
    if (xgboost)
      xgboost.maxTrees = xgbMaxTrees;
  }
  window.limitTreesToggleChange = limitTreesToggleChange;
  function xgboostSliderChange() {
    if (xgbLimitTrees) {
      xgbMaxTrees = parseInt(document.getElementById("maxTreesSlider").value);
      document.getElementById("maxTrees").value = xgbMaxTrees.toString();
      document.getElementById("maxTreesDisplay").textContent = xgbMaxTrees.toString();
    }
    xgbShrinkage = parseFloat(document.getElementById("shrinkageSlider").value);
    xgbMaxDepth = parseInt(document.getElementById("maxDepthSlider").value);
    document.getElementById("shrinkage").value = xgbShrinkage.toString();
    document.getElementById("maxDepth").value = xgbMaxDepth.toString();
    document.getElementById("shrinkageDisplay").textContent = xgbShrinkage.toFixed(2);
    document.getElementById("maxDepthDisplay").textContent = xgbMaxDepth.toString();
    if (xgboost) {
      xgboost.maxTrees = xgbMaxTrees;
      xgboost.shrinkage = xgbShrinkage;
      xgboost.maxDepth = xgbMaxDepth;
    }
  }
  window.xgboostSliderChange = xgboostSliderChange;
  function xgboostInputChange() {
    if (xgbLimitTrees) {
      xgbMaxTrees = parseInt(document.getElementById("maxTrees").value);
      document.getElementById("maxTreesSlider").value = xgbMaxTrees.toString();
      document.getElementById("maxTreesDisplay").textContent = xgbMaxTrees.toString();
    }
    xgbShrinkage = parseFloat(document.getElementById("shrinkage").value);
    xgbMaxDepth = parseInt(document.getElementById("maxDepth").value);
    document.getElementById("shrinkageSlider").value = xgbShrinkage.toString();
    document.getElementById("maxDepthSlider").value = xgbMaxDepth.toString();
    document.getElementById("shrinkageDisplay").textContent = xgbShrinkage.toFixed(2);
    document.getElementById("maxDepthDisplay").textContent = xgbMaxDepth.toString();
    if (xgboost) {
      xgboost.maxTrees = xgbMaxTrees;
      xgboost.shrinkage = xgbShrinkage;
      xgboost.maxDepth = xgbMaxDepth;
    }
  }
  window.xgboostInputChange = xgboostInputChange;
  function xgbResolutionChange() {
    xgbResolution = parseFloat(document.getElementById("xgbResolution").value);
    document.getElementById("xgbResolutionDisplay").textContent = xgbResolution.toString();
    xgbTrainInputs = createInputs(inputSize, 1, -1, xgbResolution);
    xgbTrainOutputs = xgbTrainInputs.map((inp) => test(inp));
    if (xgboost) {
      xgboost = new XGBoostEnsemble(inputSize, outputSize, xgbShrinkage, xgbMaxDepth, xgbMaxTrees, networkFormat !== "Val2in1out" && networkFormat !== "Val1in1Out");
    }
  }
  window.xgbResolutionChange = xgbResolutionChange;
})();
