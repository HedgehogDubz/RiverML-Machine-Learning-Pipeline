// Neural network engine for the desktop visualizer.
// Flat vectors per layer for speed. Seed format is shared with the TypeScript version.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

enum class Activation { ReLU, Sigmoid, Tanh };

inline const char* activationName(Activation a) {
    switch (a) {
        case Activation::ReLU: return "relu";
        case Activation::Sigmoid: return "sigmoid";
        default: return "tanh";
    }
}

inline bool activationFromName(const std::string& s, Activation& out) {
    if (s == "relu") { out = Activation::ReLU; return true; }
    if (s == "sigmoid") { out = Activation::Sigmoid; return true; }
    if (s == "tanh") { out = Activation::Tanh; return true; }
    return false;
}

inline double randUnit() {
    static std::mt19937 rng(std::random_device{}());
    static std::uniform_real_distribution<double> dist(-1.0, 1.0);
    return dist(rng);
}

inline int randInt(int n) {
    return (int)(((randUnit() + 1.0) * 0.5) * n) % n;
}

// One layer: values plus per neuron bias and a row of weights to the previous layer
struct Layer {
    int size = 0;
    int prevSize = 0;
    std::vector<double> values;
    std::vector<double> biases;
    std::vector<double> weights;     // size * prevSize
    // backprop state
    std::vector<double> grads;
    std::vector<double> biasVel;
    std::vector<double> weightGrads;
    std::vector<double> weightVel;
};

struct NeuralNetwork {
    std::vector<int> sizes;
    Activation act = Activation::ReLU;
    Activation outAct = Activation::Tanh;
    std::vector<Layer> layers;
    double error = 0;
    double meanError = 0;

    NeuralNetwork() {}
    NeuralNetwork(int inputSize, const std::vector<int>& hidden, int outputSize,
                  Activation activation, Activation outputActivation) {
        act = activation;
        outAct = outputActivation;
        sizes.push_back(inputSize);
        for (int h : hidden) sizes.push_back(h);
        sizes.push_back(outputSize);
        layers.resize(sizes.size());
        for (size_t l = 0; l < sizes.size(); l++) {
            Layer& L = layers[l];
            L.size = sizes[l];
            L.prevSize = l == 0 ? 0 : sizes[l - 1];
            L.values.assign(L.size, 0);
            if (l == 0) continue;
            L.biases.resize(L.size);
            L.weights.resize((size_t)L.size * L.prevSize);
            for (auto& b : L.biases) b = randUnit();
            for (auto& w : L.weights) w = randUnit();
            L.grads.assign(L.size, 0);
            L.biasVel.assign(L.size, 0);
            L.weightGrads.assign(L.weights.size(), 0);
            L.weightVel.assign(L.weights.size(), 0);
        }
    }

    int inputSize() const { return sizes.front(); }
    int outputSize() const { return sizes.back(); }
    int numLayers() const { return (int)sizes.size(); }

    static double applyActivation(Activation a, double x) {
        switch (a) {
            case Activation::ReLU: return x > 0 ? x : 0;
            case Activation::Sigmoid: return 1.0 / (1.0 + std::exp(-x));
            default: return std::tanh(x);
        }
    }

    static double activationDerivative(Activation a, double value) {
        // derivative expressed with the activated value
        switch (a) {
            case Activation::ReLU: return value > 0 ? 1.0 : 0.0;
            case Activation::Sigmoid: return value * (1.0 - value);
            default: return 1.0 - value * value;
        }
    }

    Activation layerActivation(int l) const {
        return l == numLayers() - 1 ? outAct : act;
    }

    const std::vector<double>& run(const std::vector<double>& inputs) {
        for (int i = 0; i < sizes[0]; i++) layers[0].values[i] = inputs[i];
        for (int l = 1; l < numLayers(); l++) {
            Layer& L = layers[l];
            const std::vector<double>& prev = layers[l - 1].values;
            Activation a = layerActivation(l);
            for (int n = 0; n < L.size; n++) {
                double sum = L.biases[n];
                const double* w = &L.weights[(size_t)n * L.prevSize];
                for (int p = 0; p < L.prevSize; p++) sum += w[p] * prev[p];
                L.values[n] = applyActivation(a, sum);
            }
        }
        return layers.back().values;
    }

    void mutate(int numWeights, double weightStrength, int numBiases, double biasStrength) {
        for (int i = 0; i < numWeights; i++) {
            int l = 1 + randInt(numLayers() - 1);
            Layer& L = layers[l];
            L.weights[randInt((int)L.weights.size())] += randUnit() * weightStrength;
        }
        for (int i = 0; i < numBiases; i++) {
            int l = 1 + randInt(numLayers() - 1);
            Layer& L = layers[l];
            L.biases[randInt(L.size)] += randUnit() * biasStrength;
        }
    }

    void addError(const std::vector<double>& targets, double power) {
        double sum = 0, meanSum = 0;
        const std::vector<double>& out = layers.back().values;
        for (int i = 0; i < outputSize(); i++) {
            double diff = std::fabs(out[i] - targets[i]);
            sum += std::pow(diff, power);
            meanSum += diff;
        }
        error += sum;
        meanError += meanSum;
    }

    // One batch of backprop with momentum
    void trainBatch(const std::vector<std::vector<double>>& inputs,
                    const std::vector<std::vector<double>>& targets,
                    double learningRate, double momentum) {
        for (int l = 1; l < numLayers(); l++) {
            Layer& L = layers[l];
            std::fill(L.grads.begin(), L.grads.end(), 0.0);
            std::fill(L.weightGrads.begin(), L.weightGrads.end(), 0.0);
        }
        std::vector<double> biasGradSum;
        std::vector<std::vector<double>> biasGrads(numLayers());
        for (int l = 1; l < numLayers(); l++) biasGrads[l].assign(sizes[l], 0.0);

        for (size_t s = 0; s < inputs.size(); s++) {
            run(inputs[s]);

            // output layer gradient
            {
                Layer& L = layers.back();
                Activation a = layerActivation(numLayers() - 1);
                for (int n = 0; n < L.size; n++) {
                    double e = L.values[n] - targets[s][n];
                    L.grads[n] = 2.0 * e * activationDerivative(a, L.values[n]);
                }
            }

            // hidden layers backwards, accumulating weight gradients
            for (int l = numLayers() - 1; l >= 1; l--) {
                Layer& L = layers[l];
                Layer& P = layers[l - 1];
                for (int n = 0; n < L.size; n++) {
                    double g = L.grads[n];
                    biasGrads[l][n] += g;
                    double* wg = &L.weightGrads[(size_t)n * L.prevSize];
                    for (int p = 0; p < L.prevSize; p++) wg[p] += g * P.values[p];
                }
                if (l > 1) {
                    Activation a = layerActivation(l - 1);
                    for (int p = 0; p < P.size; p++) {
                        double sum = 0;
                        for (int n = 0; n < L.size; n++) {
                            sum += L.grads[n] * L.weights[(size_t)n * L.prevSize + p];
                        }
                        P.grads[p] = sum * activationDerivative(a, P.values[p]);
                    }
                }
            }
        }

        // apply averaged gradients with momentum
        double invBatch = 1.0 / (double)inputs.size();
        for (int l = 1; l < numLayers(); l++) {
            Layer& L = layers[l];
            for (int n = 0; n < L.size; n++) {
                L.biasVel[n] = momentum * L.biasVel[n] - learningRate * biasGrads[l][n] * invBatch;
                L.biases[n] += L.biasVel[n];
            }
            for (size_t w = 0; w < L.weights.size(); w++) {
                L.weightVel[w] = momentum * L.weightVel[w] - learningRate * L.weightGrads[w] * invBatch;
                L.weights[w] += L.weightVel[w];
            }
        }
    }

    // Seed format shared with the TypeScript version:
    // NNSEED1|activation|outputActivation|layerSizes(csv)|params(csv)
    // params: for each layer after the input, per neuron: bias then weights
    std::string toSeed() const {
        std::string s = "NNSEED1|";
        s += activationName(act);
        s += '|';
        s += activationName(outAct);
        s += '|';
        char buf[64];
        for (size_t i = 0; i < sizes.size(); i++) {
            snprintf(buf, sizeof(buf), i ? ",%d" : "%d", sizes[i]);
            s += buf;
        }
        s += '|';
        bool first = true;
        for (int l = 1; l < numLayers(); l++) {
            const Layer& L = layers[l];
            for (int n = 0; n < L.size; n++) {
                snprintf(buf, sizeof(buf), first ? "%.17g" : ",%.17g", L.biases[n]);
                s += buf;
                first = false;
                const double* w = &L.weights[(size_t)n * L.prevSize];
                for (int p = 0; p < L.prevSize; p++) {
                    snprintf(buf, sizeof(buf), ",%.17g", w[p]);
                    s += buf;
                }
            }
        }
        return s;
    }

    static bool fromSeed(const std::string& seed, NeuralNetwork& out) {
        // split into 5 fields
        std::vector<std::string> parts;
        size_t start = 0;
        while (true) {
            size_t bar = seed.find('|', start);
            if (bar == std::string::npos) { parts.push_back(seed.substr(start)); break; }
            parts.push_back(seed.substr(start, bar - start));
            start = bar + 1;
        }
        if (parts.size() != 5 || parts[0] != "NNSEED1") return false;

        Activation a, oa;
        if (!activationFromName(parts[1], a) || !activationFromName(parts[2], oa)) return false;

        std::vector<int> sizes;
        {
            const char* p = parts[3].c_str();
            char* end;
            while (*p) {
                long v = strtol(p, &end, 10);
                if (end == p || v <= 0) return false;
                sizes.push_back((int)v);
                p = *end == ',' ? end + 1 : end;
            }
        }
        if (sizes.size() < 2) return false;

        std::vector<int> hidden(sizes.begin() + 1, sizes.end() - 1);
        out = NeuralNetwork(sizes.front(), hidden, sizes.back(), a, oa);

        const char* p = parts[4].c_str();
        char* end;
        for (int l = 1; l < out.numLayers(); l++) {
            Layer& L = out.layers[l];
            for (int n = 0; n < L.size; n++) {
                L.biases[n] = strtod(p, &end);
                if (end == p) return false;
                p = *end == ',' ? end + 1 : end;
                double* w = &L.weights[(size_t)n * L.prevSize];
                for (int q = 0; q < L.prevSize; q++) {
                    w[q] = strtod(p, &end);
                    if (end == p) return false;
                    p = *end == ',' ? end + 1 : end;
                }
            }
        }
        return *p == '\0' || *p == '\n' || *p == '\r';
    }
};

// Population that mirrors NeuralNetworkList from the TypeScript version
struct Population {
    int targetCount = 16;
    std::vector<NeuralNetwork> nets;
    int generation = 0;
    double trainRMSE = 0, trainMAE = 0, testRMSE = 0, testMAE = 0;
    double trialPower = 2.0;

    std::vector<std::vector<double>> trialInputs;
    std::vector<std::vector<double>> trialOutputs;
    std::vector<std::vector<double>> testInputs;
    std::vector<std::vector<double>> testOutputs;

    // backprop keeps a worker net and shows the best one found so far
    NeuralNetwork worker;
    bool hasWorker = false;
    double workerBestError = 1e300;

    void init(int count, int inputSize, const std::vector<int>& hidden, int outputSize,
              Activation a, Activation oa) {
        targetCount = count;
        nets.clear();
        for (int i = 0; i < count; i++) nets.emplace_back(inputSize, hidden, outputSize, a, oa);
        generation = 0;
        trainRMSE = trainMAE = testRMSE = testMAE = 0;
        hasWorker = false;
        workerBestError = 1e300;
    }

    // replace the whole population with copies of one network
    void setAll(const NeuralNetwork& nn) {
        for (auto& n : nets) n = nn;
        hasWorker = false;
        workerBestError = 1e300;
    }

    void measure(NeuralNetwork& nn) {
        nn.error = 0;
        nn.meanError = 0;
        for (size_t i = 0; i < trialInputs.size(); i++) {
            nn.run(trialInputs[i]);
            nn.addError(trialOutputs[i], trialPower);
        }
        nn.error = std::pow(nn.error / (double)trialInputs.size(), 1.0 / trialPower);
        nn.meanError /= (double)trialInputs.size();
    }

    void sortByError() {
        std::sort(nets.begin(), nets.end(),
                  [](const NeuralNetwork& a, const NeuralNetwork& b) { return a.error < b.error; });
    }

    double runGeneration(int numWeights, double weightStrength, int numBiases, double biasStrength) {
        generation++;
        for (auto& nn : nets) measure(nn);
        sortByError();
        double best = nets[0].error;
        trainRMSE = best;
        trainMAE = nets[0].meanError;

        // kill the worst half, refill with mutated clones of survivors, plus one fresh net
        int survivors = targetCount / 2;
        nets.resize(survivors);
        int index = 0;
        while ((int)nets.size() < targetCount - 1) {
            nets.push_back(nets[index]);
            nets.back().mutate(numWeights, weightStrength, numBiases, biasStrength);
            index = (index + 1) % survivors;
        }
        const NeuralNetwork& proto = nets[0];
        std::vector<int> hidden(proto.sizes.begin() + 1, proto.sizes.end() - 1);
        nets.emplace_back(proto.inputSize(), hidden, proto.outputSize(), proto.act, proto.outAct);
        return best;
    }

    double trainBackprop(double learningRate, double momentum) {
        generation++;
        if (!hasWorker) {
            worker = nets[0];
            hasWorker = true;
            workerBestError = 1e300;
        }
        worker.trainBatch(trialInputs, trialOutputs, learningRate, momentum);
        measure(worker);
        if (worker.error < workerBestError) {
            workerBestError = worker.error;
            nets[0] = worker;
        }
        trainRMSE = workerBestError;
        trainMAE = nets[0].meanError;
        return workerBestError;
    }

    void computeTestErr() {
        if (testInputs.empty()) { testRMSE = 0; testMAE = 0; return; }
        double sqSum = 0, absSum = 0;
        NeuralNetwork& nn = nets[0];
        for (size_t i = 0; i < testInputs.size(); i++) {
            const std::vector<double>& pred = nn.run(testInputs[i]);
            for (size_t j = 0; j < pred.size(); j++) {
                double diff = std::fabs(pred[j] - testOutputs[i][j]);
                sqSum += diff * diff;
                absSum += diff;
            }
        }
        testRMSE = std::sqrt(sqSum / (double)testInputs.size());
        testMAE = absSum / (double)testInputs.size();
    }
};
