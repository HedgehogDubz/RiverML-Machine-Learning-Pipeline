// Standalone neural network runner.
// Copy and paste this file anywhere, no other files needed.
// Get a seed by training a model in the visualizer and clicking Copy Seed.
// Works with seeds from both the C++ and TypeScript versions.
//
// Usage:
//   std::vector<double> outputs = neuralNetworkBySeed(seed, {0.5, -0.2});
//
// Or compile with a demo main and pass the seed file plus inputs:
//   c++ -O2 -std=c++17 -DNN_SEED_MAIN neural_network_by_seed.cpp -o run_seed
//   ./run_seed seed.txt 0.5 -0.2

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

std::vector<double> neuralNetworkBySeed(const std::string& seed, const std::vector<double>& inputs) {
    // Seed format: NNSEED1|activation|outputActivation|layerSizes(csv)|params(csv)
    // params: for each layer after the input, per neuron: bias then weights
    std::vector<std::string> parts;
    size_t start = 0;
    while (true) {
        size_t bar = seed.find('|', start);
        if (bar == std::string::npos) { parts.push_back(seed.substr(start)); break; }
        parts.push_back(seed.substr(start, bar - start));
        start = bar + 1;
    }
    if (parts.size() != 5 || parts[0] != "NNSEED1") {
        throw std::runtime_error("Invalid seed: expected NNSEED1|act|outAct|sizes|params");
    }
    const std::string& activation = parts[1];
    const std::string& outputActivation = parts[2];

    std::vector<int> sizes;
    {
        const char* p = parts[3].c_str();
        char* end;
        while (*p) {
            long v = strtol(p, &end, 10);
            if (end == p || v <= 0) throw std::runtime_error("Invalid seed: bad layer sizes");
            sizes.push_back((int)v);
            p = *end == ',' ? end + 1 : end;
        }
    }
    if (sizes.size() < 2) throw std::runtime_error("Invalid seed: need at least 2 layers");
    if ((int)inputs.size() != sizes[0]) throw std::runtime_error("Wrong number of inputs");

    auto activate = [](const std::string& fn, double x) {
        if (fn == "relu") return x > 0 ? x : 0.0;
        if (fn == "sigmoid") return 1.0 / (1.0 + std::exp(-x));
        return std::tanh(x);
    };

    // forward pass, layer by layer, reading params as we go
    const char* p = parts[4].c_str();
    char* end;
    std::vector<double> values = inputs;
    for (size_t l = 1; l < sizes.size(); l++) {
        const std::string& fn = l == sizes.size() - 1 ? outputActivation : activation;
        std::vector<double> next(sizes[l]);
        for (int n = 0; n < sizes[l]; n++) {
            double sum = strtod(p, &end);
            if (end == p) throw std::runtime_error("Invalid seed: not enough parameters");
            p = *end == ',' ? end + 1 : end;
            for (size_t w = 0; w < values.size(); w++) {
                sum += strtod(p, &end) * values[w];
                if (end == p) throw std::runtime_error("Invalid seed: not enough parameters");
                p = *end == ',' ? end + 1 : end;
            }
            next[n] = activate(fn, sum);
        }
        values = next;
    }
    if (*p != '\0' && *p != '\n' && *p != '\r') {
        throw std::runtime_error("Invalid seed: too many parameters");
    }
    return values;
}

#ifdef NN_SEED_MAIN
#include <cstdio>
#include <fstream>
#include <sstream>

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <seed file> <input1> [input2 ...]\n", argv[0]);
        return 1;
    }
    std::ifstream f(argv[1]);
    if (!f) { printf("Could not open %s\n", argv[1]); return 1; }
    std::string seed;
    std::getline(f, seed);
    std::vector<double> inputs;
    for (int i = 2; i < argc; i++) inputs.push_back(atof(argv[i]));
    try {
        std::vector<double> out = neuralNetworkBySeed(seed, inputs);
        for (double v : out) printf("%.17g\n", v);
    } catch (const std::exception& e) {
        printf("Error: %s\n", e.what());
        return 1;
    }
    return 0;
}
#endif
