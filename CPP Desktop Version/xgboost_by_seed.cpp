// Standalone XGBoost runner.
// Copy and paste this file anywhere, no other files needed.
// Get a seed by training an XGBoost model in the visualizer and clicking Copy Seed.
// Works with seeds from both the C++ and TypeScript versions.
//
// Usage:
//   std::vector<double> outputs = xgboostBySeed(seed, {0.5, -0.2});
//
// Or compile with a demo main and pass the seed file plus inputs:
//   c++ -O2 -std=c++17 -DXGB_SEED_MAIN xgboost_by_seed.cpp -o run_xgb_seed
//   ./run_xgb_seed seed.txt 0.5 -0.2

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
// skips a whole subtree without evaluating it
void skipNode(const char*& p, int outputSize) {
    char* end;
    if (*p == 'l') {
        p++;
        for (int i = 0; i < outputSize; i++) {
            if (*p != ',') throw std::runtime_error("Invalid seed: bad leaf");
            strtod(p + 1, &end);
            p = end;
        }
        return;
    }
    if (*p != 'd') throw std::runtime_error("Invalid seed: unknown node tag");
    p++;
    for (int i = 0; i < 2; i++) {
        if (*p != ',') throw std::runtime_error("Invalid seed: bad split");
        strtod(p + 1, &end);
        p = end;
    }
    if (*p != ',') throw std::runtime_error("Invalid seed: bad split");
    p++;
    skipNode(p, outputSize);
    if (*p != ',') throw std::runtime_error("Invalid seed: bad split");
    p++;
    skipNode(p, outputSize);
}

// walks one tree to its leaf, reading tokens as it goes
std::vector<double> walkNode(const char*& p, const std::vector<double>& inputs, int outputSize) {
    char* end;
    if (*p == 'l') {
        p++;
        std::vector<double> value(outputSize);
        for (int i = 0; i < outputSize; i++) {
            if (*p != ',') throw std::runtime_error("Invalid seed: bad leaf");
            value[i] = strtod(p + 1, &end);
            if (end == p + 1) throw std::runtime_error("Invalid seed: bad leaf");
            p = end;
        }
        return value;
    }
    if (*p != 'd') throw std::runtime_error("Invalid seed: unknown node tag");
    p++;
    if (*p != ',') throw std::runtime_error("Invalid seed: bad split");
    long feature = strtol(p + 1, &end, 10);
    if (*end != ',') throw std::runtime_error("Invalid seed: bad split");
    double threshold = strtod(end + 1, &end);
    if (*end != ',') throw std::runtime_error("Invalid seed: bad split");
    p = end + 1;
    if (inputs[feature] < threshold) {
        std::vector<double> value = walkNode(p, inputs, outputSize);
        if (*p != ',') throw std::runtime_error("Invalid seed: bad split");
        p++;
        skipNode(p, outputSize);
        return value;
    }
    skipNode(p, outputSize);
    if (*p != ',') throw std::runtime_error("Invalid seed: bad split");
    p++;
    return walkNode(p, inputs, outputSize);
}
} // namespace

std::vector<double> xgboostBySeed(const std::string& seed, const std::vector<double>& inputs) {
    // Seed format: XGBSEED1|inputSize,outputSize|shrinkage|tree;tree;...
    // each tree is preorder csv tokens: d,feature,threshold,... or l,v0,v1,...
    std::vector<std::string> parts;
    size_t start = 0;
    while (true) {
        size_t bar = seed.find('|', start);
        if (bar == std::string::npos) { parts.push_back(seed.substr(start)); break; }
        parts.push_back(seed.substr(start, bar - start));
        start = bar + 1;
    }
    if (parts.size() != 4 || parts[0] != "XGBSEED1") {
        throw std::runtime_error("Invalid seed: expected XGBSEED1|sizes|shrinkage|trees");
    }
    char* end;
    int inputSize = (int)strtol(parts[1].c_str(), &end, 10);
    if (*end != ',') throw std::runtime_error("Invalid seed: bad sizes");
    int outputSize = (int)strtol(end + 1, &end, 10);
    if (*end != '\0' || inputSize <= 0 || outputSize <= 0) throw std::runtime_error("Invalid seed: bad sizes");
    double shrinkage = strtod(parts[2].c_str(), &end);
    if (end == parts[2].c_str()) throw std::runtime_error("Invalid seed: bad shrinkage");
    if ((int)inputs.size() != inputSize) throw std::runtime_error("Wrong number of inputs");

    // sum the trees, first tree full weight, the rest scaled by shrinkage
    std::vector<double> result(outputSize, 0.0);
    const std::string& treesStr = parts[3];
    size_t pos = 0;
    int t = 0;
    while (pos <= treesStr.size()) {
        size_t semi = treesStr.find(';', pos);
        std::string treeStr = treesStr.substr(pos, semi == std::string::npos ? std::string::npos : semi - pos);
        const char* p = treeStr.c_str();
        std::vector<double> pred = walkNode(p, inputs, outputSize);
        double weight = t == 0 ? 1.0 : shrinkage;
        for (int j = 0; j < outputSize; j++) result[j] += pred[j] * weight;
        t++;
        if (semi == std::string::npos) break;
        pos = semi + 1;
    }
    return result;
}

#ifdef XGB_SEED_MAIN
#include <cstdio>
#include <fstream>

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
        std::vector<double> out = xgboostBySeed(seed, inputs);
        for (double v : out) printf("%.17g\n", v);
    } catch (const std::exception& e) {
        printf("Error: %s\n", e.what());
        return 1;
    }
    return 0;
}
#endif
