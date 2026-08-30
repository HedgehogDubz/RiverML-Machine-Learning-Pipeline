// XGBoost style ensemble of decision trees, trained on residuals.
// Seed format is shared with the TypeScript version.
#pragma once

#include "decisiontree.h"

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <string>

struct XGBoostEnsemble {
    std::vector<DecisionTree> trees;
    int generation = 0;
    int inputSize = 2;
    int outputSize = 1;
    double shrinkage = 0.1;
    int maxDepth = 4;
    int maxTrees = INT_MAX;
    double trainRMSE = 0, trainMAE = 0, testRMSE = 0, testMAE = 0;

    void reset(int inSize, int outSize) {
        trees.clear();
        generation = 0;
        inputSize = inSize;
        outputSize = outSize;
        trainRMSE = trainMAE = testRMSE = testMAE = 0;
    }

    std::vector<double> predict(const std::vector<double>& input) const {
        std::vector<double> result(outputSize, 0.0);
        for (size_t t = 0; t < trees.size(); t++) {
            const std::vector<double>& pred = trees[t].predict(input);
            double weight = t == 0 ? 1.0 : shrinkage; // base tree full weight, the rest scaled
            for (int j = 0; j < outputSize; j++) result[j] += pred[j] * weight;
        }
        return result;
    }

    // add one tree trained on the residuals of the current ensemble
    void train(const std::vector<std::vector<double>>& inputs,
               const std::vector<std::vector<double>>& outputs,
               const std::vector<std::vector<double>>& testInputs,
               const std::vector<std::vector<double>>& testOutputs) {
        generation++;

        if (trees.empty()) {
            DecisionTree base;
            base.inputSize = inputSize;
            base.outputSize = outputSize;
            base.maxDepth = maxDepth;
            base.train(inputs, outputs);
            trees.push_back(std::move(base));
            computeErrors(inputs, outputs, testInputs, testOutputs);
            return;
        }

        if ((int)trees.size() < maxTrees) {
            std::vector<std::vector<double>> residuals(inputs.size());
            for (size_t i = 0; i < inputs.size(); i++) {
                std::vector<double> pred = predict(inputs[i]);
                residuals[i].resize(outputSize);
                for (int j = 0; j < outputSize; j++) residuals[i][j] = outputs[i][j] - pred[j];
            }
            DecisionTree tree;
            tree.inputSize = inputSize;
            tree.outputSize = outputSize;
            tree.maxDepth = maxDepth;
            tree.train(inputs, residuals);
            trees.push_back(std::move(tree));
        }

        computeErrors(inputs, outputs, testInputs, testOutputs);
    }

    void computeErrors(const std::vector<std::vector<double>>& inputs,
                       const std::vector<std::vector<double>>& outputs,
                       const std::vector<std::vector<double>>& testInputs,
                       const std::vector<std::vector<double>>& testOutputs) {
        double sqSum = 0, absSum = 0;
        for (size_t i = 0; i < inputs.size(); i++) {
            std::vector<double> pred = predict(inputs[i]);
            for (int j = 0; j < outputSize; j++) {
                double diff = std::fabs(pred[j] - outputs[i][j]);
                sqSum += diff * diff;
                absSum += diff;
            }
        }
        trainRMSE = std::sqrt(sqSum / (double)inputs.size());
        trainMAE = absSum / (double)inputs.size();

        if (testInputs.empty()) { testRMSE = 0; testMAE = 0; return; }
        sqSum = 0; absSum = 0;
        for (size_t i = 0; i < testInputs.size(); i++) {
            std::vector<double> pred = predict(testInputs[i]);
            for (int j = 0; j < outputSize; j++) {
                double diff = std::fabs(pred[j] - testOutputs[i][j]);
                sqSum += diff * diff;
                absSum += diff;
            }
        }
        testRMSE = std::sqrt(sqSum / (double)testInputs.size());
        testMAE = absSum / (double)testInputs.size();
    }

    // Seed format shared with the TypeScript version:
    // XGBSEED1|inputSize,outputSize|shrinkage|tree;tree;...
    // each tree is preorder csv tokens: d,feature,threshold,... or l,v0,v1,...
    std::string toSeed() const {
        char buf[64];
        std::string s = "XGBSEED1|";
        snprintf(buf, sizeof(buf), "%d,%d|%.17g|", inputSize, outputSize, shrinkage);
        s += buf;
        for (size_t t = 0; t < trees.size(); t++) {
            if (t) s += ';';
            serNode(trees[t], 0, s);
        }
        return s;
    }

    static bool fromSeed(const std::string& seed, XGBoostEnsemble& out) {
        std::vector<std::string> parts;
        size_t start = 0;
        while (true) {
            size_t bar = seed.find('|', start);
            if (bar == std::string::npos) { parts.push_back(seed.substr(start)); break; }
            parts.push_back(seed.substr(start, bar - start));
            start = bar + 1;
        }
        if (parts.size() != 4 || parts[0] != "XGBSEED1") return false;

        char* end;
        int inSize = (int)strtol(parts[1].c_str(), &end, 10);
        if (*end != ',') return false;
        int outSize = (int)strtol(end + 1, &end, 10);
        if (*end != '\0' || inSize <= 0 || outSize <= 0) return false;
        double shrink = strtod(parts[2].c_str(), &end);
        if (end == parts[2].c_str()) return false;

        out.reset(inSize, outSize);
        out.shrinkage = shrink;

        // split trees on ';', parse each preorder
        std::string& treesStr = parts[3];
        size_t pos = 0;
        while (pos <= treesStr.size()) {
            size_t semi = treesStr.find(';', pos);
            std::string treeStr = treesStr.substr(pos, semi == std::string::npos ? std::string::npos : semi - pos);
            DecisionTree tree;
            tree.inputSize = inSize;
            tree.outputSize = outSize;
            const char* p = treeStr.c_str();
            if (parseNode(tree, p, outSize) < 0 || *p != '\0') return false;
            out.trees.push_back(std::move(tree));
            if (semi == std::string::npos) break;
            pos = semi + 1;
        }
        out.generation = (int)out.trees.size();
        return !out.trees.empty();
    }

private:
    static void serNode(const DecisionTree& t, int ni, std::string& s) {
        const TreeNode& n = t.nodes[ni];
        char buf[64];
        if (n.isLeaf()) {
            s += 'l';
            for (double v : n.value) {
                snprintf(buf, sizeof(buf), ",%.17g", v);
                s += buf;
            }
            return;
        }
        snprintf(buf, sizeof(buf), "d,%d,%.17g,", n.feature, n.threshold);
        s += buf;
        serNode(t, n.left, s);
        s += ',';
        serNode(t, n.right, s);
    }

    // parses one node from csv tokens, advancing p, returns the node index or -1
    static int parseNode(DecisionTree& tree, const char*& p, int outputSize) {
        int me = (int)tree.nodes.size();
        tree.nodes.push_back(TreeNode());
        char* end;
        if (*p == 'l') {
            p++;
            for (int i = 0; i < outputSize; i++) {
                if (*p != ',') return -1;
                double v = strtod(p + 1, &end);
                if (end == p + 1) return -1;
                tree.nodes[me].value.push_back(v);
                p = end;
            }
            return me;
        }
        if (*p != 'd') return -1;
        p++;
        if (*p != ',') return -1;
        long f = strtol(p + 1, &end, 10);
        if (end == p + 1 || *end != ',') return -1;
        double th = strtod(end + 1, &end);
        if (*end != ',') return -1;
        p = end + 1;
        tree.nodes[me].feature = (int)f;
        tree.nodes[me].threshold = th;
        int L = parseNode(tree, p, outputSize);
        if (L < 0 || *p != ',') return -1;
        p++;
        int R = parseNode(tree, p, outputSize);
        if (R < 0) return -1;
        tree.nodes[me].left = L;
        tree.nodes[me].right = R;
        return me;
    }
};
