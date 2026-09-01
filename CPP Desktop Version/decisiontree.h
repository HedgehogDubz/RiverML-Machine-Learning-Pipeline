// Decision tree for regression, used by the XGBoost stage of the RiverML pipeline.
// Nodes live in a flat vector and reference children by index.
#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace RiverML {

struct TreeNode {
    int feature = -1;
    double threshold = 0;
    int left = -1;
    int right = -1;
    std::vector<double> value; // leaf when non empty
    bool isLeaf() const { return !value.empty(); }
};

struct DecisionTree {
    std::vector<TreeNode> nodes; // nodes[0] is the root
    int maxDepth = 4;
    int minSamplesLeaf = 2;
    int inputSize = 2;
    int outputSize = 1;

    void train(const std::vector<std::vector<double>>& inputs,
               const std::vector<std::vector<double>>& outputs) {
        nodes.clear();
        std::vector<int> idx(inputs.size());
        std::iota(idx.begin(), idx.end(), 0);
        buildTree(inputs, outputs, idx, 0);
    }

    const std::vector<double>& predict(const std::vector<double>& input) const {
        int ni = 0;
        while (!nodes[ni].isLeaf()) {
            ni = input[nodes[ni].feature] < nodes[ni].threshold ? nodes[ni].left : nodes[ni].right;
        }
        return nodes[ni].value;
    }

    // node indices along the prediction path, for highlighting
    std::vector<int> getPath(const std::vector<double>& input) const {
        std::vector<int> path;
        int ni = 0;
        while (true) {
            path.push_back(ni);
            if (nodes[ni].isLeaf()) break;
            ni = input[nodes[ni].feature] < nodes[ni].threshold ? nodes[ni].left : nodes[ni].right;
        }
        return path;
    }

private:
    std::vector<double> calculateMean(const std::vector<std::vector<double>>& outputs,
                               const std::vector<int>& idx) const {
        std::vector<double> mean(outputSize, 0.0);
        if (idx.empty()) return mean;
        for (int i : idx)
            for (int d = 0; d < outputSize; d++) mean[d] += outputs[i][d];
        for (double& m : mean) m /= (double)idx.size();
        return mean;
    }

    // returns the new node's index
    int buildTree(const std::vector<std::vector<double>>& X,
              const std::vector<std::vector<double>>& Y,
              std::vector<int>& idx, int depth) {
        int me = (int)nodes.size();
        nodes.push_back(TreeNode());

        if (depth >= maxDepth || (int)idx.size() < minSamplesLeaf) {
            nodes[me].value = calculateMean(Y, idx);
            return me;
        }

        // find the split minimizing weighted variance, using sorted prefix sums
        int n = (int)idx.size();
        double bestCost = 1e300;
        int bestFeature = -1;
        double bestThreshold = 0;

        std::vector<std::pair<double, int>> order(n);
        std::vector<double> sumFull(outputSize, 0.0);
        double sqFull = 0;
        for (int i : idx) {
            for (int d = 0; d < outputSize; d++) {
                sumFull[d] += Y[i][d];
                sqFull += Y[i][d] * Y[i][d];
            }
        }

        std::vector<double> sumL(outputSize);
        for (int f = 0; f < inputSize; f++) {
            for (int k = 0; k < n; k++) order[k] = { X[idx[k]][f], idx[k] };
            std::ranges::sort(order);

            std::fill(sumL.begin(), sumL.end(), 0.0);
            double sqL = 0;
            for (int k = 1; k < n; k++) {
                const std::vector<double>& y = Y[order[k - 1].second];
                for (int d = 0; d < outputSize; d++) {
                    sumL[d] += y[d];
                    sqL += y[d] * y[d];
                }
                if (order[k - 1].first == order[k].first) continue;

                // cost = nL*varL + nR*varR, where n*var = sq - sum^2/n
                double cost = sqFull;
                for (int d = 0; d < outputSize; d++) {
                    double sumR = sumFull[d] - sumL[d];
                    cost -= sumL[d] * sumL[d] / k + sumR * sumR / (n - k);
                }
                if (cost < bestCost) {
                    bestCost = cost;
                    bestFeature = f;
                    bestThreshold = (order[k - 1].first + order[k].first) / 2;
                }
            }
        }

        if (bestFeature < 0) {
            nodes[me].value = calculateMean(Y, idx);
            return me;
        }

        std::vector<int> leftIdx, rightIdx;
        for (int i : idx) {
            if (X[i][bestFeature] < bestThreshold) leftIdx.push_back(i);
            else rightIdx.push_back(i);
        }

        nodes[me].feature = bestFeature;
        nodes[me].threshold = bestThreshold;
        int L = buildTree(X, Y, leftIdx, depth + 1);
        int R = buildTree(X, Y, rightIdx, depth + 1);
        nodes[me].left = L;
        nodes[me].right = R;
        return me;
    }
};

} // namespace RiverML
