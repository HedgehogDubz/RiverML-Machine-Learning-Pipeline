# COMPILED ON MAC
brew install raylib
make ./mlvisualizer && ./mlvisualizer

# SEEDS
Copy Seed puts the model on the clipboard, in the seed box, and in seed.txt.
Load Seed reads the seed box: click the box, cmd V to paste, then press Load Seed.
Seeds work in both directions with the TypeScript version.

# RUNNING A SEED ELSEWHERE
neural_network_by_seed.cpp and xgboost_by_seed.cpp are standalone, copy either file anywhere.
Both are in namespace RiverML and parse the seed once in the constructor:

    RiverML::NeuralNetwork nn(seed);
    std::vector<double> outputs = nn.run({0.5, -0.2});

    RiverML::XGBoost model(seed);
    std::vector<double> outputs2 = model.run({0.5, -0.2});

They also build as demo CLIs:
    make run_seed && ./run_seed seed.txt 0.5 -0.2
    make run_xgb_seed && ./run_xgb_seed seed.txt 0.5 -0.2