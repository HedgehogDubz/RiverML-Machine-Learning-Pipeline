# Build the desktop visualizer. Needs raylib (brew install raylib).
CXX = c++
CXXFLAGS = -O2 -std=c++17
RAYLIB = $(shell pkg-config --cflags --libs raylib)

all: mlvisualizer

mlvisualizer: main.cpp neuralnetwork.h decisiontree.h xgboost.h
	$(CXX) $(CXXFLAGS) main.cpp -o mlvisualizer $(RAYLIB)

# standalone seed runner demos
run_seed: neural_network_by_seed.cpp
	$(CXX) $(CXXFLAGS) -DNN_SEED_MAIN neural_network_by_seed.cpp -o run_seed

run_xgb_seed: xgboost_by_seed.cpp
	$(CXX) $(CXXFLAGS) -DXGB_SEED_MAIN xgboost_by_seed.cpp -o run_xgb_seed

clean:
	rm -f mlvisualizer run_seed run_xgb_seed

.PHONY: all clean
