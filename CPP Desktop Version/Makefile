# Build the RiverML desktop pipeline. Needs raylib (brew install raylib).
CXX = c++
CXXFLAGS = -O2 -std=c++23
RAYLIB = $(shell pkg-config --cflags --libs raylib)

all: mlvisualizer

mlvisualizer: main.cpp neuralnetwork.h decisiontree.h xgboost.h
	$(CXX) $(CXXFLAGS) main.cpp -o mlvisualizer $(RAYLIB)

clean:
	rm -f mlvisualizer

.PHONY: all clean
