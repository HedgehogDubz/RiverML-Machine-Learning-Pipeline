## Machine Learning Visualizer

Two versions of the same simulator live in this repo:

- `Typescript Web Version/` runs in the browser (this is what gets deployed to GitHub Pages)
- `CPP Desktop Version/` is a native desktop build using raylib, much faster for training

Both can export a trained model as a seed string (Copy Seed button) and load one back in (Load Seed). Seeds work across both versions. Each folder also has a standalone `neural_network_by_seed` file: one function that takes a seed and inputs and runs the network, no other files needed.

This project started as a simple idea: make machine learning easier to understand by seeing it happen.
I wanted something lightweight that anyone could run directly in a browser. That’s why this visualizer is built in TypeScript — it keeps everything web-native, easy to run, and accessible without needing extra setup or support.

## Why I Built This

Machine learning can feel abstract when you only see formulas or final results. I wanted something where you can:
See the model update live
Follow what the network is actually doing
Understand the logic step-by-step instead of treating it like a black box
Change parameters to see how it affects the model

This was also an opportunity for me to code from scratch. First and foremost, I created this project to level up my coding ability! 
This project was also made to show my fundamental understanding of some of the most popular Machine Learning Models.

## Design Philosophy

This project is intentionally designed around readability and learning.

Some choices I made on purpose:
Code is written to be easy to follow, in an object-oriented programming style.
Logic is separated cleanly so you can trace how data flows through the model.
Variable names and structure are meant to be readable for beginners or people just learning ML internals.
Everything runs directly on the web — no backend, no external compute required.

## Why TypeScript?

A big reason for using TypeScript was portability.
Runs directly in the browser
No external ML libraries required
Strong typing makes the logic easier to reason about
I wanted this to be something you could share via a link and immediately explore.
Later implementations may include an executable built in C++ capable of running the simulations 100-1000x faster


## Some directions I might explore later:

Different algorithms and training options
Better visualization controls
Richer debugging views
More customizability in the neural network and algorithms
The C++ implementation executable
