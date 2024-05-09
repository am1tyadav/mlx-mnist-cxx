# MLX MNIST Logistic Regression

MNIST classification with MLX C++ on a slice of MNIST data (Just the 0s and 1s)

## Requirements

- [CMake](https://cmake.org/download/)
- [Task](https:://taskfile.dev)
- gzip
- C++ compiler

MLX and Raylib will be downloaded as part of the build.

## Usage

```bash
# List all available tasks
task
# Download the dataset
task download
# Build the project
task build
# Run visualiser (sanity check)
task run:viz # Press [SPACE] to cycle through random images
# Train a logistic regression model
task run
```
