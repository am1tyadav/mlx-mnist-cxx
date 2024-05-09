// Copyright © 2023 Apple Inc.

#include <chrono>
#include <cmath>
#include <iostream>

#include "arena.h"
#include "dataset.h"
#include "mlx/mlx.h"

using namespace mlx::core;

typedef struct {
    array X;
    array y;
} MNISTArrayDataset;

MNISTArrayDataset *create_array(Arena *arena, MNISTData *data, int num_examples, int num_features) {
    MNISTArrayDataset *dataset = (MNISTArrayDataset *) arena_allocate(arena, sizeof(MNISTArrayDataset));

    MNISTData *data_zeros_and_ones = get_zeros_and_ones(arena, data);

    float *X = (float *) arena_allocate(arena, num_examples * num_features * sizeof(float));
    float *y = (float *) arena_allocate(arena, num_examples * sizeof(float));

    for (int i = 0; i < num_examples; ++i) {
        for (int j = 0; j < num_features; ++j) {
            float pixel = (float) data_zeros_and_ones->images[i * num_features + j] / 255.0f;
            X[i * num_features + j] = pixel;
        }
        float current_label = (float) data_zeros_and_ones->labels[i];
        y[i] = current_label;
    }

    dataset->X = array(X, {num_examples, num_features}, float32);
    dataset->y = array(y, {num_examples}, float32);

    return dataset;
};

int main() {
    int num_iters = 400;
    int log_interval = 20;
    float learning_rate = 0.01;

    Arena *arena = arena_create(200'000'000);

    MNISTData *train_data = load_dataset(arena, NUM_TRAIN_EXAMPLES, TRAIN_IMAGES_FILEPATH, TRAIN_LABELS_FILEPATH);
    MNISTData *train_data_zeros_and_ones = get_zeros_and_ones(arena, train_data);
    MNISTData *test_data = load_dataset(arena, NUM_TEST_EXAMPLES, TEST_IMAGES_FILEPATH, TEST_LABELS_FILEPATH);
    MNISTData *test_data_zeros_and_ones = get_zeros_and_ones(arena, test_data);

    int num_features = train_data_zeros_and_ones->num_cols * train_data_zeros_and_ones->num_rows;
    int num_train_examples = train_data_zeros_and_ones->num_items;
    int num_test_examples = test_data_zeros_and_ones->num_items;

    std::cout << "Training examples: " << num_train_examples << std::endl;
    std::cout << "Test examples: " << num_test_examples << std::endl;

    // Create float arrays X and y and initialise them to 0
    MNISTArrayDataset *train_dataset = create_array(arena, train_data_zeros_and_ones, num_train_examples, num_features);
    MNISTArrayDataset *test_dataset = create_array(arena, test_data_zeros_and_ones, num_test_examples, num_features);

    // Initialize random parameters
    array w = 1e-2 * random::normal({num_features});

    auto loss_fn = [&](array w) {
        auto logits = matmul(train_dataset->X, w);
        auto scale = (1.0f / num_train_examples);
        return scale * sum(logaddexp(array(0.0f), logits) - train_dataset->y * logits);
    };

    auto test_loss_fn = [&](array w) {
        auto logits = matmul(test_dataset->X, w);
        auto scale = (1.0f / num_test_examples);
        return scale * sum(logaddexp(array(0.0f), logits) - test_dataset->y * logits);
    };

    auto grad_fn = grad(loss_fn);

    auto compute_test_acc = [&](array w) {
        auto acc = sum((matmul(test_dataset->X, w) > 0) == test_dataset->y) / num_test_examples;
        return acc;
    };

    for (int it = 0; it < num_iters; ++it) {
        auto grad = grad_fn(w);
        w = w - learning_rate * grad;
        eval(w);

        if (it % log_interval == 0) {
            auto loss = loss_fn(w);
            auto test_loss = test_loss_fn(w);
            auto test_acc = compute_test_acc(w);

            std::cout << "Iter " << it << ", Loss " << loss << ", Test loss " << test_loss << ", Test acc " << test_acc << std::endl;
        }
    }

    auto test_loss = test_loss_fn(w);
    auto test_acc = compute_test_acc(w);

    std::cout << "Final test loss " << test_loss << std::endl;
    std::cout << "Final test accuracy, " << test_acc << std::endl;

    arena_destroy(arena);

    return 0;
}
