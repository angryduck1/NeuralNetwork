#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoidDerivative(double x) {
    return x * (1.0 - x);
}

class Perceptron {
public:
    Perceptron(size_t input_size, size_t output_size, double learningWeight = 0.1) : input_size(input_size), output_size(output_size), learningWeight(learningWeight) {
        srand(static_cast<unsigned>(time(nullptr)));

        biases.resize(output_size);
        weights.resize(output_size, vector<double>(input_size));

        for (int i = 0; i < output_size; ++i) {
            biases[i] = ((double)rand() / RAND_MAX) * 2 - 1;
            for (int x = 0; x < input_size; ++x) {
                weights[i][x] = ((double)rand() / RAND_MAX) * 2 - 1;
            }
        }
    }

    vector<double> predict(const vector<double>& input) {
        vector<double> outputs(output_size);
        for (int i = 0; i < output_size; ++i) {
            double sum = biases[i];
            for (int x = 0; x < input_size; ++x) {
                sum += weights[i][x] * input[x];
            }
            outputs[i] = sigmoid(sum);
        }

        return outputs;
    }

    void train(const vector<double>& input, const vector<double>& target) {
        vector<double> result = predict(input);

        for (int i = 0; i < output_size; ++i) {
            double error = target[i] - result[i];
            double delta = error * sigmoidDerivative(result[i]);
            for (int x = 0; x < input_size; ++x) {
                weights[i][x] += learningWeight * error * input[x];
            }

            biases[i] += learningWeight * delta;
        }
    }
private:
    size_t input_size;
    size_t output_size;
    vector<double> biases;
    vector<vector<double>> weights;
    double learningWeight = 0.1;
};

void what_is_that(Perceptron& net, const vector<double> input) {
    vector<double> result = net.predict(input);

    for (auto &i : input) {
        cout << static_cast<int>(i) << ",";
    }

    cout << endl;

    if (result[0] > result[1]) {
        cout << "This is two since probability: " << result[0] << endl;
        cout << "probability three: " << result[1] << endl;
    } else {
        cout << "This is three since probability: " << result[1] << endl;
        cout << "probability two: " << result[0] << endl;
    }
}

int main() {
    vector<vector<double>> two_one = {
        {1,1,0,0,0,0}, {0,1,0,1,0,0}, {0,0,0,1,1,0}, {0,0,0,1,0,1}, {1,0,0,1,0,0},
        {1,0,1,0,0,0}, {0,1,1,0,0,0}, {1,0,0,0,1,0}
    };

    vector<vector<double>> three_one = {
        {1,1,1,0,0,0}, {0,1,0,1,1,0}, {0,0,1,1,1,0}, {0,1,0,1,0,1}, {1,0,0,1,0,1},
        {1,1,0,1,0,1}, {0,1,1,1,0,0}, {0,0,1,1,0,1}
    };

    vector<pair<vector<double>, vector<double>>>train_data;

    for (auto& i : two_one) {
        train_data.push_back(std::make_pair(i, std::vector<double>{1, 0}));
    }

    for (auto& i : three_one) {
        train_data.push_back(std::make_pair(i, std::vector<double>{0, 1}));
    }

    Perceptron net(6, 2);

    for (int epoch = 0; epoch < 5000; ++epoch) {
        for (auto& i : train_data) {
            net.train(i.first, i.second);
        }
    }

    what_is_that(net, {1,0,0,1,0,1}); // three
    what_is_that(net, {1,0,0,1,0,0}); // two
    what_is_that(net, {0,0,0,1,1,1}); // three
    what_is_that(net, {1,0,1,0,0,0}); // two
    what_is_that(net, {1,1,0,1,0,0}); // three
}
