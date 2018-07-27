#include "Network.h"

#include "Layer.h"
//#include "Matrix.h"
#include <vector>
#include <list>
#include <iostream>
#include <ctime>

Network::Network(){}

Network::~Network(){}

//Trains the network for a given set of inputs
void Network::Train(std::list<Matrix> &inputs, std::list<Matrix> &expected, int hiddenCount, int outputCount, int trainingIterations) {
	//HiddenLayer = Layer(inputs.front().columns, hiddenCount);
	//OutputLayer = Layer(hiddenCount, outputCount);
	HiddenLayer.Create(inputs.front().columns, hiddenCount);
	OutputLayer.Create(hiddenCount, outputCount);

	std::srand(time(NULL));
	HiddenLayer.Init();
	OutputLayer.Init();

	//per batch
	for (int i = 0; i < trainingIterations; i++) {
		//per input
		double mse = 0;
		auto inputsA = inputs.begin();
		auto expectedA = expected.begin();
		while (inputsA != inputs.end()) {
			Inputs = Matrix(inputsA->matrix);
			Matrix Expected = Matrix(expectedA->matrix);

			Hidden = HiddenLayer.Feedforward(Inputs);
			Outputs = OutputLayer.Feedforward(Hidden);

			OutputLayer.Backpropagate(Outputs - Expected, Hidden);
			HiddenLayer.Backpropagate(OutputLayer.BiasGradients.Dot(OutputLayer.Weights.Transpose()), Inputs);

			mse += MSE(Expected) * MSE(Expected);

			++inputsA;
			++expectedA;
		}

		mse = mse / expected.size() * 100;

		HiddenLayer.UpdateWeights(mse > prevError);
		OutputLayer.UpdateWeights(mse > prevError);
		prevError = mse;

		PrintBatch(i, mse);

		if (mse < 0.0001)
			break;
	}

	IsTrained = true;
}

//Runs the network with a given set of inputs
void Network::Run(Matrix &inputs) {
	Inputs = inputs;

	if (IsTrained) {
		Hidden = HiddenLayer.Feedforward(Inputs);
		Outputs = OutputLayer.Feedforward(Hidden);
		PrintTest(Inputs);
	}
}

double Network::MSE(Matrix &expected) {
	return (expected - Outputs).Sum();
}

#pragma region PRINTING FUNCTIONS

void Network::PrintResults(Matrix &expected, int &i) {
	std::cout << "Iteration: " << i+1 << " ";
	std::cout << "Error: " << MSE(expected) << std::endl;
	std::cout << "Input: ";
	Inputs.PrintMatrix();
	std::cout << ", Output: ";
	Outputs.PrintMatrix();
	std::cout << ", Expected: ";
	expected.PrintMatrix();
	std::cout << std::endl;
}

void Network::PrintBatch(int &i, double &mse) {
	std::cout << "Iteration: " << i + 1 << " ";
	std::cout << "Error: " << mse << std::endl;
}

void Network::PrintTest(Matrix &inputs) {
	std::cout << "test: ";
	inputs.PrintMatrix();
	std::cout << "result: ";
	Outputs.Step().PrintMatrix();
	std::cout << std::endl;
}

#pragma endregion 