#include "Layer.h"

#include "Matrix.h"
#include <iostream>
#include <vector>

Layer::Layer() {}
Layer::~Layer() {}

void Layer::Create(int rows, int columns, double initialValues)
{
	Weights =		Matrix(rows, columns);
	Gradients =		Matrix(rows, columns);
	PrevGradients = Matrix(rows, columns);
	LearningRates = Matrix(std::vector<std::vector<double> >(rows, std::vector<double>(columns, initialValues)));
	PrevUpdates =	Matrix(rows, columns);

	Bias =			Matrix(std::vector<std::vector<double> >(1, std::vector<double>(columns, 1.0)));
	BiasGradients = Matrix(1, columns);
	BiasPrevGradients = Matrix(1, columns);
	BiasLearningRates = Matrix(std::vector<std::vector<double> >(1, std::vector<double>(columns, initialValues)));
	BiasPrevUpdates = Matrix(1, columns);

	Activations =	Matrix(1, columns);
}

//Randomize the starting weights of the network
void Layer::Init() {
	Weights = Weights.ApplyRandomize();
}

//Calculate the outputs of the network for the given inputs
Matrix Layer::Feedforward(Matrix &in) {
	Activations = in.Dot(Weights) + Bias;
	return  Activations.ApplySigmoid();
}

//Calculate the gradient descents for the network weights.
void Layer::Backpropagate(Matrix &error, Matrix &in) {
	BiasGradients = error * Activations.ApplySigmoidP();
	Gradients = (in.Transpose().Dot(BiasGradients)); //subtract instead of add
}

int Layer::Sign(double x)
{
	if (std::abs(x) < ZERO_TOLERANCE)
		return 0;
	if (x > 0)
		return 1;
	return -1;
}

double Layer::iResilientPlus(int i, int j, bool isWorse)
{
	int change = Sign(Gradients.matrix[i][j] * PrevGradients.matrix[i][j]);
	double weightChange = 0;
	if (change < 0)
	{
		LearningRates.matrix[i][j] = std::fmax(LearningRates.matrix[i][j] * NEGATIVE_ETA, MIN_STEP);
		if (isWorse)
			weightChange = -PrevUpdates.matrix[i][j];
		PrevGradients.matrix[i][j] = 0;
	}
	else {
		if (change > 0)
			LearningRates.matrix[i][j] = std::fmin(LearningRates.matrix[i][j] * POSITIVE_ETA, MAX_STEP);
		weightChange = Sign(Gradients.matrix[i][j]) * LearningRates.matrix[i][j];
		PrevGradients.matrix[i][j] = Gradients.matrix[i][j];
	}
	return weightChange;
}

double Layer::BiasiResilientPlus(int i, int j, bool isWorse)
{
	int change = Sign(BiasGradients.matrix[i][j] * BiasPrevGradients.matrix[i][j]);
	double weightChange = 0;
	if (change < 0)
	{
		BiasLearningRates.matrix[i][j] = std::fmax(BiasLearningRates.matrix[i][j] * NEGATIVE_ETA, MIN_STEP);
		if (isWorse)
			weightChange = -BiasPrevUpdates.matrix[i][j];
		BiasPrevGradients.matrix[i][j] = 0;
	}
	else {
		if (change > 0)
			BiasLearningRates.matrix[i][j] = std::fmin(BiasLearningRates.matrix[i][j] * POSITIVE_ETA, MAX_STEP);
		weightChange = Sign(BiasGradients.matrix[i][j]) * BiasLearningRates.matrix[i][j];
		BiasPrevGradients.matrix[i][j] = BiasGradients.matrix[i][j];
	}
	return weightChange;
}

void Layer::UpdateWeights(bool isWorse) {
	Matrix updates = Matrix(Gradients.rows, Gradients.columns);
	Matrix biasUpdates = Matrix(BiasGradients.rows, BiasGradients.columns);
	
	for (int i = 0; i < Gradients.rows; i++)
		for (int j = 0; j < Gradients.columns; j++)
			updates.matrix[i][j] = iResilientPlus(i, j, isWorse);

	for (int i = 0; i < BiasGradients.rows; i++)
		for (int j = 0; j < BiasGradients.columns; j++)
			biasUpdates.matrix[i][j] = BiasiResilientPlus(i, j, isWorse);
	
	PrevUpdates = updates;
	BiasPrevUpdates = biasUpdates;

	Weights = Weights + updates.MultiplyScalar(.0001);
	Bias = Bias + biasUpdates.MultiplyScalar(.0001);
}