#pragma once

#include <vector>

class Matrix
{
public:
	Matrix();
	Matrix(int rows, int columns);
	Matrix(std::vector<std::vector<double> > const &matrix);

	int rows;
	int columns;
	std::vector<std::vector<double> > matrix;

	Matrix operator+(Matrix const &matrix2);
	Matrix operator-(Matrix &matrix2);
	Matrix operator*(Matrix const &matrix2);
	Matrix Dot(Matrix const &matrix2);
	Matrix MultiplyScalar(double const &scalar);
	Matrix operator^(int const exponent);
	double Sum();
	Matrix Transpose();
	Matrix Step();
	void PrintMatrix();

	Matrix ApplyRandomize();
	Matrix ApplyHyperbolic();
	Matrix ApplyHyperbolicP();
	Matrix ApplySigmoid();
	Matrix ApplySigmoidP();

	~Matrix();
private:
	double DotProduct(Matrix const &matrix2, int &m1Row, int &m2Col);

	double Randomize(double x);
	double HyperbolicTangent(double x);
	double HyperbolicDerivative(double x);
	double Sigmoid(double x);
	double SigmoidDerivative(double x);
};

