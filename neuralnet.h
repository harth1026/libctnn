#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

#include "neuron.h"

#define LEARNINGRATE	0.15f
#define MOMENTUMWEIGHT	0.5f
#define SMOOTHINGFACTOR	100.0f

class neuralnet
{
private:
	std::vector<Layer> m_layers;
	double m_error;
	double m_recentAverageError;

	double m_learningrate;
	double m_momentumweight;
	double m_smoothingfactor;

public:
	neuralnet();
	~neuralnet();

	bool initialize(const std::vector<int>& topology);
	bool uninitialize();
	bool feedforward(const std::vector<double>& input, std::vector<double>& output);
	bool backprop(const std::vector<double>& targetVals, double* avg_error);

	bool settopology(const std::vector<int>& topology);
	bool gettopology(std::vector<int>& topology);
	bool saveWeights(std::string filename);
	bool loadWeights(std::string filename);

	double getlearningrate() { return m_learningrate; }
	void setlearningrate(double value) { m_learningrate = value; }
	double getmomentumweight() { return m_momentumweight; }
	void setmomentumweight(double value) { m_momentumweight = value; }
	double getsmoothingfactor() { return m_smoothingfactor; }
	void setsmoothingfactor(double value) { m_smoothingfactor = value; }
};

