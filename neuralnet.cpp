#include "neuralnet.h"

neuralnet::neuralnet()
{
	m_error = 0;
	m_recentAverageError = 0;
	m_learningrate = LEARNINGRATE;
	m_momentumweight = MOMENTUMWEIGHT;
	m_smoothingfactor = SMOOTHINGFACTOR;
}

neuralnet::~neuralnet()
{
	for (auto layer : m_layers)
		layer.clear();
	m_layers.clear();
}

bool neuralnet::initialize(const std::vector<int>& topology)
{
	m_error = 0;
	m_recentAverageError = 0;

	for (size_t i = 0; i < topology.size(); i++)
	{
		m_layers.push_back(Layer());
		int numOutputs = i == topology.size() - 1 ? 0 : topology[i + 1];

		for (int j = 0; j <= topology[i]; j++)
			m_layers.back().push_back(Neuron(numOutputs, j));

		m_layers.back().back().setOutputVal(1.0);
	}
	return true;
}

bool neuralnet::uninitialize()
{
	for (auto layer : m_layers)
		layer.clear();
	m_layers.clear();

	m_error = 0;
	m_recentAverageError = 0;

	return true;
}

bool neuralnet::feedforward(const std::vector<double>& input, std::vector<double>& output)
{
	bool result = false;
	if (input.size() == m_layers[0].size() - 1)
	{
		// Assign {latch} the input values into the input neurons
		for (int i = 0; i < input.size(); ++i)
			m_layers[0][i].setOutputVal(input[i]);

		// Forward propagate
		for (size_t layerNum = 1; layerNum < m_layers.size(); layerNum++)
		{
			Layer& prevLayer = m_layers[layerNum - 1];
			for (int i = 0; i < m_layers[layerNum].size() - 1; i++)
				m_layers[layerNum][i].feedForward(prevLayer);
		}

		output.clear();

		for (size_t n = 0; n < m_layers.back().size() - 1; n++)
			output.push_back(m_layers.back()[n].getOutputVal());

		result = true;
	}

	return result;
}

bool neuralnet::backprop(const std::vector<double>& targetVals, double* avg_error)
{
	bool result = false;

	if (avg_error)
	{
		// Calculate overal net error (RMS of output neuron errors)
		Layer& outputLayer = m_layers.back();
		m_error = 0.0;

		for (size_t n = 0; n < outputLayer.size() - 1; ++n)
		{
			double delta = targetVals[n] - outputLayer[n].getOutputVal();
			m_error += delta * delta;
		}
		m_error /= outputLayer.size() - 1;	// get average error squared
		m_error = sqrt(m_error);			// RMS

		// Implement a recent average measurement:
		m_recentAverageError = (m_recentAverageError * m_smoothingfactor + m_error) / (m_smoothingfactor + 1.0);

		// Calculate output layer gradients
		for (int n = 0; n < outputLayer.size() - 1; n++)
			outputLayer[n].calcOutputGradients(targetVals[n]);

		// Calculate gradients on hidden layers
		for (size_t layerNum = m_layers.size() - 2; layerNum > 0; layerNum--)
		{
			Layer& hiddenLayer = m_layers[layerNum];
			Layer& nextLayer = m_layers[layerNum + 1];

			for (int n = 0; n < hiddenLayer.size(); n++)
				hiddenLayer[n].calcHiddenGradients(nextLayer);
		}

		// For all layers from outputs to first hidden layer,
		// update connection weights
		for (size_t layerNum = m_layers.size() - 1; layerNum > 0; layerNum--)
		{
			Layer& layer = m_layers[layerNum];
			Layer& prevLayer = m_layers[layerNum - 1];

			for (int n = 0; n < layer.size() - 1; ++n)
				layer[n].updateInputWeights(prevLayer, m_learningrate, m_momentumweight);
		}

		*avg_error = m_recentAverageError;
	}
	return result;
}

bool neuralnet::settopology(const std::vector<int>& topology)
{
	for (auto layer : m_layers)
		layer.clear();
	m_layers.clear();

	m_error = 0;
	m_recentAverageError = 0;

	bool result = false;

	if (topology.size() > 1)
	{
		for (size_t i = 0; i < topology.size(); i++)
		{
			m_layers.push_back(Layer());
			int numOutputs = i == topology.size() - 1 ? 0 : topology[i + 1];

			for (int j = 0; j <= topology[i]; j++)
				m_layers.back().push_back(Neuron(numOutputs, j));

			m_layers.back().back().setOutputVal(1.0);
		}
		result = true;
	}

	return result;
}

bool neuralnet::gettopology(std::vector<int>& topology)
{
	topology.clear();
	for (auto layer : m_layers)
		topology.push_back((int)layer.size()-1);
	return true;
}

bool neuralnet::saveWeights(std::string filename)
{
	bool result = false;

	FILE* pFile = nullptr;

	fopen_s(&pFile, filename.c_str(), "w");

	if (pFile)
	{
		fprintf(pFile, "Topology: ");
		for (auto layer : m_layers)
			fprintf(pFile, "%d ", (int)layer.size());
		fprintf(pFile, "\n");

		for (auto layer : m_layers)
		{
			for (auto node : layer)
			{
				for (int wi = 0; wi < node.getnumberofoutputs(); wi++)
				{
					fprintf(pFile, "%f ", node.getweight(wi));
				}
				fprintf(pFile, "\n");
			}
		}

		fclose(pFile);
		result = true;
	}
	return result;

}

bool neuralnet::loadWeights(std::string filename)
{
	std::ifstream m_weightsfile;
	m_weightsfile.open(filename.c_str());
	if (m_weightsfile.fail())
		return false;

	std::string line;
	std::string label;

	getline(m_weightsfile, line);
	std::stringstream ss(line);
	ss >> label;
	if (label.compare("Topology:") != 0)
		return false;

	int n;
	int li = 0;

	for (auto layer : m_layers)
	{
		ss >> n;
		if (n != layer.size())
			return false;	// invalid neuralnet size
	}

	float weight = 0;
	for (auto layer : m_layers)
	{
		for(auto node : layer)
		{
			getline(m_weightsfile, line);
			std::stringstream ss(line);
			for (int wi = 0; wi < node.getnumberofoutputs(); wi++)
			{
				ss >> weight;
				node.setweight(wi, weight);
			}
		}
	}
	return true;
}

