
#include "libctnn.h"
#include "neuralnet.h"

static neuralnet* nn = nullptr;

bool ctnn_initialize(const std::vector<int>& topology)
{
	if (nn)
	{
		delete nn;
		nn = nullptr;
	}

	nn = new neuralnet();

	return nn->initialize(topology);
}

bool ctnn_feedforward(const std::vector<double>& input, std::vector<double>& output)
{
	bool result = false;
	if (nn)
	{
		result = nn->feedforward(input, output);
	}
	return result;
}

bool ctnn_backprop(const std::vector<double>& targetVals, double* avg_error)
{
	bool result = false;
	if (nn && avg_error)
	{
		result = nn->backprop(targetVals, avg_error);
	}

	return result;
}

bool ctnn_settopology(const std::vector<int>& topology)
{
	bool result = false;
	if (nn)
	{
		nn->settopology(topology);
		result = true;
	}
	return result;
}

bool ctnn_gettopology(std::vector<int>& topology)
{
	bool result = false;
	if (nn)
	{
		nn->gettopology(topology);
		result = true;
	}

	return result;
}

bool ctnn_saveWeights(std::string filename)
{
	bool result = false;
	if (nn)
	{
		result = nn->saveWeights(filename);
	}

	return result;
}

bool ctnn_loadWeights(std::string filename)
{
	bool result = false;
	if (nn)
	{
		result = nn->loadWeights(filename);
	}

	return result;
}

bool ctnn_setproperty(int propid, double value)
{
	bool result = false;
	if (nn)
	{
		switch (propid)
		{
		case PROPID_LEARNINGRATE:
			nn->setlearningrate(value);
			result = true;
			break;
		case PROPID_MOMENTUMWEIGHT:
			nn->setmomentumweight(value);
			result = true;
			break;
		case PROPID_SMOOTHINGFACTOR:
			nn->setsmoothingfactor(value);
			result = true;
			break;
		}
	}

	return result;
}

bool ctnn_getproperty(int propid, double* value)
{
	bool result = false;
	if (nn && value)
	{
		switch (propid)
		{
		case PROPID_LEARNINGRATE:
			*value = nn->getlearningrate();
			result = true;
			break;
		case PROPID_MOMENTUMWEIGHT:
			*value = nn->getmomentumweight();
			result = true;
			break;
		case PROPID_SMOOTHINGFACTOR:
			*value = nn->getsmoothingfactor();
			result = true;
			break;
		}
	}

	return result;
}

