#pragma once

// interface for LIBCTNN

#include <vector>
#include <string>

#ifdef WIN32
#ifdef	__cplusplus
#define _DLLFUNC	extern "C" __declspec(dllexport)
#else
#define _DLLFUNC	__declspec(dllexport)
#endif
#else	// WIN32
#define _DLLFUNC	
#endif

#if defined(_MSC_VER)
#ifdef LIBCTNN_EXPORTS
#define _DLLSTDCALL
#else
#define _DLLSTDCALL __declspec(dllimport)
#endif
#elif defined(__GNUC__)
	//  GCC
#ifdef LIBCTNN_EXPORTS
#define _DLLSTDCALL __attribute__ ((visibility("default")))
#else
#define _DLLSTDCALL 
#endif
#else
	//  do nothing and hope for the best?
#define _DLLSTDCALL 
#pragma warning Unknown dynamic link export/import semantics.
#endif

enum NNPROPID
{
	PROPID_LEARNINGRATE = 1,
	PROPID_MOMENTUMWEIGHT = 2,
	PROPID_SMOOTHINGFACTOR = 3,
};
	
// MANAGEMENT
_DLLFUNC bool ctnn_initialize(const std::vector<int>& topology);
_DLLFUNC bool ctnn_settopology(const std::vector<int>& topology);
_DLLFUNC bool ctnn_gettopology(std::vector<int>& topology);
_DLLFUNC bool ctnn_saveWeights(std::string filename);
_DLLFUNC bool ctnn_loadWeights(std::string filename);

_DLLFUNC bool ctnn_setproperty(int propid, double value);
_DLLFUNC bool ctnn_getproperty(int propid, double* value);

_DLLFUNC bool ctnn_feedforward(const std::vector<double>& input, std::vector<double>& output);
_DLLFUNC bool ctnn_backprop(const std::vector<double>& targetVals, double* avg_error);

