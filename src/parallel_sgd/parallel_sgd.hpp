/*
	Parallel version of SGD with OpenMP
*/

#ifndef PSGD_H
#define PSGD_H


#include <vector>
#include "seq_sgd/seq_sgd.hpp"

class parallel_sgd : public seq_sgd{
public:
	parallel_sgd();
	parallel_sgd(double (*obj)(const std::vector<double>&, \
		const std::vector< std::vector<double> >&, const std::vector<double>&), \
		std::vector<double> (*obj_grad)(const std::vector<double>&, \
		const std::vector< std::vector<double> >&, const std::vector<double>&), \
		const std::vector<double>& init_weights,
		const std::vector< std::vector<double> >& X,\
		const std::vector<double>& Y, \
		const float lr=0.01, \
		const unsigned int num_iters=1000, \
		const unsigned int num_batches=64, \
		const unsigned int num_threads=4);

	// functions
	// perform parallel SGD
	void update(const unsigned int print_every=0);

protected:
	unsigned int k; // the number of threads
};

#endif