/*
	Parallel version of SGD with OpenMP
*/

#include "parallel_sgd.hpp"
#include <iostream>
#include <algorithm>
#include <functional> // -std=c++11
#include <omp.h>


parallel_sgd::parallel_sgd(double (*obj)(const std::vector<double>&, \
		const std::vector< std::vector<double> >&, const std::vector<double>&), \
		std::vector<double> (*obj_grad)(const std::vector<double>&, \
		const std::vector< std::vector<double> >&, const std::vector<double>&), \
		const std::vector<double>& init_weights,
		const std::vector< std::vector<double> >& X,\
		const std::vector<double>& Y, \
		const float lr, \
		const unsigned int num_iters, \
		const unsigned int num_batches, \
		const unsigned int num_threads) \
		: seq_sgd(obj, obj_grad, init_weights, X, Y, lr, num_iters, num_batches)
{
	k = num_threads;
}


void parallel_sgd::update(const unsigned int print_every) {
	// ensure the number of iterations for convergence guarantee
	unsigned int T = static_cast<int>(m/k) + 1;
	if (num_iters < T)
		num_iters = T;

	// declare new weights for parallelism
	// if necessary, do initilization
	std::vector< std::vector<double> > parallel_weights(k, std::vector<double>(d));
	std::vector<double> new_weights(d);

	// openmp for parallelism
	# pragma omp parallel for num_threads(k) 
	for (unsigned int j = 0; j < k; j++) {
		for (unsigned int i = 0; i < num_iters; ++i) {
			int index = std::rand() % m;

			// compute gradient
			std::vector<double> weights_grad(gradient(parallel_weights[j], Xs[index], Ys[index]));

			// multiply by lr
			std::transform(weights_grad.begin(), weights_grad.end(), weights_grad.begin(), \
				std::bind(std::multiplies<double>(), std::placeholders::_1, -lr));

			// descent
			std::transform(parallel_weights[j].begin(), parallel_weights[j].end(), \
				weights_grad.begin(), parallel_weights[j].begin(), std::plus<double>());

			// verbose info
			// if (print_every > 0 && i % print_every == 0) {
			// 	std::cout << "Thread " << omp_get_thread_num() << ", Iteration " << i \
			// 			<< ", loss: " << get_loss() << std::endl;
			// }
		}

		// reduction
		for (unsigned int i = 0; i < d; i++)
			new_weights[i] += parallel_weights[j][i]/k;
	}

	// assgin back
	weights = new_weights;
} 