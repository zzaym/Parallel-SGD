/*
	Sequential version of SGD
*/

#include "seq_sgd.hpp"
#include <iostream>
#include <functional> // -std=c++11

// constructor
seq_sgd::seq_sgd(double (*obj)(const std::vector<double>&, \
			const std::vector< std::vector<double> >&, const std::vector<double>&), \
			std::vector<double> (*obj_grad)(const std::vector<double>&, \
			const std::vector< std::vector<double> >&, const std::vector<double>&), \
			const std::vector<double>& init_weights, \
			const std::vector< std::vector<double> >& X,\
			const std::vector<double>& Y, \
			const float lr, \
			const unsigned int num_iters, \
			const unsigned int num_batches) {
	objective = obj;
	gradient = obj_grad;
	weights = init_weights;
	this->lr = lr;
	this->num_iters = num_iters;
	this->num_batches = num_batches;
	// may be discarded if no memory
	this->X = X;
	this->Y = Y;

	// shuffle indices
	int N=X.size(), batch_size = static_cast<int>(N/num_batches), d = X[0].size(), index = 0;
	std::vector<int> indices(N);
	for (int i = 0; i < N; i++)
		indices[i] = i;
	std::random_shuffle(indices.begin(), indices.end());

	// init batches
	Xs = std::vector< std::vector< std::vector<double> > >(num_batches, \
						std::vector< std::vector<double> >(batch_size, std::vector<double>(d)));
	Ys = std::vector< std::vector<double> >(num_batches, std::vector<double>(batch_size));
	for (int i = 0; i < num_batches; i++) {
		for (int j = 0; j < batch_size; j++) {
			for (int k = 0; k < d; k++) 
				Xs[i][j][k] = X[indices[index]][k];
			Ys[i][j] = Y[indices[index]];
			index++;
		}
	}
	
}


seq_sgd* seq_sgd::clone() const {
	return new seq_sgd(*this);
}

seq_sgd::~seq_sgd(){}


void seq_sgd::update(const unsigned int print_every) {
	for (int i = 0; i < num_iters; ++i) {
		int index = std::rand() % num_batches;

		// compute gradient
		std::vector<double> weights_grad(gradient(weights, Xs[index], Ys[index]));

		// multiply by lr
		std::transform(weights_grad.begin(), weights_grad.end(), weights_grad.begin(), \
			std::bind(std::multiplies<double>(), std::placeholders::_1, -lr));

		// descent
		std::transform(weights.begin(), weights.end(), \
			weights_grad.begin(), weights.begin(), std::plus<double>());

		// verbose info
		if (print_every > 0 && i % print_every == 0) {
			std::cout << "Iteration " << i << ", loss: " << get_loss() << std::endl;
		}
	}
} 


std::vector<double> seq_sgd::get_weights() {
	return weights;
}


double seq_sgd::get_loss() {
	loss = objective(weights, X, Y);
	return loss;
}
