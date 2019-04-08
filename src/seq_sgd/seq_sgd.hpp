/*
	Sequential version of SGD
*/

#include <vector>


class seq_sgd {
public:
	// constructor
	seq_sgd();
	seq_sgd(double (*obj)(const std::vector<double>&, \
		const std::vector< std::vector<double> >&, const std::vector<double>&), \
		std::vector<double> (*obj_grad)(const std::vector<double>&, \
		const std::vector< std::vector<double> >&, const std::vector<double>&), \
		const std::vector<double>& init_weights,
		const std::vector< std::vector<double> >& X,\
		const std::vector<double>& Y, \
		const float lr=0.01, \
		const unsigned int num_iters=1000);
	
	// copy constructor
	seq_sgd* clone() const;

	// destructor
	~seq_sgd();

	// functions
	// perform SGD
	void update(const unsigned int print_every=0);

	// get weights
	std::vector<double> get_weights();

	// get loss
	double get_loss();

private:
	// variables
	std::vector<double> weights;
	std::vector< std::vector<double> > X;
	std::vector<double> Y;
	float lr;
	unsigned int num_iters;
	double loss;

	// functions
	double (*objective)(const std::vector<double>&, \
		const std::vector< std::vector<double> >&, const std::vector<double>&);
	std::vector<double> (*gradient)(const std::vector<double>&, \
		const std::vector< std::vector<double> >&, const std::vector<double>&);
};