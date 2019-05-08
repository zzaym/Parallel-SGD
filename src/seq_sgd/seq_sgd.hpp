/*
    Sequential version of SGD
*/

#ifndef SSGD_H
#define SSGD_H


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
        const unsigned int num_iters=1000, \
        const unsigned int num_batches=64);

    // clone
    seq_sgd* clone() const;

    // destructor
    ~seq_sgd();

    // functions
    // perform SGD
    virtual void update(const unsigned int print_every=0);

    // get weights
    std::vector<double> get_weights();

    // get loss
    double get_loss();

protected:
    // variables
    std::vector< std::vector< std::vector<double> > > Xs;
    std::vector< std::vector<double> > Ys;
    std::vector<double> weights;
    float lr;
    unsigned int num_iters;
    unsigned int m; // the number of batch
    unsigned int N; // the number of input samples
    unsigned int d; // the dims of input
    unsigned int b; // the batch size
    double loss;
    // may be discarded if no memory
    std::vector< std::vector<double> > X; 
    std::vector<double> Y;

    // functions
    double (*objective)(const std::vector<double>&, \
        const std::vector< std::vector<double> >&, const std::vector<double>&);
    std::vector<double> (*gradient)(const std::vector<double>&, \
        const std::vector< std::vector<double> >&, const std::vector<double>&);
};

#endif