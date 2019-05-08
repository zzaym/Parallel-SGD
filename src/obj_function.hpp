#include <vector>

/*
    Linear Regression (least squares): 
    Input: X, Y
    Parameter: W
    output:
        loss = 0.5 * \sum_{i=1}^n r_i^2 = 0.5 * \sum_{i=1}^n (y_i - w^Tx_i)^2
        gradient = X^T(XW-Y)
*/
double linear_reg_obj(const std::vector<double>& weights, \
        const std::vector< std::vector<double> >& X, \
        const std::vector<double>& Y);

std::vector<double> linear_reg_obj_grad(const std::vector<double>& weights, \
                    const std::vector< std::vector<double> >& X, \
                    const std::vector<double>& Y);