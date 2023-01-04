#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

void Matrix_transpose(const float *X, float *output, size_t n, size_t m) {
    for (size_t i = 0; i < n; i ++)
        for (size_t j = 0; j < m; j ++)
            output[j*n+i] = X[i*m+j];
}

void Matrix_Mul(const float *X, const float *Y, float* output, 
                size_t n, size_t p, size_t m) 
{
    for (size_t i = 0; i < n; i ++)
        for (size_t j = 0; j < m; j ++){
            output[i*m+j] = 0;
            for (size_t k = 0; k < p; k ++)
                output[i*m+j] += X[i*p+k] * Y[k*m+j];
        }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    int num_iter = ceil((float)m / batch);
    printf("num_iter: %d\n", num_iter);
    for (int iter = 0; iter < num_iter; iter ++) {
        float *X_batch = new float[batch*n];
        unsigned char *y_batch = new unsigned char[batch];
        for (size_t i = iter*batch; i < (size_t)((iter+1)*batch); i ++) {
            for (size_t j = 0; j < n; j ++) 
                X_batch[(i-iter*batch)*n+j] = X[i*n+j];
            y_batch[i-iter*batch] = y[i];
        }
        float *exp_xtheta = new float[batch*k];
        Matrix_Mul(X_batch, theta, exp_xtheta, batch, n, k);
        for (size_t i = 0; i < batch; i ++)
            for (size_t j = 0; j < k; j ++)
                exp_xtheta[i*k+j] = exp(exp_xtheta[i*k+j]);
        float *Z = new float[batch*k];
        for (size_t i = 0; i < batch; i ++) {
            float sum = 0;
            for (size_t j = 0; j < k; j ++) sum += exp_xtheta[i*k+j];
            for (size_t j = 0; j < k; j ++) {
                Z[i*k+j] = exp_xtheta[i*k+j] / sum;
                if (j == y_batch[i]) Z[i*k+j] -= 1;
            }
        }
        float *grad = new float[n*k];
        float *X_T = new float[n*batch];
        Matrix_transpose(X, X_T, batch, n);
        Matrix_Mul(X_T, Z, grad, n, batch, k);
        for (size_t i = 0; i < n; i ++)
            for (size_t j = 0; j < k; j ++)
                theta[i*k+j] -= grad[i*k+j] / batch *lr;
        float norm = 0;
        for (size_t i = 0; i < n; i ++)
            for (size_t j = 0; j < k; j ++)
                norm += (theta[i*k+j]*theta[i*k+j]);
        printf("theta norm: %f\n", sqrt(norm));
        delete[] X_batch;
        delete[] y_batch;
        delete[] exp_xtheta;
        delete[] Z;
        delete[] grad;
        delete[] X_T;
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
