//
// Created by  Valentina Smirnova on 25.05.2022.
//

#ifndef VSPEED_REGRESSION_H
#define VSPEED_REGRESSION_H

#include <iostream>
#include <cstdio>
#include <vector>
#include "MapFrames.h"

using namespace std;

class Regression {
    // Dynamic array which is going
    // to contain all (i-th x)
    std::vector<float> x;

    // Dynamic array which is going
    // to contain all (i-th y)
    std::vector<float> y;

    // Store the coefficient/slope in
    // the best fitting line
    float coeff;

    // Store the constant term in
    // the best fitting line
    float const_term;

    // Contains sum of product of
    // all (i-th x) and (i-th y)
    float sum_xy;

    // Contains sum of all (i-th x)
    float sum_times;

    // Contains sum of all (i-th y)
    float sum_coords;

    // Contains sum of square of
    // all (i-th x)
    float sum_x_square;

    // Contains sum of square of
    // all (i-th y)
    float sum_y_square;

public:
    // Constructor to provide the default
    // values to all the terms in the
    // object of class regression
    Regression() : x({}), y({}), coeff(0.0), const_term(0.0), sum_xy(0.0),
                   sum_times(0.0), sum_coords(0.0), sum_x_square(0.0),
                   sum_y_square(0.0) {};

    Regression(const std::vector<CutFrame>& vec, char dim);

    // Function that calculate the coefficient/
    // slope of the best fitting line
    void calculate_coefficient();

    // Member function that will calculate
    // the constant term of the best
    // fitting line
    void calculate_constant_term();

    // Function that return the number
    // of entries (xi, yi) in the data set
    int size_of_data();

    // Function that return the coeffecient/
    // slope of the best fitting line
    float coefficient();

    // Function that return the constant
    // term of the best fitting line
    float constant();

    // Function that print the best
    // fitting line
    void print_best_fitting_line();

    // Function to take input from the dataset
    void take_input(int n);


    // Function to show the data set
    void show_data();

    // Function to predict the value
    // correspondng to some input
    float predict(float x);


    // Function that returns overall
    // sum of square of errors
    float error_square();

    // Functions that return the error
    // i.e the difference between the
    // actual value and value predicted
    // by our model
    float error_in(float num);
};


#endif //VSPEED_REGRESSION_H
