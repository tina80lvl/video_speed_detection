#include "Regression.h"

Regression::Regression(const std::vector<CutFrame>& vec, char dim) {
    int c;
    if (dim == 'x') c = 0;

    for (const auto& cut_frame : vec) {
        float ftime = cut_frame.time.get_float();
        float coord;
        if (dim == 'x') coord = cut_frame.coords.x;
        if (dim == 'y') coord = cut_frame.coords.y;
        if (dim == 'z') coord = cut_frame.coords.zl;

        sum_xy += ftime * coord;
        sum_times += ftime;
        sum_coords += coord;
        sum_x_square += ftime * ftime;
        sum_y_square += coord * coord;
        x.push_back(ftime);
        y.push_back(coord);
    }
}


// Function that calculate the coefficient/
// slope of the best fitting line
void Regression::calculate_coefficient() {
    float N = x.size();
    float numerator
            = (N * sum_xy - sum_times * sum_coords);
    float denominator
            = (N * sum_x_square - sum_times * sum_times);
    coeff = numerator / denominator;
}

// Member function that will calculate
// the constant term of the best
// fitting line
void Regression::calculate_constant_term() {
    float N = x.size();
    float numerator
            = (sum_coords * sum_x_square - sum_times * sum_xy);
    float denominator
            = (N * sum_x_square - sum_times * sum_times);
    const_term = numerator / denominator;
}

// Function that return the number
// of entries (xi, yi) in the data set
int Regression::size_of_data() {
    return x.size();
}

// Function that return the coeffecient/
// slope of the best fitting line
float Regression::coefficient() {
    if (coeff == 0)
        calculate_coefficient();
    return coeff;
}

// Function that return the constant
// term of the best fitting line
float Regression::constant() {
    if (const_term == 0)
        calculate_constant_term();
    return const_term;
}

// Function that print the best
// fitting line
void Regression::print_best_fitting_line() {
    if (coeff == 0 && const_term == 0) {
        calculate_coefficient();
        calculate_constant_term();
    }
    cout << "The best fitting line is y = "
         << coeff << "x + " << const_term << endl;
}

// Function to take input from the dataset
void Regression::take_input(int n) {
    for (int i = 0; i < n; i++) {
        // In a csv file all the values of
        // xi and yi are separated by commas
        char comma;
        float xi;
        float yi;
        cin >> xi >> comma >> yi;
        sum_xy += xi * yi;
        sum_times += xi;
        sum_coords += yi;
        sum_x_square += xi * xi;
        sum_y_square += yi * yi;
        x.push_back(xi);
        y.push_back(yi);
    }
}

// Function to show the data set
void Regression::show_data() {
    for (int i = 0; i < 62; i++) {
        printf("_");
    }
    printf("\n\n");
    printf("|%15s%5s %15s%5s%20s\n",
           "X", "", "Y", "", "|");

    for (int i = 0; i < x.size(); i++) {
        printf("|%20f %20f%20s\n",
               x[i], y[i], "|");
    }

    for (int i = 0; i < 62; i++) {
        printf("_");
    }
    printf("\n");
}

// Function to predict the value
// correspondng to some input
float Regression::predict(float x) {
    return coeff * x + const_term;
}

// Function that returns overall
// sum of square of errors
float Regression::error_square() {
    float ans = 0;
    for (int i = 0; i < x.size(); ++i) {
        ans += ((predict(x[i]) - y[i])
                * (predict(x[i]) - y[i]));
    }
    return ans;
}

// Functions that return the error
// i.e the difference between the
// actual value and value predicted
// by our model
float Regression::error_in(float num) {
    for (int i = 0; i < x.size(); ++i) {
        if (num == x[i]) {
            return (y[i] - predict(x[i]));
        }
    }
    return 0;
}

