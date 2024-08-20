#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

using namespace std;

int main()
{
    // Dataset
    vector<float> x1 = {0.5, 0.8, 0.1, 0.7, 0.2, 0.4, 0.1, 0.9, 0.6, 0.8}; // x coordinates of all data points
    vector<float> x2 = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1}; // y coordinates of all data points
    vector<float> x3 = {0.7, 0.3, 0.4, 0.2, 0.5, 0.6, 0.9, 0.1, 0.8, 0.4}; // z coordinates of all data points
    vector<int> y = {1, 1, -1, 1, 1, -1, -1, -1, 1, 1};                    // class labels of all data points
    float c = 5;

    cout << "// Dataset: " << endl;
    cout << "x1" << '\t' << "x2" << '\t' << "x3" << '\t' << "y" << endl;
    for (int i = 0; i < 10; i++)
    {
        cout << x1[i] << '\t' << x2[i] << '\t' << x3[i] << '\t' << y[i] << endl;
    }
    cout << endl;
    cout << "c = " << c << endl;
    cout << endl;

    // LINGO input code script
    cout << "// LINGO input code script: " << endl
         << endl;
    cout << "MIN = 0.5 * ( ";

    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            cout << "lambda_" << i + 1 << " * lambda_" << j + 1 << " * " << y[i] * y[j] * (x1[i] * x1[j] + x2[i] * x2[j] + x3[i] * x3[j]);
            if (i != 9 || j != 9)
                cout << " + ";
        }
    }
    cout << " ) - ( ";

    for (int i = 0; i < 10; i++)
    {
        cout << "lambda_" << i + 1;
        if (i < 9)
            cout << " + ";
    }
    cout << " );" << endl
         << endl;

    for (int i = 0; i < 10; i++)
    {
        cout << "lambda_" << i + 1 << " * " << y[i];
        if (i < 9)
            cout << " + ";
    }
    cout << " = 0;" << endl
         << endl;

    for (int i = 0; i < 10; i++)
    {
        cout << "@BND(0, lambda_" << i + 1 << ", " << c << ");" << endl;
    }
    cout << endl;

    // LINGO result
    cout << "// LINGO result: " << endl;
    vector<float> lambda = {4.464285, 0.7162082E-06, 5.0, 5.0, 5.0, 5.0, 4.464285, 5.0, 5.0, 0.2819476E-07};
    
    for (int i = 0; i < 10; i++)
    {
        cout << "lambda_" << i + 1 << " = " << lambda[i] << endl;
    }
    cout << endl;

    lambda = {4.464285, 0.0, 5.0, 5.0, 5.0, 5.0, 4.464285, 5.0, 5.0, 0.0}; // Rounding-off

    // Post-processing of LINGO result
    cout << "// Post-processing of LINGO result: " << endl;
    vector<float> w = {0, 0, 0};
    float b, b_plus, b_minus, best_b_plus = -numeric_limits<float>::max(), best_b_minus = numeric_limits<float>::max();

    for (int i = 0; i < 10; i++)
    {
        w[0] += lambda[i] * y[i] * x1[i];
        w[1] += lambda[i] * y[i] * x2[i];
        w[2] += lambda[i] * y[i] * x3[i];
    }

    for (int i = 0; i < 3; i++)
    {
        cout << "w" << i + 1 << " = " << w[i] << endl;
    }

    for (int i = 0; i < 10; i++)
    {
        if (lambda[i] > 0 && lambda[i] < c)
        {
            if (y[i] == 1)
            {
                b_plus = 1 - (w[0] * x1[i] + w[1] * x2[i] + w[2] * x3[i]);
                if (b_plus > best_b_plus)
                    best_b_plus = b_plus;
            }
            else if (y[i] == -1)
            {
                b_minus = -1 - (w[0] * x1[i] + w[1] * x2[i] + w[2] * x3[i]);
                if (b_minus < best_b_minus)
                    best_b_minus = b_minus;
            }
        }
    }

    b = 0.5 * (best_b_plus + best_b_minus);
    cout << "b = " << b << endl;

    vector<float> q;
    float q_i;
    for (int i = 0; i < 10; i++)
    {
        if (lambda[i] < c)
        {
            q.push_back(0);
        }
        else
        {
            q_i = 1 - y[i] * (w[0] * x1[i] + w[1] * x2[i] + w[2] * x3[i] + b);
            q.push_back(q_i);
        }
    }

    for (int i = 0; i < 10; i++)
    {
        cout << "q" << i + 1 << " = " << q[i] << endl;
    }

    return 0;
}