#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

using namespace std;

int main()
{
    // Dataset
    vector<float> x1 = {1, 5, 8, 9, 6, 5, 2, 4, 3, 8};   // x coordinates of all data points
    vector<float> x2 = {2, 4, 6, 5, 8, 4, 6, 7, 5, 1};   // y coordinates of all data points
    vector<float> x3 = {3, 2, 2, 7, 6, 5, 4, 1, 8, 6};   // z coordinates of all data points
    vector<int> y = {1, 1, 1, 1, 1, -1, -1, -1, -1, -1}; // class labels of all data points
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
    vector<float> lambda = {5.0, 5.0, 0.7964247, 4.565516, 2.24917, 5.0, 2.210601, 5.0, 0.40051, 5.0};

    for (int i = 0; i < 10; i++)
    {
        cout << "lambda_" << i + 1 << " = " << lambda[i] << endl;
    }
    cout << endl;

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