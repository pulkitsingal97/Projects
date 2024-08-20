#include <iostream>
#include <vector>

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
    cout << "MIN = 0.5 * ( w1 * w1 + w2 * w2 + w3 * w3) + " << c << " * ( q1 + q2 + q3 + q4 + q5 + q6 + q7 + q8 + q9 + q10 );" << endl
         << endl;

    for (int i = 0; i < 10; i++)
    {
        cout << y[i] << " * ( w1 * " << x1[i] << " + w2 * " << x2[i] << " + w3 * " << x3[i] << " + "
             << "b ) + q" << i + 1 << " >= 1;" << endl;
    }
    cout << endl;

    cout << "@FREE(w1);" << endl;
    cout << "@FREE(w2);" << endl;
    cout << "@FREE(w3);" << endl;
    cout << "@FREE(b);" << endl;
    cout << endl;

    // LINGO result
    cout << "// LINGO result: " << endl;
    cout << "w1 = " << 2.285714 << endl;
    cout << "w2 = " << -2.178571 << endl;
    cout << "w3 = " << 1.107143 << endl;
    cout << "b = " << -0.7 << endl;

    vector<float> q = {0.0, 0.0, 0.3178571, 0.75, 1.778571, 0.5714286, 0.0, 0.725, 1.403571, 0.0};
    for (int i = 0; i < 10; i++)
    {
        cout << "q" << i + 1 << " = " << q[i] << endl;
    }

    return 0;
}