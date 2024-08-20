#include <iostream>
#include <vector>

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
    cout << "w1 = " << 0.3333333 << endl;
    cout << "w2 = " << 0.3333333 << endl;
    cout << "w3 = " << 0.0 << endl;
    cout << "b = " << -3.666667 << endl;

    vector<float> q = {3.666667, 1.666667, 0.0, 0.0, 0.0, 0.3333333, 0.0, 1.000000, 0.0, 0.3333333};
    for (int i = 0; i < 10; i++)
    {
        cout << "q" << i + 1 << " = " << q[i] << endl;
    }

    return 0;
}