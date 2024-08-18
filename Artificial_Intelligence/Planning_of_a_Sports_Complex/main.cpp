#include <iostream>
#include <bits/stdc++.h>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <random>
#include <cstdlib>
#include <ctime>

#include "SportsLayout.h"

using namespace std;

chrono::high_resolution_clock::time_point startTime;
chrono::high_resolution_clock::time_point currentTime;

int main(int argc, char **argv)
{
    startTime = chrono::high_resolution_clock::now();

    // Parse the input.
    if (argc < 3)
    {
        cout << "Missing arguments\n";
        cout << "Correct format : \n";
        cout << "sh run.sh <input_filename> <output_filename>";
        exit(0);
    }
    string inputfilename(argv[1]);
    string outputfilename(argv[2]);

    SportsLayout *s = new SportsLayout(inputfilename);

    s->compute_allocation();

    cout << "Final Global Optimal State Details = ";
    for (int i = 0; i < s->z; i++)
    {
        s->mapping[i] = s->global_opt_state[i];
        cout << s->global_opt_state[i] << " ";
    }
    cout << ": " << s->global_opt_state_cost << endl
         << endl;

    s->write_to_file(outputfilename);

    long long cost = s->cost_fn();
    cout << "cost: " << cost << endl;

    currentTime = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsedTime = currentTime - startTime;
    cout << "Total Running Time (seconds) = " << elapsedTime.count() << endl
         << endl;

    return 0;
}