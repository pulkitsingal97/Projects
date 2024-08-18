
#ifndef SPORTSLAYOUT_H
#define SPORTSLAYOUT_H

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
using namespace std;

extern chrono::high_resolution_clock::time_point startTime;
extern chrono::high_resolution_clock::time_point currentTime;

class SportsLayout
{

private:
    int l;
    int **T;
    int **N;

public:
    int z;
    float time;
    float loop_time;
    int *mapping;

    vector<int> global_opt_state;
    long long global_opt_state_cost;

    SportsLayout(string inputfilename);

    bool check_output_format();

    // void readOutputFile(string output_filename);

    long long cost_fn();

    void write_to_file(string outputfilename);

    void readInInputFile(string inputfilename);

    void compute_allocation();

    pair<vector<int>, long long> compute_best_neighbour(vector<int> curr_state, long long curr_state_cost, vector<int> all_locs);

    long long state_cost_fn(vector<int> state);
};

#endif