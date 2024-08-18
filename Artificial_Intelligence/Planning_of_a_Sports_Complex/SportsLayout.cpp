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

#include "SportsLayout.h"

SportsLayout::SportsLayout(string inputfilename)
{

    readInInputFile(inputfilename);
    mapping = new int[z];
}

bool SportsLayout::check_output_format()
{

    vector<bool> visited(l, false);
    for (int i = 0; i < z; i++)
    {
        if ((mapping[i] >= 1 && mapping[i] <= l))
        {
            if (!visited[mapping[i] - 1])
                visited[mapping[i] - 1] = true;
            else
            {
                cout << "Repeated locations, check format\n";
                return false;
            }
        }
        else
        {
            cout << "Invalid location, check format\n";
            return false;
        }
    }

    return true;
}

// void SportsLayout::readOutputFile(string output_filename)
// {
//         fstream ipfile;
//         ipfile.open(output_filename, ios::in);
//         if (!ipfile) {
//             cout << "No such file\n";
//             exit( 0 );
//         }
//         else {

//             vector<int> ip;

//             while (1) {
//                 int t;
//                 ipfile >> t;
//                 ip.push_back(t);
//                 if (ipfile.eof())
//                     break;

//             }

//         if(ip.size()!=z)
//         {
//             cout<<"number of values not equal to number of zones, check output format\n";
//             exit(0);
//         }
//         for(int i=0;i<z;i++)
//         mapping[i]=ip[i];
//     ipfile.close();

//     if(!check_output_format())
//         exit(0);
//     cout<<"Read output file, format OK"<<endl;

//         }

// }

long long SportsLayout::cost_fn()
{

    long long cost = 0;

    for (int i = 0; i < z; i++)
    {
        for (int j = 0; j < z; j++)
        {
            cost += (long long)N[i][j] * (long long)T[mapping[i] - 1][mapping[j] - 1];
        }
    }

    return cost;
}

long long SportsLayout::state_cost_fn(vector<int> state)
{

    long long cost = 0;

    for (int i = 0; i < z; i++)
    {
        for (int j = 0; j < z; j++)
        {
            cost += (long long)N[i][j] * (long long)T[state[i] - 1][state[j] - 1];
        }
    }

    return cost;
}

void SportsLayout::readInInputFile(string inputfilename)
{
    fstream ipfile;
    ipfile.open(inputfilename, ios::in);
    if (!ipfile)
    {
        cout << "No such file\n";
        exit(0);
    }
    else
    {

        ipfile >> time;
        loop_time = time * 60 - 2.0;
        ipfile >> z;
        ipfile >> l;

        if (z > l)
        {
            cout << "Number of zones more than locations, check format of input file\n";
            exit(0);
        }

        int **tempT;
        int **tempN;

        tempT = new int *[l];
        for (int i = 0; i < l; ++i)
            tempT[i] = new int[l];

        tempN = new int *[z];
        for (int i = 0; i < z; ++i)
            tempN[i] = new int[z];

        for (int i = 0; i < z; i++)
        {
            for (int j = 0; j < z; j++)
                ipfile >> tempN[i][j];
        }

        for (int i = 0; i < l; i++)
        {
            for (int j = 0; j < l; j++)
                ipfile >> tempT[i][j];
        }

        ipfile.close();

        T = tempT;
        N = tempN;
    }
}

void SportsLayout::write_to_file(string outputfilename)
{

    // Open the file for writing
    ofstream outputFile(outputfilename);

    // Check if the file is opened successfully
    if (!outputFile.is_open())
    {
        cerr << "Failed to open the file for writing." << std::endl;
        exit(0);
    }

    for (int i = 0; i < z; i++)
        outputFile << mapping[i] << " ";

    // Close the file
    outputFile.close();

    cout << "Allocation written to the file successfully." << endl;
}

void SportsLayout::compute_allocation()
{
    // you can write your code here
    // comment out following dummy code
    // for (int i = 0; i < z; i++)
    //     mapping[i] = i + 1;

    bool starter = true;
    bool breaker = false;
    long long start_state_cost;
    long long curr_state_cost;

    vector<int> start_state;
    vector<int> curr_state;
    vector<int> all_locs;
    vector<int> temp_all_locs;

    for (int i = 0; i < l; i++)
    {
        all_locs.push_back(i + 1);
    }

    unsigned seed = chrono::high_resolution_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);

    for (int i = 0;; i++)
    {
        // cout << "Start no. " << i + 1 << endl;

        temp_all_locs = all_locs;
        shuffle(temp_all_locs.begin(), temp_all_locs.end(), generator);

        start_state.clear();
        for (int i = 0; i < z; i++)
        {
            start_state.push_back(temp_all_locs[i]);
        }

        start_state_cost = state_cost_fn(start_state);

        // cout << "Start State Details = ";
        // for (int i = 0; i < z; i++)
        // {
        //     cout << start_state[i] << " ";
        // }
        // cout << ": " << start_state_cost << endl;

        if (starter || start_state_cost < global_opt_state_cost)
        {
            global_opt_state = start_state;
            global_opt_state_cost = start_state_cost;
            starter = false;
        }

        currentTime = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsedTime = currentTime - startTime;

        if (elapsedTime.count() >= loop_time)
            break;

        curr_state = start_state;
        curr_state_cost = state_cost_fn(start_state);

        while (true) // hill climbing loop
        {
            // cout << "Current State Details = ";
            // for (int i = 0; i < z; i++)
            // {
            //     cout << curr_state[i] << " ";
            // }
            // cout << ": " << curr_state_cost << endl;

            pair<vector<int>, long long> best_neighbour_details = compute_best_neighbour(curr_state, curr_state_cost, all_locs);

            // cout << "Best Neighbour State Details = ";
            // for (int i = 0; i < best_neighbour_details.first.size(); i++)
            // {
            //     cout << best_neighbour_details.first[i] << " ";
            // }
            // cout << ": " << best_neighbour_details.second << endl
            //      << endl;

            if (best_neighbour_details.second < curr_state_cost)
            {
                curr_state = best_neighbour_details.first;
                curr_state_cost = best_neighbour_details.second;
            }
            else
            {
                // cout << "Reached Plateau / Hill Top!" << endl;
                break;
            }

            currentTime = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsedTime = currentTime - startTime;

            if (elapsedTime.count() >= loop_time)
            {
                breaker = true;
                break;
            }
        }

        if (curr_state_cost < global_opt_state_cost)
        {
            global_opt_state = curr_state;
            global_opt_state_cost = curr_state_cost;
        }

        if (breaker)
            break;

        // cout << "Current Global Optimal State Details = ";
        // for (int i = 0; i < z; i++)
        // {
        //     cout << global_opt_state[i] << " ";
        // }
        // cout << ": " << global_opt_state_cost << endl
        //      << endl
        //      << endl;
    }
}

pair<vector<int>, long long> SportsLayout::compute_best_neighbour(vector<int> curr_state, long long curr_state_cost, vector<int> all_locs)
{
    bool breaker2 = false;

    vector<int> temp_curr_state = curr_state;
    sort(temp_curr_state.begin(), temp_curr_state.end());

    vector<int> empty_locs;
    set_difference(all_locs.begin(), all_locs.end(), temp_curr_state.begin(), temp_curr_state.end(), back_inserter(empty_locs));

    vector<int> neighbour_state;
    long long neighbour_state_cost;

    vector<int> best_neighbour_state = curr_state;
    long long best_neighbour_state_cost = curr_state_cost;

    for (int i = 0; i < z; i++)
    {
        neighbour_state = curr_state;
        for (int j = 0; j < empty_locs.size(); j++)
        {
            neighbour_state[i] = empty_locs[j];
            neighbour_state_cost = state_cost_fn(neighbour_state);

            // for (int k = 0; k < neighbour_state.size(); k++)
            // {
            //     cout << neighbour_state[k] << " ";
            // }
            // cout << ": " << neighbour_state_cost << endl;

            if (neighbour_state_cost < best_neighbour_state_cost)
            {
                best_neighbour_state = neighbour_state;
                best_neighbour_state_cost = neighbour_state_cost;
            }

            currentTime = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsedTime = currentTime - startTime;

            if (elapsedTime.count() >= loop_time)
            {
                breaker2 = true;
                break;
            }
        }
        if (breaker2)
            break;
    }
    return make_pair(best_neighbour_state, best_neighbour_state_cost);
}