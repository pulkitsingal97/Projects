#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

using namespace std;

int main(int argc, char *argv[])
{
    ifstream satfile(string(argv[1]) + ".satoutput");
    ofstream outfile(string(argv[1]) + ".mapping");

    if (!satfile.is_open())
    {
        cout << "Error opening sat file\n";
        return -1;
    }

    string status;
    satfile >> status;
    if (status == "UNSAT")
    {
        outfile << "0";
        satfile.close();
        outfile.close();
        return 0;
    }

    ifstream infile(string(argv[1]) + ".graph");
    if (!infile.is_open())
    {
        cout << "Error opening input file\n";
        return -1;
    }
    int n, edge, k1, k2;
    infile >> n >> edge >> k1 >> k2;
    infile.close();

    string line;
    getline(satfile, line);
    getline(satfile, line);
    vector<int> soln_vector;
    istringstream iss(line);
    for (string s; iss >> s;)
        soln_vector.push_back(stoi(s));
    satfile.close();

    int count1 = 0, count2 = 0;

    outfile << "#1" << endl;
    for (int i = 0; i < soln_vector.size(); i += 2)
    {
        if (soln_vector[i] > 0)
        {
            outfile << i / 2 + 1;
            count1++;
        }
        if (count1 == k1)
            break;
        if (soln_vector[i] > 0)
            outfile << " ";
    }
    outfile << endl;

    outfile << "#2" << endl;
    for (int i = 1; i < soln_vector.size(); i += 2)
    {
        if (soln_vector[i] > 0)
        {
            outfile << i / 2 + 1;
            count2++;
        }
        if (count2 == k2)
            break;
        if (soln_vector[i] > 0)
            outfile << " ";
    }

    outfile.close();
    return 0;
}