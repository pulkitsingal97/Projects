#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>

using namespace std;

unsigned long long nCk(int n, int k)
{
    if (k < 0 || k > n)
        return 0;

    if (k > n - k)
        k = n - k;

    unsigned long long result = 1;

    for (int i = 0; i < k; ++i)
    {
        result *= (n - i);
        result /= (i + 1);
    }

    return result;
}

void generateCombinations(ofstream &outfile, vector<int> &elements, int k)
{
    int n = elements.size();
    vector<int> combination(k, 0);

    for (int i = 0; i < k; ++i)
    {
        combination[i] = i;
    }

    while (true)
    {
        for (int i = 0; i < k; ++i)
        {
            outfile << elements[combination[i]] << " ";
        }
        outfile << "0" << endl;

        int i = k - 1;
        while (i >= 0 && combination[i] == n - k + i)
        {
            --i;
        }

        if (i < 0)
        {
            break;
        }

        ++combination[i];

        for (int j = i + 1; j < k; ++j)
        {
            combination[j] = combination[j - 1] + 1;
        }
    }
}

int main(int argc, char *argv[])
{
    ifstream infile(string(argv[1]) + ".graph");
    if (!infile.is_open())
    {
        cout << "Error opening input file\n";
        return -1;
    }

    int n, edge;
    infile >> n >> edge;
    int k_max = (1 + sqrt(1 + 8 * edge)) / 2;

    vector<vector<int>> edge_matrix(n, vector<int>(n, 0));

    string line;
    getline(infile, line);
    while (getline(infile, line))
    {
        vector<int> edge_vector;
        istringstream iss(line);
        for (string s; iss >> s;)
            edge_vector.push_back(stoi(s));
        edge_matrix[edge_vector[0] - 1][edge_vector[1] - 1] = 1;
        edge_matrix[edge_vector[1] - 1][edge_vector[0] - 1] = 1;
    }

    infile.close();

    vector<int> clique_elements;
    for (int i = 0; i < n; i++)
        clique_elements.push_back(i + 1);

    int left = 2, right = k_max, k, mid;

    while (left <= right)
    {
        mid = (left + right) / 2;
        k = mid;
        // cout << endl
        //      << "k: " << k << endl;

        ofstream satinput(string(argv[1]) + ".satinput");
        satinput << "p cnf " << n << " " << (nCk(n, 2) - edge) + nCk(n, n - k + 1) << endl;

        for (int i = 0; i < n - 1; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                if (edge_matrix[i][j] == 0)
                {
                    satinput << "-" << i + 1 << " -" << j + 1 << " 0" << endl;
                }
            }
        }

        generateCombinations(satinput, clique_elements, n - k + 1);

        satinput.close();

        string command = "./minisat " + string(argv[1]) + ".satinput " + string(argv[1]) + ".satoutput";
        system(command.c_str());

        ifstream satoutput(string(argv[1]) + ".satoutput");
        if (!satoutput.is_open())
        {
            cout << "Error opening satoutput file\n";
            return -1;
        }

        string status;
        satoutput >> status;

        if (status == "UNSAT")
        {
            satoutput.close();
            right = mid - 1;
            continue;
        }

        ofstream outfile(string(argv[1]) + ".mapping");
        string line;
        getline(satoutput, line);
        getline(satoutput, line);
        vector<int> soln_vector;
        istringstream iss(line);
        for (string s; iss >> s;)
            soln_vector.push_back(stoi(s));
        satoutput.close();

        int count = 0;

        outfile << "#1" << endl;
        for (int i = 0; i < soln_vector.size(); i++)
        {
            if (soln_vector[i] > 0)
            {
                outfile << i + 1;
                count++;
            }
            if (count == k)
                break;
            if (soln_vector[i] > 0)
                outfile << " ";
        }
        outfile.close();
        left = mid + 1;
    }

    return 0;
}