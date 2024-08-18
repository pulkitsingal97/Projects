#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

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

// void generateCombinations(ofstream &outfile, vector<int> &currentCombo, vector<int> &elements, int k, int index)
// {
//     if (k == 0)
//     {
//         for (int element : currentCombo)
//         {
//             outfile << element << " ";
//         }
//         outfile << "0" << endl;
//         return;
//     }

//     for (int i = index; i < elements.size(); ++i)
//     {
//         currentCombo.push_back(elements[i]);
//         generateCombinations(outfile, currentCombo, elements, k - 1, i + 1);
//         currentCombo.pop_back();
//     }
// }

int main(int argc, char *argv[])
{
    ifstream infile(string(argv[1]) + ".graph");
    ofstream outfile(string(argv[1]) + ".satinput");

    if (!infile.is_open())
    {
        cout << "Error opening input file\n";
        return -1;
    }

    int n, edge, k1, k2;
    infile >> n >> edge >> k1 >> k2;

    if (edge < (k1 * (k1 - 1)) / 2 + (k2 * (k2 - 1)) / 2)
    {
        outfile << "p cnf 1 2" << endl;
        outfile << "1 0" << endl;
        outfile << "-1 0" << endl;
        infile.close();
        outfile.close();
        return 0;
    }

    outfile << "p cnf " << 2 * n << " " << n + 2 * (nCk(n, 2) - edge) + nCk(n, n - k1 + 1) + nCk(n, n - k2 + 1) << endl;
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

    for (int i = 0; i < n; i++)
    {
        outfile << -(2 * i + 1) << " " << -(2 * i + 2) << " 0" << endl;
    }

    for (int i = 0; i < n - 1; i++)
    {
        for (int j = i + 1; j < n; j++)
        {
            if (edge_matrix[i][j] == 0)
            {
                outfile << "-" << 2 * i + 1 << " -" << 2 * j + 1 << " 0" << endl;
                outfile << "-" << 2 * i + 2 << " -" << 2 * j + 2 << " 0" << endl;
            }
        }
    }
    vector<int> clique1_elements;
    vector<int> clique2_elements;
    for (int i = 0; i < n; i++)
    {
        clique1_elements.push_back(2 * i + 1);
        clique2_elements.push_back(2 * i + 2);
    }

    generateCombinations(outfile, clique1_elements, n - k1 + 1);
    generateCombinations(outfile, clique2_elements, n - k2 + 1);

    infile.close();
    outfile.close();

    return 0;
}