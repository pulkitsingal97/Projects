#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace std;

bool sortcol(const vector<int>& v1, const vector<int>& v2)
{
    if(v1[0] < v2[0])
    {
        return true;
    }
    else if(v1[0] == v2[0])
    {
        if(v1[1] < v2[1])
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    return false;
}

int main(int argc, char *argv[])
{
    ifstream inputFile(argv[1]);
    ofstream outputFile("format_f.txt_graph");

    if (!inputFile.is_open())
    {
        cout << " Hey!...Error opening file for reading." << std::endl;
        return 1;
    }

    int graph_counter = 0;
    string line;
    string s;
    int int_item;

    while (getline(inputFile, line))
    {
        outputFile << "t # " << graph_counter << endl;
        graph_counter++;
        vector<vector<int>> edges;

        getline(inputFile, line);
        istringstream iss(line);
        iss >> s;
        int num_nodes = stoi(s);
        ;

        int node_counter = 0;
        for (int i = 0; i < num_nodes; i++)
        {
            getline(inputFile, line);
            istringstream iss2(line);
            iss2 >> s;
            outputFile << "v " << node_counter << " " << s << endl;
            node_counter++;
        }

        getline(inputFile, line);
        istringstream iss3(line);
        iss3 >> s;
        int num_edges = stoi(s);

        for (int i = 0; i < num_edges; i++)
        {
            getline(inputFile, line);
            istringstream iss4(line);
            vector<int> edge_data;
            for (string s; iss4 >> s;)
            {
                int_item = stoi(s);
                edge_data.push_back(int_item);
            }
            // outputFile << "e " << edge_data[0] << " " << edge_data[1] << " " << edge_data[2] << endl;
            edges.push_back(edge_data);
        }

        sort(edges.begin(), edges.end(), sortcol);
            for(int i = 0; i < num_edges; i++)
            {
                outputFile << "u " << edges[i][0] << " " << edges[i][1] << " " << edges[i][2] << endl;
            }

        getline(inputFile, line);
    }

    inputFile.close();
    outputFile.close();

    return 0;
}
