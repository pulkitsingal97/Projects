#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace std;

int main(int argc, char* argv[])
{
    ifstream inputFile(argv[1]);
    ofstream outputFile("format_g.txt_graph");

    if (!inputFile.is_open())
    {
        cout << " Hey!...Error opening file for reading." << std::endl;
        return 1;
    }

    int graph_counter = 0;
    string line;
    string s;
    int int_item;
    int num_ascii_s;
    string str_ascii_s;
    vector<string> mapping;

    while (getline(inputFile, line))
    {
        outputFile << "t # " << graph_counter << endl;
        graph_counter++;

        getline(inputFile, line);
        istringstream iss(line);
        iss >> s;
        int num_nodes = stoi(s);

        int node_counter = 0;
        for (int i = 0; i < num_nodes; i++)
        {
            getline(inputFile, line);
            istringstream iss2(line);
            iss2 >> s;
            num_ascii_s = 0;

            if(find(mapping.begin(), mapping.end(), s) == mapping.end())
            {
                mapping.push_back(s);
            }

            str_ascii_s = to_string(num_ascii_s);
            outputFile << "v " << node_counter << " " << find(mapping.begin(), mapping.end(), s)-mapping.begin() << endl;
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
            outputFile << "e " << edge_data[0] << " " << edge_data[1] << " " << edge_data[2] << endl;
        }

        getline(inputFile, line);
    }

    inputFile.close();
    outputFile.close();
    //cout << graph_counter << endl;

    return 0;
}
