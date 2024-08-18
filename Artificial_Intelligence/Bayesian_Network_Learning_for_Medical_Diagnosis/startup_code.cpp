#include <iostream>
#include <string>
#include <vector>
#include <list>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <set>
#include <map>
#include <algorithm>
#include <iomanip>
#include <chrono>

using namespace std;

// Our graph consists of a list of nodes where each node is represented as follows:
class Graph_Node
{

private:
	string Node_Name;		// Variable name
	int node_depth;			// depth of the node
	vector<int> Children;	// Children of a particular node - these are index of nodes in graph.
	vector<string> Parents; // Parents of a particular node- note these are names of parents
	int nvalues;			// Number of categories a variable represented by this node can take
	vector<string> values;	// Categories of possible values
	vector<float> CPT;		// conditional probability table as a 1-d array . Look for BIF format to understand its meaning

public:
	// Constructor- a node is initialised with its name and its categories
	Graph_Node(string name, int n, vector<string> vals)
	{
		Node_Name = name;
		nvalues = n;
		values = vals;
	}
	string get_name()
	{
		return Node_Name;
	}
	vector<int> get_children()
	{
		return Children;
	}
	vector<string> get_Parents()
	{
		return Parents;
	}
	vector<float> get_CPT()
	{
		return CPT;
	}
	int get_nvalues()
	{
		return nvalues;
	}
	vector<string> get_values()
	{
		return values;
	}
	void set_CPT(vector<float> new_CPT)
	{
		CPT.clear();
		CPT = new_CPT;
	}
	void set_Parents(vector<string> Parent_Nodes)
	{
		Parents.clear();
		Parents = Parent_Nodes;
	}
	// add another node in a graph as a child of this node
	int add_child(int new_child_index)
	{
		for (int i = 0; i < Children.size(); i++)
		{
			if (Children[i] == new_child_index)
				return 0;
		}
		Children.push_back(new_child_index);
		return 1;
	}
};

// The whole network represted as a list of nodes
class network
{
	list<Graph_Node> Pres_Graph;

public:
	int addNode(Graph_Node node)
	{
		Pres_Graph.push_back(node);
		return 0;
	}

	int netSize()
	{
		return Pres_Graph.size();
	}
	// get the index of node with a given name
	int get_index(string val_name)
	{
		list<Graph_Node>::iterator listIt;
		int count = 0;
		for (listIt = Pres_Graph.begin(); listIt != Pres_Graph.end(); listIt++)
		{
			if (listIt->get_name().compare(val_name) == 0)
				return count;
			count++;
		}
		return -1;
	}
	// get the node at nth index
	list<Graph_Node>::iterator get_nth_node(int n)
	{
		list<Graph_Node>::iterator listIt;
		int count = 0;
		for (listIt = Pres_Graph.begin(); listIt != Pres_Graph.end(); listIt++)
		{
			if (count == n)
				return listIt;
			count++;
		}
		return listIt;
	}
	// get the iterator of a node with a given name
	list<Graph_Node>::iterator search_node(string val_name)
	{
		list<Graph_Node>::iterator listIt;
		for (listIt = Pres_Graph.begin(); listIt != Pres_Graph.end(); listIt++)
		{
			if (listIt->get_name().compare(val_name) == 0)
				return listIt;
		}

		cout << "node not found\n";
		return listIt;
	}
};

network read_network(string filename)
{
	network Alarm;
	string line;
	int find = 0;
	ifstream myfile(filename);
	string temp;
	string name;
	vector<string> values;

	if (myfile.is_open())
	{
		while (!myfile.eof())
		{
			stringstream ss;
			getline(myfile, line);

			ss.str(line);
			ss >> temp;

			if (temp.compare("variable") == 0)
			{

				ss >> name;
				getline(myfile, line);

				stringstream ss2;
				ss2.str(line);
				for (int i = 0; i < 4; i++)
				{

					ss2 >> temp;
				}
				values.clear();
				while (temp.compare("};") != 0)
				{
					values.push_back(temp);

					ss2 >> temp;
				}
				Graph_Node new_node(name, values.size(), values);
				int pos = Alarm.addNode(new_node);
			}
			else if (temp.compare("probability") == 0)
			{

				ss >> temp;
				ss >> temp;

				list<Graph_Node>::iterator listIt;
				list<Graph_Node>::iterator listIt1;
				listIt = Alarm.search_node(temp);
				int index = Alarm.get_index(temp);
				ss >> temp;
				values.clear();
				while (temp.compare(")") != 0)
				{
					listIt1 = Alarm.search_node(temp);
					listIt1->add_child(index);
					values.push_back(temp);

					ss >> temp;
				}
				listIt->set_Parents(values);
				getline(myfile, line);
				stringstream ss2;

				ss2.str(line);
				ss2 >> temp;

				ss2 >> temp;

				vector<float> curr_CPT;
				string::size_type sz;
				while (temp.compare(";") != 0)
				{

					curr_CPT.push_back(atof(temp.c_str()));

					ss2 >> temp;
				}

				listIt->set_CPT(curr_CPT);
			}
			else
			{
			}
		}

		if (find == 1)
			myfile.close();
	}

	return Alarm;
}

int find_CPT_pos(int node_id, vector<int> parents_id, vector<string> record, vector<map<string, int>> vect_mp, int node_par_val_prod)
{
	vector<pair<int, int>> val_indx_size;

	for (auto par_id : parents_id)
	{
		string val = record[par_id];
		int val_indx = vect_mp[par_id][val];
		int val_size = vect_mp[par_id].size();
		val_indx_size.push_back(make_pair(val_indx, val_size));
	}

	string node_val = record[node_id];
	int node_val_indx = vect_mp[node_id][node_val];
	int CPT_indx = 0;
	CPT_indx += (node_val_indx)*node_par_val_prod;

	for (int i = 0; i < val_indx_size.size(); i++)
	{
		node_par_val_prod /= val_indx_size[i].second;
		CPT_indx += (val_indx_size[i].first) * node_par_val_prod;
	}

	return CPT_indx;
}

float calc_cond_prob(int node_id, network Alarm, vector<int> record_CPT_pos)
{
	list<Graph_Node>::iterator iter = Alarm.get_nth_node(node_id);
	vector<float> node_CPT = iter->get_CPT();
	int CPT_indx = record_CPT_pos[node_id];
	return node_CPT[CPT_indx];
}

float calc_markov_prob(int node_id, network Alarm, vector<int> record_CPT_pos)
{
	float prod = 1.0;
	prod *= calc_cond_prob(node_id, Alarm, record_CPT_pos);

	list<Graph_Node>::iterator iter = Alarm.get_nth_node(node_id);
	vector<int> children_id = iter->get_children();

	for (auto child_id : children_id)
	{
		prod *= calc_cond_prob(child_id, Alarm, record_CPT_pos);
	}

	return prod;
}

void update_all_weights(vector<float> &all_weights, network Alarm, vector<int> all_unknown_idx, vector<vector<int>> updated_CPT_pos)
{
	all_weights.clear();
	int updated_CPT_pos_idx = -1;

	for (int i = 0; i < all_unknown_idx.size(); i++)
	{
		int node_id = all_unknown_idx[i];

		if (node_id != -1)
		{
			list<Graph_Node>::iterator iter = Alarm.get_nth_node(node_id);
			int num_val = iter->get_nvalues();
			vector<float> record_weights;
			float sum = 0.0;

			for (int j = 0; j < num_val; j++)
			{
				updated_CPT_pos_idx++;
				vector<int> record_CPT_pos = updated_CPT_pos[updated_CPT_pos_idx];
				float node_markov_prob = calc_markov_prob(node_id, Alarm, record_CPT_pos);
				sum += node_markov_prob;
				record_weights.push_back(node_markov_prob);
			}

			for (auto weight : record_weights)
			{
				weight /= sum;
				all_weights.push_back(weight);
			}
		}
		else
		{
			all_weights.push_back(1);
			updated_CPT_pos_idx++;
		}
	}

	return;
}

void update_all_CPT(network &Alarm, vector<float> all_weights, vector<int> parents_value_products, vector<vector<int>> updated_CPT_pos)
{
	for (int i = 0; i < Alarm.netSize(); i++)
	{
		int node_id = i;
		list<Graph_Node>::iterator iter = Alarm.get_nth_node(node_id);
		int node_CPT_size = iter->get_CPT().size();
		vector<float> CPT_vect(node_CPT_size, 0.0035);
		int node_par_val_prod = parents_value_products[node_id];

		for (int j = 0; j < updated_CPT_pos.size(); j++)
		{
			int node_CPT_pos = updated_CPT_pos[j][node_id];
			CPT_vect[node_CPT_pos] += all_weights[j];
		}

		for (int k = 0; k < node_par_val_prod; k++)
		{
			float sum = 0;

			for (int m = 0; m < (iter->get_nvalues()); m++)
				sum += CPT_vect[m * node_par_val_prod + k];

			for (int m = 0; m < (iter->get_nvalues()); m++)
				CPT_vect[m * node_par_val_prod + k] /= sum;
		}

		iter->set_CPT(CPT_vect);
	}

	return;
}

void write_output(string filename, network Alarm)
{
	int node_id = 0;
	ifstream inputfile(filename);

	if (!inputfile.is_open())
	{
		cout << " Hey!...Error opening file for reading in write_output()" << std::endl;
		return;
	}

	ofstream outfile("solved_alarm.bif");
	string line;

	while (getline(inputfile, line))
	{
		istringstream iss(line);
		string s;
		iss >> s;

		if (s != "table")
		{
			outfile << line;
			if (node_id != Alarm.netSize())
				outfile << endl;
			continue;
		}
		else
			outfile << "\t" << s << " ";

		list<Graph_Node>::iterator iter = Alarm.get_nth_node(node_id);
		vector<float> node_CPT = iter->get_CPT();

		for (auto prob : node_CPT)
		{
			// if (prob >= 0.0001)
			outfile << fixed << setprecision(4) << prob << " ";
			// else
			// 	outfile << "0.0001 ";
		}

		outfile << ";" << endl;
		node_id++;
	}

	outfile.close();
	inputfile.close();

	return;
}

int main(int argc, char *argv[])
{
	auto start = chrono::high_resolution_clock::now();
	int time_limit = 115;

	network Alarm;
	Alarm = read_network(argv[1]);

	ifstream records_data_file(argv[2]);
	if (!records_data_file.is_open())
	{
		cout << " Hey!...Error opening records data file for reading." << std::endl;
		return 1;
	}

	vector<int> all_unknown_idx;
	vector<vector<string>> patients_data;
	string line;

	// reading records data file
	while (getline(records_data_file, line))
	{
		int rec_unknown_idx = -1;
		int rec_unknown_counter = -1;
		vector<string> record;
		istringstream iss(line);

		for (string s; iss >> s;)
		{
			record.push_back(s);
			rec_unknown_counter++;
			if (s == "\"?\"")
			{
				rec_unknown_idx = rec_unknown_counter;
			}
		}

		all_unknown_idx.push_back(rec_unknown_idx);
		patients_data.push_back(record);
	}
	records_data_file.close();

	// initializing CPT tables
	for (int i = 0; i < Alarm.netSize(); i++)
	{
		int node_id = i;
		list<Graph_Node>::iterator iter = Alarm.get_nth_node(node_id);
		int value = iter->get_nvalues();
		vector<float> CPT_vect = iter->get_CPT();

		for (int j = 0; j < CPT_vect.size(); j++)
			CPT_vect[j] = 1.0 / value;

		iter->set_CPT(CPT_vect);
	}

	// creating vector of all_parents_id;
	vector<vector<int>> all_parents_id;
	for (int i = 0; i < Alarm.netSize(); i++)
	{
		int node_id = i;
		list<Graph_Node>::iterator iter = Alarm.get_nth_node(node_id);
		vector<string> parents_name = iter->get_Parents();
		vector<int> parents_id;

		for (int j = 0; j < parents_name.size(); j++)
		{
			int par_id = Alarm.get_index(parents_name[j]);
			parents_id.push_back(par_id);
		}

		all_parents_id.push_back(parents_id);
	}

	// creating map vector
	vector<map<string, int>> vect_mp;
	for (int i = 0; i < Alarm.netSize(); i++)
	{
		int node_id = i;
		map<string, int> mp;
		list<Graph_Node>::iterator iter = Alarm.get_nth_node(node_id);
		vector<string> node_vals = iter->get_values();

		for (int j = 0; j < node_vals.size(); j++)
			mp[node_vals[j]] = j;

		vect_mp.push_back(mp);
	}

	// creating vector of parents_value_products
	vector<int> parents_value_products;
	for (int i = 0; i < Alarm.netSize(); i++)
	{
		int node_id = i;
		vector<int> parents_id = all_parents_id[node_id];
		int prod = 1;

		for (auto par_id : parents_id)
		{
			int val_size = vect_mp[par_id].size();
			prod *= val_size;
		}

		parents_value_products.push_back(prod);
	}

	// creating updated patients data
	vector<vector<string>> updated_patients_data;
	for (int i = 0; i < patients_data.size(); i++)
	{
		int node_id = all_unknown_idx[i];
		if (node_id != -1)
		{
			for (auto mp : vect_mp[node_id])
			{
				string node_val = mp.first;
				patients_data[i][node_id] = node_val;
				updated_patients_data.push_back(patients_data[i]);
			}
		}
		else
			updated_patients_data.push_back(patients_data[i]);
	}

	patients_data.clear();

	// creating updated_CPT_pos
	vector<vector<int>> updated_CPT_pos;
	for (auto record : updated_patients_data)
	{
		vector<int> record_CPT_pos;

		for (int i = 0; i < record.size(); i++)
		{
			int node_id = i;
			vector<int> parents_id = all_parents_id[node_id];
			int node_par_val_prod = parents_value_products[node_id];
			int node_CPT_pos = find_CPT_pos(node_id, parents_id, record, vect_mp, node_par_val_prod);
			record_CPT_pos.push_back(node_CPT_pos);
		}

		updated_CPT_pos.push_back(record_CPT_pos);
	}

	updated_patients_data.clear();
	all_parents_id.clear();
	vect_mp.clear();

	vector<float> all_weights;

	while (true)
	{
		update_all_weights(all_weights, Alarm, all_unknown_idx, updated_CPT_pos);
		update_all_CPT(Alarm, all_weights, parents_value_products, updated_CPT_pos);
		write_output(argv[1], Alarm);

		// system("./Format_Checker");
		// cout << endl;

		auto current = chrono::high_resolution_clock::now();
		auto elapsed = chrono::duration_cast<chrono::seconds>(current - start);
		if (elapsed.count() > time_limit)
			break;
	}

	// auto current = chrono::high_resolution_clock::now();
	// auto fin_elapsed = chrono::duration_cast<chrono::milliseconds>(current - start);
	// cout << "Total time: " << fin_elapsed.count() << endl;

	return 0;
}