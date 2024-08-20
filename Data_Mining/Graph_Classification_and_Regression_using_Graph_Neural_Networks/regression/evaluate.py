import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import torch

import matplotlib.pyplot as plt
import torch.nn
from torch.nn import Linear
from torch_geometric.data import DataLoader
import gzip
from io import StringIO
from torch_geometric.utils import to_networkx
from torch_geometric.loader import NeighborLoader as nl
import networkx as nx
from torch_geometric.nn import GAT, MLP, GCNConv,GATConv, global_mean_pool, global_add_pool
from networkx.classes.function import neighbors
from torch_geometric.data import Dataset, Data
import torch.nn.functional as F
import pickle
import argparse
import sys

def tocsv(y_arr, *, task):
    r"""Writes the numpy array to a csv file.
    params:
        y_arr: np.ndarray. A vector of all the predictions. Classes for
        classification and the regression value predicted for regression.

        task: str. Must be either of "classification" or "regression".
        Must be a keyword argument.
    Outputs a file named "y_classification.csv" or "y_regression.csv" in
    the directory it is called from. Must only be run once. In case outputs
    are generated from batches, only call this output on all the predictions
    from all the batches collected in a single numpy array. This means it'll
    only be called once.

    This code ensures this by checking if the file already exists, and does
    not over-write the csv files. It just raises an error.

    Finally, do not shuffle the test dataset as then matching the outputs
    will not work.
    """
    assert task in ["classification", "regression"], f"task must be either \"classification\" or \"regression\". Found: {task}"
    assert isinstance(y_arr, np.ndarray), f"y_arr must be a numpy array, found: {type(y_arr)}"
    assert len(y_arr.squeeze().shape) == 1, f"y_arr must be a vector. shape found: {y_arr.shape}"
    assert not os.path.isfile(f"y_{task}.csv"), f"File already exists. Ensure you are not calling this function multiple times (e.g. when looping over batches). Read the docstring. Found: y_{task}.csv"
    y_arr = y_arr.squeeze()
    df = pd.DataFrame(y_arr)
    df.to_csv(f"y_{task}.csv", index=False, header=False)


# The following is just for an example to show how to create the
# right data format, and how to call the function.
#
# The following assumes binary classification task. And the final
# output of the model is a number between 0 and 1, the probability
# of a sample belonging to class 1. For ROC-AUC, we need to save
# this number to the csv file.
#
# Notice that the example code for regression will also be the same.
def test(model, test_loader, device):
    model.eval()
    all_ys = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index, batch.batch)
            ys_this_batch = output.cpu().numpy().tolist()
            all_ys.extend(ys_this_batch)
    numpy_ys = np.asarray(all_ys)
    tocsv(numpy_ys, task="classification") # <- Called outside the loop. Called in the eval code.



class Evaluator:
    def __init__(self, name):
        self.name = name

        meta_info = pd.read_csv(os.path.join(os.path.dirname(__file__), 'master.csv'), index_col=0, keep_default_na=False)
        self.num_tasks = int(meta_info[self.name]['num tasks'])
        self.eval_metric = meta_info[self.name]['eval metric']


    def _parse_and_check_input(self, input_dict):
        if not 'y_true' in input_dict:
            raise RuntimeError('Missing key of y_true')
        if not 'y_pred' in input_dict:
            raise RuntimeError('Missing key of y_pred')

        y_true, y_pred = input_dict['y_true'], input_dict['y_pred']
        # print("y_true: ",y_true.reshape(-1,1))
        # print("y_pred: ", y_pred.reshape(-1,1))
        '''
            y_true: numpy ndarray or torch tensor of shape (num_graphs, num_tasks)
            y_pred: numpy ndarray or torch tensor of shape (num_graphs, num_tasks)
        '''

        # converting to torch.Tensor to numpy on cpu
        if torch is not None and isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()

        if torch is not None and isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()


        ## check type
        if not isinstance(y_true, np.ndarray):
            raise RuntimeError('Arguments to Evaluator need to be either numpy ndarray or torch tensor')

        if not y_true.shape == y_pred.shape:
            raise RuntimeError('Shape of y_true and y_pred must be the same')

        if not y_true.ndim == 2:
            raise RuntimeError('y_true and y_pred mush to 2-dim arrray, {}-dim array given'.format(y_true.ndim))

        if not y_true.shape[1] == self.num_tasks:
            raise RuntimeError('Number of tasks for {} should be {} but {} given'.format(self.name, self.num_tasks, y_true.shape[1]))

        return y_true, y_pred



    def eval(self, input_dict):

        if self.eval_metric == 'rocauc':
            y_true, y_pred = self._parse_and_check_input(input_dict)
            return self._eval_rocauc(y_true, y_pred)
        elif self.eval_metric == 'rmse':
            y_true, y_pred = self._parse_and_check_input(input_dict)
            return self._eval_rmse(y_true, y_pred)

    @property
    def expected_input_format(self):
        desc = '==== Expected input format of Evaluator for {}\n'.format(self.name)
        if self.eval_metric == 'rocauc':
            desc += '{\'y_true\': y_true, \'y_pred\': y_pred}\n'
            desc += '- y_true: numpy ndarray or torch tensor of shape (num_graphs, num_tasks)\n'
            desc += '- y_pred: numpy ndarray or torch tensor of shape (num_graphs, num_tasks)\n'
            desc += 'where y_pred stores score values (for computing AUC score),\n'
            desc += 'num_task is {}, and '.format(self.num_tasks)
            desc += 'each row corresponds to one graph.\n'
            desc += 'nan values in y_true are ignored during evaluation.\n'
        elif self.eval_metric == 'rmse':
            desc += '{\'y_true\': y_true, \'y_pred\': y_pred}\n'
            desc += '- y_true: numpy ndarray or torch tensor of shape (num_graphs, num_tasks)\n'
            desc += '- y_pred: numpy ndarray or torch tensor of shape (num_graphs, num_tasks)\n'
            desc += 'where num_task is {}, and '.format(self.num_tasks)
            desc += 'each row corresponds to one graph.\n'
            desc += 'nan values in y_true are ignored during evaluation.\n'
        return desc

    @property
    def expected_output_format(self):
        desc = '==== Expected output format of Evaluator for {}\n'.format(self.name)
        if self.eval_metric == 'rocauc':
            desc += '{\'rocauc\': rocauc}\n'
            desc += '- rocauc (float): ROC-AUC score averaged across {} task(s)\n'.format(self.num_tasks)
        elif self.eval_metric == 'rmse':
            desc += '{\'rmse\': rmse}\n'
            desc += '- rmse (float): root mean squared error averaged across {} task(s)\n'.format(self.num_tasks)
        return desc

    def _eval_rocauc(self, y_true, y_pred):
        '''
            compute ROC-AUC averaged across tasks
        '''

        rocauc_list = []

        for i in range(y_true.shape[1]):
            #AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
                # ignore nan values
                is_labeled = y_true[:,i] == y_true[:,i]
                rocauc_list.append(roc_auc_score(y_true[is_labeled,i], y_pred[is_labeled,i]))

        if len(rocauc_list) == 0:
            raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

        return {'rocauc': sum(rocauc_list)/len(rocauc_list)}


    def _eval_rmse(self, y_true, y_pred):
        '''
            compute RMSE score averaged across tasks
        '''
        rmse_list = []

        for i in range(y_true.shape[1]):
            # ignore nan values
            is_labeled = y_true[:,i] == y_true[:,i]
            rmse_list.append(np.sqrt(((y_true[is_labeled,i] - y_pred[is_labeled,i])**2).mean()))

        return {'rmse': sum(rmse_list)/len(rmse_list)}

    def _eval_acc(self, y_true, y_pred):
        acc_list = []

        for i in range(y_true.shape[1]):
            is_labeled = y_true[:,i] == y_true[:,i]
            correct = y_true[is_labeled,i] == y_pred[is_labeled,i]
            acc_list.append(float(np.sum(correct))/len(correct))

        return {'acc': sum(acc_list)/len(acc_list)}


if __name__ == '__main__':

    # ***************************
    gt_labels = False
    data_list=[]

    train_files=[]
    df_train=[]
    print('Load dataset')
    # for item in files:
    #   path=os.path.join(dir,item)
    path = sys.argv[4]
    for graph_file in os.listdir(path):
        train_files.append(graph_file)
        with gzip.open(os.path.join(path,graph_file), "rb") as gz_file:
            content=gz_file.read().decode("utf-8")
            content=StringIO(content)
            x=pd.read_csv(content, header=None)
            df_train.append(x)
    print('Dataset loaded')
    num_nodes_idx = train_files.index("num_nodes.csv.gz")
    node_features_idx = train_files.index("node_features.csv.gz")
    num_edges_idx = train_files.index("num_edges.csv.gz")
    edge_featuress_idx = train_files.index("edge_features.csv.gz")
    # graph_labels_idx = train_files.index("graph_labels.csv.gz")
    edges_idx = train_files.index("edges.csv.gz")
    if train_files.count("graph_labels.csv.gz"):
        gt_labels = True
    if gt_labels:
        graph_labels_idx = train_files.index("graph_labels.csv.gz")
        null_idx=df_train[graph_labels_idx].isnull()
    # null_idx

    num_graphs=df_train[num_nodes_idx].shape[0]
    node_ptr=0
    edge_ptr=0
    # node_features_concat=[]
    # edge_features_concat=[]
    # edge_concat=[]
    print("Creating data_list")
    for x in range(num_graphs):

        # if x==2:
        #   break
        num_nodes=df_train[num_nodes_idx][0][x]
        node_features=torch.tensor(df_train[node_features_idx].loc[node_ptr:node_ptr+num_nodes-1].values, dtype=torch.float32)
        # node_features_concat.append(node_features)

        num_edges=df_train[num_edges_idx][0][x]
        edges=torch.tensor(df_train[edges_idx].loc[edge_ptr:edge_ptr+num_edges-1].values, dtype=torch.float32)
        edges=np.transpose(edges)
        # edge_concat.append(edges)

        edge_features=torch.tensor(df_train[edge_featuress_idx].loc[edge_ptr:edge_ptr+num_edges-1].values, dtype=torch.float32)
        # edge_features_concat.append(edge_features)

        if gt_labels:
            graph_label=torch.tensor(df_train[graph_labels_idx][0][x], dtype=torch.float32)

        node_ptr=node_ptr+num_nodes
        edge_ptr=edge_ptr+num_edges

        if gt_labels:
            if null_idx[0][x]==True:
                continue
            data=Data(x=node_features, edge_index=edges.to(torch.int64), edge_attr=edge_features, y=graph_label)
        else:
            data=Data(x=node_features, edge_index=edges.to(torch.int64), edge_attr=edge_features)
        data_list.append(data)

        # print(data.edge_index)
        # print(data)
        

    print("data_list created")

    print('Create class GNN')
    class GNN(torch.nn.Module):
      def __init__(self, num_node_features, hidden_channels, num_edge_features):
          super(GNN, self).__init__()
          torch.manual_seed(12345)
          self.conv1 = GATConv(num_node_features, hidden_channels, edge_dim=num_edge_features)
          self.conv2 = GATConv(hidden_channels, hidden_channels, edge_dim=num_edge_features)
          self.lin = Linear(hidden_channels, 1)

      def forward(self, x, edge_index, edge_attr, batch=None):
          # Node embedding
          x = self.conv1(x, edge_index, edge_attr=edge_attr)
          x = x.relu()
          x = self.conv2(x, edge_index, edge_attr=edge_attr) 
          # Readout layer
          batch = torch.zeros(x.shape[0], dtype=int) if batch is None else batch
          x = global_add_pool(x, batch)
          # Final classifier
          x = F.dropout(x, p=0.5, training=self.training)
          x = self.lin(x)
          # x = F.softmax(x, dim=1)

          return x  # Add this line to return the final output

    print("Model Created")

    model_file = sys.argv[2]
    with open(model_file, 'rb') as file:
      model_new = pickle.load(file)

    # print("Print model's weight")
    # for name, param in model_new.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)

    if gt_labels:
        loss=0
        pred_list = []
        true_list = []
        for data in data_list:
            pred=model_new.forward(data.x, data.edge_index, data.edge_attr)
            pred_list.append(float(pred))
            true_list.append(float(data.y))
            # pred=pred.(dim=1)
            # loss+=(float(pred)-float(data.y))**2
            # print(f"{float(pred):.4f} {float(data.y):.4f} {(float(pred)-float(data.y)):.4f}")
            # loss/=len(data_list)
            # print(f"Loss: {loss:.4f}")
        pred_list = np.array(pred_list).reshape(-1,1)
        true_list = np.array(true_list).reshape(-1,1)

        ## auc case
        # evaluator = Evaluator('dataset-2')
        # print(evaluator.expected_input_format)
        # print(evaluator.expected_output_format)
        # y_true = torch.tensor(np.random.randint(2, size = (100,1)))
        # y_pred = torch.tensor(np.random.randn(100,1))
        # input_dict = {'y_true': y_true, 'y_pred': y_pred}
        # result = evaluator.eval(input_dict)
        # print(result)

        ### rmse case
        evaluator = Evaluator('dataset-1')
        print(evaluator.expected_input_format)
        print(evaluator.expected_output_format)
        tocsv(pred_list, task="regression")

        # y_true = np.random.randn(100,1)
        # y_pred = np.random.randn(100,1)
        input_dict = {'y_true': true_list, 'y_pred': pred_list}
        # print("input_dict: ", input_dict)
        result = evaluator.eval(input_dict)
        print(result)
    else:
        loss=0
        pred_list = []
        # true_list = []
        for data in data_list:
            pred=model_new.forward(data.x, data.edge_index, data.edge_attr)
            pred_list.append(float(pred))
            # true_list.append(float(data.y))
            # pred=pred.(dim=1)
            # loss+=(float(pred)-float(data.y))**2
            # print(f"{float(pred):.4f} {float(data.y):.4f} {(float(pred)-float(data.y)):.4f}")
            # loss/=len(data_list)
            # print(f"Loss: {loss:.4f}")
        pred_list = np.array(pred_list).reshape(-1,1)
        # true_list = np.array(true_list).reshape(-1,1)

        ## auc case
        # evaluator = Evaluator('dataset-2')
        # print(evaluator.expected_input_format)
        # print(evaluator.expected_output_format)
        # y_true = torch.tensor(np.random.randint(2, size = (100,1)))
        # y_pred = torch.tensor(np.random.randn(100,1))
        # input_dict = {'y_true': y_true, 'y_pred': y_pred}
        # result = evaluator.eval(input_dict)
        # print(result)

        # ### rmse case
        # evaluator = Evaluator('dataset-1')
        # print(evaluator.expected_input_format)
        # print(evaluator.expected_output_format)
        tocsv(pred_list, task="regression")

        # y_true = np.random.randn(100,1)
        # y_pred = np.random.randn(100,1)
        # input_dict = {'y_true': true_list, 'y_pred': pred_list}
        # # print("input_dict: ", input_dict)
        # result = evaluator.eval(input_dict)
        # print(result)

