#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <math.h>
using namespace std;

// Class of Node
class Node {
public:
    int variable = -1;
    double split_value = -1;
    int var_pos = -1;
    string variable_name;
    string label;
    Node *left_node;
    Node *right_node;
};
// Size of training data set
const int TrainingSize = 1280;
// Number of input variables
const int NumOfVariable = 11;
// Range of quality (between 0 and 10)
const int QualityRange = 10;
// Applies pruning
const bool ApplyPruning = false;
// Minimum number of leaf node (for pruning)
const int MinLeaf = 30;
// Number of output variables,
//     2: BAD(quality<7), GOOD(quality>=7)
//     3: BAD(quality<5), OK(5<=quality<=6), GOOD(quality>=7)
const int NumOfOutputVar = 2;
// Ways of splitting the data sets
//      0: First 80% (training), last 20% (testing)
//      1: Every 1 in 5 data sets (testing)
const int split_way = 0;
const int algorithm = 1;
// Current variable position (for calculation)
int CurrentVarPos = -1;
// // Set of all input data
vector<vector<double>> all_data_set;

// Sorts the data sets using the current variable position
bool SortDataSet(vector<double> a, vector<double> b) {
    return (a[CurrentVarPos] < b[CurrentVarPos]);
}



// Gets the array storing the number of each quality
vector<int> GetNumbers(vector<vector<double>> data_set){
    vector<int> numbers;
    for (int i = 0; i < NumOfOutputVar; ++i){
        numbers.push_back(0);
    }
    for (int i = 0; i < data_set.size(); ++i) {
        ++numbers[data_set[i][12]];
    }
    return numbers;
}

// Calculates and returns split info
double GetSplitInfo(vector<vector<double>> data_set){
    vector<int> numbers;
    vector<double> double_v;
    int continue_num = 0;
    double continue_val = data_set[0][CurrentVarPos];
    for (int i = 0; i < data_set.size(); ++i) {
        double cur_val = data_set[i][CurrentVarPos];
        if (cur_val != continue_val) {
            numbers.push_back(continue_num);
            double_v.push_back(continue_val);
            continue_val = cur_val;
            continue_num = 1;
            if (i == data_set.size() - 1) {
                numbers.push_back(continue_num);
                double_v.push_back(continue_val);
            }
        }
        else  continue_num++;
    }
    double res = 0.0;
    for (int i = 0; i < numbers.size(); ++i) {
        double pi = (double)numbers[i]/ data_set.size();
        double cal = 0.0;
        if (pi!=0 &&pi!=1)  cal = pi*(log2(pi));
        cal=-cal;
        res+=cal;
    }
    return res;
}

// Returns the classified label
int GetUniqueLabel(vector<vector<double>> data_set) {
    vector<int> numbers = GetNumbers(data_set);
    int max = numbers[0];
    int label = 0;
    for (int i = 0; i < numbers.size(); ++i) {
        if (numbers[i] >= max) {
            max = numbers[i];
            label = i;
        }
    }
    return label;
}

// Calculates and returns entropy
double CalculateEntropy(vector<vector<double>> data_set) {
    double res = 0.0;
    double set_size = data_set.size();
    vector<int> numbers = GetNumbers(data_set);
    if (set_size==1) return 0;
    for (int i = 0; i < numbers.size(); ++i) {
        double log_value = 0;
        double probability = (double)numbers[i]/set_size;
        if (probability != 0 && probability != 1) log_value = probability*(log2(probability));
        res = res + log_value;
    }
    res = -res;
    if (res == -0) res = 0;
    return res;
}

// Chooses and returns the variable and its splitting value which has the best information gain ratio
vector<double> ChooseBestInfo(vector<vector<double>> data_set) {
    double best_res = -1;
    double best_variable = -1;
    double best_split = -1;
    for (int variable = 0; variable<NumOfVariable; ++variable) {
        // Gets the entropy of the all data sets
        double node_entropy = CalculateEntropy(data_set);
        CurrentVarPos = variable;
        // Sorts the data sets
        vector<vector<double>> c_data_set = data_set;
        std::sort(c_data_set.begin(), c_data_set.end(), SortDataSet);
        for (int row = 0; row < c_data_set.size()-1; ++row) {
            double split_value = 0.5*(c_data_set[row][variable] + c_data_set[row+1][variable]);
            // Splits the data sets
            vector<vector<double>> left_set;
            vector<vector<double>> right_set;
            for (int k = 0; k < c_data_set.size(); ++k) {
                if (c_data_set[k][variable] <= split_value) left_set.push_back(c_data_set[k]);
                else right_set.push_back(c_data_set[k]);
            }
            // Gets the entropy of the left data sets
            double p_l = (double)left_set.size()/c_data_set.size();
            double left_entropy = CalculateEntropy(left_set) * p_l;

            // Gets the entropy of the right data sets
            double p_r = (double)right_set.size()/c_data_set.size();
            double right_entropy = CalculateEntropy(right_set) * p_r;
            // Calculates information Gain:
            double info_gain = node_entropy-(left_entropy+right_entropy);
            // Calculates information gain ratio:
            double split_info = GetSplitInfo(data_set);
            double info = info_gain/ split_info;
            if (algorithm==1) info = info_gain;
            // Finds the best information
            if (info > best_res) {
                best_res = info;
                best_variable = variable;
                best_split = split_value;
            }
        }
    }
    vector<double> res;
    res.push_back(best_variable);
    res.push_back(best_split);
    return res;
}

// Returns the label name corresponding to the label position
string GetLabelName(int label) {
    string name = "X";
    if (NumOfOutputVar==2) {
        if (label == 0) name = "BAD";
        else if (label == 1) name = "GOOD";
    } else if (NumOfOutputVar==3) {
        if (label == 0) name = "BAD";
        else if (label == 1) name = "OK";
        else if (label == 1) name = "GOOD";
    }
    return name;
}

// Generates the decision tree
Node *DecisionTreeLearning(Node *node, vector<vector<double>> data_set) {
    // Creates a new node
    class Node* new_node = new class Node();
    bool same_label = true;
    for (int i = 0; i < data_set.size() -1; ++i) {
        if (data_set[i][12] != data_set[i+1][12]) {
            same_label = false;
            break;
        }
    }
    //Returns the node as a leaf node labeled with a quality if the quality of all data sets is the same
    if (same_label) {
        new_node->label = GetLabelName(data_set[0][12]);
        return new_node;
    }
    // If pruning is applied and the number of the data sets is smaller than MinLeaf
    if (ApplyPruning && data_set.size()<=MinLeaf) {
        int label = GetUniqueLabel(data_set);
        //  Returns the node as a leaf node labeled with a quality if the quality is valid
        if (label != -1) {
            new_node->label = GetLabelName(label);
            return new_node;
        }
    }
    // Gets the variable and its splitting value which has the best information gain ratio
    vector<double> best_info = ChooseBestInfo(data_set);
    new_node->variable = best_info[0];
    new_node->split_value = best_info[1];
    // Splits the data sets
    vector<vector<double>> left_data_set;
    vector<vector<double>> right_data_set;
    for (int k = 0; k < data_set.size(); ++k) {
        if (data_set[k][best_info[0]] <= best_info[1]) left_data_set.push_back(data_set[k]);
        else  right_data_set.push_back(data_set[k]);
    }
    // Attaches the left child node
    new_node->left_node = DecisionTreeLearning(new_node, left_data_set);
    // Attaches the right child node
    new_node->right_node = DecisionTreeLearning(new_node, right_data_set);
    return new_node;
}

// Traverses the nodes
string NodeTraversal(Node *T, vector<double> row){
    string label;
    if (T->variable == -1 && T->split_value == -1) {
        label = T->label;
    } else {
        if (row[T->variable] <= T->split_value) {
            if (T->left_node != NULL) label = NodeTraversal(T->left_node, row);
        } else {
            if (T->right_node != NULL) label = NodeTraversal(T->right_node, row);
        }
    }
    return label;
}

// Predicts the quality of testing data sets
void Predict(Node *node, vector<vector<double>> test_data_set){
    double correct = 0;
    int good =0;
    int bad =0;
    int ok =0;
    for (int i = 0; i < test_data_set.size(); ++i) {
        vector<double> row = test_data_set[i];
        Node *anode = node;
        string predict_level_name =  NodeTraversal(anode, row);

        if (predict_level_name=="GOOD") good++;
        if (predict_level_name=="BAD") bad++;
        if (predict_level_name=="OK") ok++;


        string level_name =  GetLabelName(row[12]);

        if (GetLabelName(row[12])==predict_level_name) correct++;
    }
    double accuracy = correct/test_data_set.size();
    cout<<" accuracy "<<accuracy<<"  ("<<correct<<"/"<<test_data_set.size()<<")"<<endl;

    cout<<" good "<< double (good)/test_data_set.size()<<"  ("<<good<<"/"<<test_data_set.size()<<")"<<endl;
    cout<<" bad "<<double (bad)/test_data_set.size()<<"  ("<<bad<<"/"<<test_data_set.size()<<")"<<endl;
    cout<<" ok "<<double (ok)/test_data_set.size()<<"  ("<<ok<<"/"<<test_data_set.size()<<")"<<endl;
}

int main(int argc,char *argv[]) {
    // Reads the file
    string filename = "winequality-red.csv";
    ifstream fin(filename);
    string line;
    getline(fin, line);
    int line_num = 1;
    vector<vector<double>> train_data_set;
    vector<vector<double>> test_data_set;
    while (getline(fin, line)) {
        istringstream sin(line);
        vector<double> data;
        string info;
        while (getline(sin, info, ',')) {
            double aa = stod(info);
            data.push_back(aa);
        }
        if (NumOfOutputVar==2) {
            if (data[11]<7) data.push_back(0);
            else if (data[11]>=7) data.push_back(1);
        }
        if (NumOfOutputVar==3) {
            if (data[11]<5) data.push_back(0);
            else if (data[11]<=6 && data[11]>=5) data.push_back(1);
            else if (data[11]>6) data.push_back(2);
        }
        all_data_set.push_back(data);
        ++line_num;
    }
    for (int i = 0; i < all_data_set.size(); ++i) {
        all_data_set[i].push_back(i);
        if (split_way==0) {
            if (i < TrainingSize) train_data_set.push_back(all_data_set[i]);
            else  test_data_set.push_back(all_data_set[i]);
        } else if (split_way==1) {
            if (i%5==0) test_data_set.push_back(all_data_set[i]);
            else train_data_set.push_back(all_data_set[i]);
        }
    }
    Node *node = NULL;
    // Generates the decision tree and gets the root node
    Node* root_node = DecisionTreeLearning(node, train_data_set);
    // Predicts the quality of testing data sets
    Predict(root_node, test_data_set);
    return 0;
}
