#include "fully_connected_nn.h"
using namespace std;

void FullyConnectednn::backward_propagation(int j, label_t label)
{
    vector<double> y(this->layers.back().get_map_count(), -0.8); y[label] = 0.8;
    vector<double>::iterator iter_map = this->layers.back().get_map_data() + j*this->layers.back().get_map_count();
    vector<double>::iterator iter_delt_map = this->layers.back().get_delt_map_data() + j*this->layers.back().get_map_count();
    for (int i = 0; i < this->layers.back().get_map_count(); i++)
    {
        *iter_delt_map = LossFunction::DerivativeOfSumSquaredError( *iter_map, y[i]) * ActivationFunction::DerivativeTanh(*iter_map);
        iter_map ++; 
        iter_delt_map++;
    }
    for(size_t i = this->layers.size()-1; i>0;  i--) 
        this->layers[i].backward(j, this->layers[i-1]);
}

void FullyConnectednn::train(vector<one_image> &train_sample, vector<label_t> &train_label, double learning_rate)
{
    // 打乱样本顺序
    int j = 0, t = 0;
    vector<int> rand_perm(train_sample.size());
    for (size_t i = 0; i < train_sample.size(); i++) rand_perm[i] = i;
    random_device rd; 
    mt19937 gen(rd()); 
    uniform_int_distribution<> dis(0, numeric_limits<int>::max() );

    printf("%d\n", train_sample.size());
    for (size_t i = 0; i < train_sample.size(); i++)
    {
        j = dis(gen) % (train_sample.size() - i) + i;
        t = rand_perm[j];
        rand_perm[j] = rand_perm[i];
        rand_perm[i] = t;
    }

    // 批量训练
    int batch_count = train_sample.size() / batch_size;
    int progress_current{0}, progress_previous{0};
    
    for (int i = 0; i < batch_count; i++) 
    {
        // 重置可训练参数
        this->reset_weights();
        
        int index;
        #pragma omp parallel for private(index)
        for (j = 0; j < batch_size; j++)
        {
            // 一个样本
            index = i*batch_size + j;
            copy(train_sample[rand_perm[index]].begin(), train_sample[rand_perm[index]].end(), 
                this->layers[0].get_map_data() + j * train_sample[rand_perm[index]].size()); 
            
            this->forward_propagation(j);
            this->backward_propagation(j, train_label[rand_perm[index]]);
        }

        index = (i+1)*batch_size - 1;
        // 更新可训练参数
        this->update_weights(learning_rate);
        // 输出进度条
        progress_current = 100*index/train_sample.size();
        if ( progress_current - progress_previous >= 5) 
        {
            progress_bar(progress_current); 
            progress_previous = progress_current;
        }
    }
    cout<<"\n";
}

void FullyConnectednn::predict(vector<one_image> &test_sample, vector<label_t> &test_label)
{
    int num_success = 0;
    vector<int> confusion_matrix(num_classes * num_classes, 0);
    for (size_t i = 0; i < test_sample.size(); i++) 
    {
        copy(test_sample[i].begin(), test_sample[i].end(), this->layers[0].get_map_data() ); 
        this->forward_propagation(); 
        uint32_t digit = this->find_max_likelihood_index();
        uint32_t actual_value = test_label[i]; 
        if (digit == actual_value) num_success++;

        confusion_matrix[digit * num_classes + actual_value]++;
    }
    cout<<"accuracy: "<<num_success<<"/"<< test_sample.size()<<"\n";
    cout<<"\n   *  ";
    for (int i = 0; i < num_classes; i++) cout<<setw(4)<<i<<"  ";
    cout<<"\n";
    
    for (int i = 0; i < num_classes; i++) 
    {
        cout<<setw(4)<<i<<"  ";
        for (int j = 0; j < num_classes; j++) cout<<setw(4)<< confusion_matrix[i*num_classes + j]<<"  ";
        cout<<"\n";
    }
    cout<<"\n";
}
