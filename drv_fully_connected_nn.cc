/** 
 * \file drv_fully_connected_ann.cc 
 * \brief 手写体数字识别的全连接网络测试程序
 */
#include "fully_connected_nn.h"
using namespace std;

int main(){
    if(openblas_get_parallel() !=0 )
    {
        if(openblas_get_num_threads() > 1) 
            openblas_set_num_threads(1);
    }
    // 训练数据集
    vector<one_image> train_sample;
    parse_mnist_images("./mnist/train-images.idx3-ubyte", &train_sample, 
            -1.0, 1.0, 0, 0);
    vector<label_t> train_label;
    parse_mnist_labels("./mnist/train-labels.idx1-ubyte", &train_label);
    // 测试数据集
    vector<one_image> test_sample;
    parse_mnist_images("./mnist/t10k-images.idx3-ubyte", &test_sample,  
            -1.0, 1.0, 0, 0);
    vector<label_t> test_label;
    parse_mnist_labels("./mnist/t10k-labels.idx1-ubyte", &test_label);
    
    // 构建网络
    FullyConnectednn net(4);
    // 随机数生成器
    std::random_device rd;  
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<> dis(0.0, 1.0);
    // 初始化网络
    std::array<int, 4> in_channels{0, 28*28, 256, 128}, map_count{28*28, 256, 128, 10};
    net.get_a_layer(0)->set_type("input");            
    net.get_a_layer(1)->set_type("fully_connected"); 
    net.get_a_layer(2)->set_type("fully_connected");
    net.get_a_layer(3)->set_type("fully_connected");
    
    for(int i=0; i< 4; i++)  
        net.get_a_layer(i)->init_layer(in_channels[i], map_count[i], batch_size, dis, gen);
    
    // 训练与测试
    double learning_rate =  0.01 * sqrt((double)batch_size);
    const int epoch = 3;
    for (int i = 0; i < epoch; i++)
    {
        std::cout<<"current epoch: "<<i + 1 <<" *****************\n";
        Timer t; 
        net.train(train_sample, train_label, learning_rate);
        t.printDiff("training time is...");
        net.predict(test_sample, test_label);
        t.printDiff("predicting time is...");
        learning_rate *= 0.85;
        std::cout<<"\n";
    }
    return 0;
}
