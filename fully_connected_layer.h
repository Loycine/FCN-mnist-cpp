#pragma once
#include <cmath>
#include <random>
#include <cblas.h>
#include "mnist_parser.h"
#include "utilities_sc.h"
#include <vector>
using namespace std;

/**
 * \brief 一个层
 * @param[type_]: 层的类型：输入层或普通全连接层 
 * @param[map_count_]: 神经元个数，等于卷积层中特征图的个数 
 * @param[map_data_] : 保存输出特征图的数组
 * @param[kernel_]   : 保存可训练权重参数的数组 
 * @param[bias_]     : 保存可训练偏置的数组
 * \note 以"_delt_"开始的数据成员保存了对应成员的梯度
 */

class FullyConnectedLayer
{
public:
    void forward(FullyConnectedLayer &prev_layer);
    void forward(int j, FullyConnectedLayer &prev_layer);
    void backward(int j, FullyConnectedLayer &prev_layer);

    void update_params(const double learning_rate, const int batch_size);
    void reset_params()
    {
         std::fill(this->delt_kernel.begin(), this->delt_kernel.end(), 0.0);
        std::fill(this->delt_bias.begin(),   this->delt_bias.end(), 0.0);
    }
    void set_type(std::string const &s){type = s;}

    int get_map_count(){ return map_count;}
    std::vector<flt_type>::iterator get_map_data(){ return map_data.begin();}
    std::vector<flt_type>::iterator get_delt_map_data(){ return delt_map_data.begin();}

    void init_layer(int in_channels, int map_count, int batch_size,
            std::uniform_real_distribution<> &dis, std::mt19937 &gen);


private:
    std::string type;
    int map_count;
    int batch_size;

    std::vector<double> map_data; 
    std::vector<double> delt_map_data; 
    std::vector<double> bias;
    std::vector<double> delt_bias;
    std::vector<double> kernel; 
    std::vector<double> delt_kernel; 

    void init_kernel(std::uniform_real_distribution<> &dis, std::mt19937 &gen, 
        double* kernel, const int size, const double weight_base)
    {
        for(int i=0; i<size; i++)
            this->kernel[i] = (dis(gen)-0.5)*2*weight_base;
    }
};




// 激活函数及它们的导数
class ActivationFunction{
public:
    inline static double Tanh(double const val)
    {
        double ep = exp(val);
        double em = exp(-val);
        return (ep - em) / (ep + em);
    }
    
    inline static double DerivativeTanh(double const val)
    {
        return 1.0 - val*val;
    }
    
    inline static double ReLU(double const val)
    {
        return val > 0.0 ? val : 0.0;
    }
    
    inline static double DerivativeReLU(double const val)
    {
        return val > 0.0 ? 1.0 : 0.0;
    }
    
    inline double Sigmoid(double const val) 
    { 
        return 1.0 / (1.0 + exp(-val)); 
    }
    double DerivativeSigmoid(double const val){ 
        return val * (1.0 - val); 
    }
};

// 更新一个参数
inline double GradientDescent(double const W, double const dW, 
        double const alpha, double const lambda){
    return W - alpha * (dW + lambda * W);
}