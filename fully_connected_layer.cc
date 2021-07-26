#include "fully_connected_layer.h"

using namespace std;

void FullyConnectedLayer::init_layer(int in_channels, int map_count, int batch_size, std::uniform_real_distribution<> &dis, std::mt19937 &gen)
{
    const double scale = 6.0;
    int fan_in = 0, fan_out = 0;
    if(this->type.compare(std::string("fully_connected")) ==0 )
    {
        fan_in = in_channels;
        fan_out = map_count ;
        int denominator = fan_in + fan_out;
        double weight_base = 
            (denominator != 0) ? sqrt(scale / (double)denominator) : 0.5;
        this->map_count = map_count;
        this->batch_size = batch_size;

        int kernelcount = in_channels * map_count;
        int extendcount = kernelcount * batch_size;

        this->kernel.resize(kernelcount); 
        init_kernel(dis, gen, this->kernel.data(), kernelcount, weight_base);
        this->bias.resize(this->map_count, 0.0); 


        this->delt_kernel.resize(extendcount, 0.0);
        this->map_data.resize(this->map_count * batch_size, 0.0);
        this->delt_map_data.resize(this->map_count * batch_size, 0.0); 
        this->delt_bias.resize(this->map_count * batch_size, 0.0);
    }
    else if(this->type.compare(std::string("input")) == 0)
    {
        this->map_count = map_count;
        this->map_data.resize(this->map_count * batch_size, 0.0);
        this->delt_map_data.resize(this->map_count * batch_size, 0.0); 
    }
    else
    {
        std::cout<<"The type of this->("<<this->type<<") is not implemented! \n";
    }
    return;
}

void FullyConnectedLayer::forward(int j, FullyConnectedLayer &prev_layer)
{
    int M = this->map_count; 
    int N = prev_layer.map_count;

    int MM = j * M;
    int NN = j * N;

    for (int i = 0; i < this->map_count; i++) this->map_data[i + MM] = this->bias[i + MM]; 
    
    //Y(m) = kernel(m,n) * Y_prev(n) + bias(m) 
    cblas_dgemv(CblasRowMajor, CblasNoTrans, M, N, 1.0, this->kernel.data(), N, 
            prev_layer.map_data.data() + NN, 1, 1.0, this->map_data.data()+MM, 1);
    
    for (int i = 0; i < this->map_count; i++) 
        this->map_data[i+MM] = ActivationFunction::Tanh(this->map_data[i+MM]);
}


void FullyConnectedLayer::forward(FullyConnectedLayer &prev_layer)
{
    int M = this->map_count; 
    int N = prev_layer.map_count;

    for (int i = 0; i < this->map_count; i++) this->map_data[i] = this->bias[i]; 
    
    //Y(m) = kernel(m,n) * Y_prev(n) + bias(m) 
    cblas_dgemv(CblasRowMajor, CblasNoTrans, M, N, 1.0, this->kernel.data(), N, 
            prev_layer.map_data.data(), 1, 1.0, this->map_data.data(), 1);
    
    for (int i = 0; i < this->map_count; i++) 
        this->map_data[i] = ActivationFunction::Tanh(this->map_data[i]);
}


void FullyConnectedLayer::backward(int batch_index, FullyConnectedLayer &prev_layer)
{
    // delta x
    int M = this->map_count; 
    int N = prev_layer.map_count;
    
    int MM = batch_index * M;
    int NN = batch_index * N;
    int MN = batch_index * M * N;

    // d_prev(n) = σ( kernel(m,n)T * d(m) ) 
    cblas_dgemv(CblasRowMajor, CblasTrans, M, N, 1.0, this->kernel.data(), N, 
            this->delt_map_data.data() + MM, 1, 0.0, prev_layer.delt_map_data.data()+NN, 1);
    for (int i = 0; i < prev_layer.map_count; i++)
    {
        prev_layer.delt_map_data[i+NN] *= 
            ActivationFunction::DerivativeTanh(prev_layer.map_data[i+NN]);
    }
    
    // dW
    // dw(j, i) = d[j] * y[i]
    #pragma omp parallel for
    for (int j = 0; j < this->map_count; j++) 
    {
        for (int i = 0; i < prev_layer.map_count; i++) 
        {
            this->delt_kernel[j*prev_layer.map_count + i + MN] += 
                this->delt_map_data[j + MM] * prev_layer.map_data[i + NN];
        }
    }
    
    // db
    // db = d[i]
    for (int i = 0; i < this->map_count; i++) {
        this->delt_bias[i + MM] += this->delt_map_data[i + MM];
    }
}

// 更新可训练参数
void FullyConnectedLayer::update_params(double const learning_rate, 
        int const batch_size)
{
    double const lambda = 0.0;
    if(this->kernel.size() != 0)
    {
        #pragma omp parallel for
        for (size_t i = 1; i < this->kernel.size() ; i++)
        {
            for(int j=0; j<batch_size; j++)
                this->delt_kernel[i] += this->delt_kernel[i+j*this->kernel.size()];
        }
    }
    for (int i = 1; i < this->map_count; i++) {
        for(int j=0; j<batch_size; j++)
            this->delt_bias[i] += this->delt_bias[i+j*this->map_count];
    }

    if(this->kernel.size() != 0)
    {
        for (size_t i = 0; i < this->kernel.size() ; i++)
            this->kernel[i] = GradientDescent(this->kernel[i], 
                    this->delt_kernel[i] / (double)batch_size, learning_rate, lambda);
    }
    for (int i = 0; i < this->map_count; i++) {
        this->bias[i] = GradientDescent(this->bias[i], 
                this->delt_bias[i] / (double)batch_size, learning_rate, lambda);
    }
}
