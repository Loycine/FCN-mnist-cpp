#pragma once
#include "fully_connected_layer.h"

using namespace std;
const int batch_size = 30;
const int num_classes = 10;

class FullyConnectednn{
    public:
        FullyConnectednn(int n) {layers.resize(n);}

        int get_layerscount() {return layers.size();}

        std::vector<FullyConnectedLayer>::iterator get_a_layer(int n)
        {
            if(n<get_layerscount() && n>=0) return layers.begin()+n;
            else
            {
                std::cout << "Wrong input of layer index \n";
                return layers.end();
            }
        }

        void forward_propagation(int j)
        {
            for(size_t i=0; i<this->layers.size()-1; i++)
                this->layers[i+1].forward(j, this->layers[i]);
        }

        void forward_propagation()
        {
            for(size_t i=0; i<this->layers.size()-1; i++)
                this->layers[i+1].forward(this->layers[i]);
        }

        void update_weights(double learning_rate)
        {
            for(int i=1; i<this->layers.size(); i++)
                this->layers[i].update_params(learning_rate, batch_size);
        }

        void reset_weights()
        {
            for(int i=1; i<this->layers.size(); i++)
                this->layers[i].reset_params();
        }

        void backward_propagation(int j, label_t label);

        void train(vector<one_image>& train_sample, vector<label_t>& train_label, double learning_rate);
        void predict(vector<one_image>& test_sample, vector<label_t>& test_label);

    private:
        vector<FullyConnectedLayer> layers;
        int find_max_likelihood_index()
        {
            int index = 0;
            vector<double>::iterator iter = this->layers.back().get_map_data();

            double max_val = *iter;
            for (int i=1; i<this->layers.back().get_map_count(); i++) {
                if(*(iter + i) > max_val) {
                    max_val = *(iter +i);
                    index = i;
                }
            }
            return index;
        }
};




class LossFunction{
public:
    inline static double SumSquaredError(const double y, const double t)
    { 
        return (y - t) * (y - t) / 2; 
    }
    inline static double DerivativeOfSumSquaredError(const double y, 
            const double t)
    { 
        return y - t;
    }
};