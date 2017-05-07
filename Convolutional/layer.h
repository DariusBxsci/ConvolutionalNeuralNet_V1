#ifndef ADB_CONVNET_LAYER_H
#define ADB_CONVNET_LAYER_H

#include "kernel.h"
#include "filter.h"
#include "../FeedForward/network.h"

namespace adb {
namespace cnn {

class Layer { //this class is the base class for all layers in the network

    protected:

        vector<FeatureMap> featureMaps; //input feature maps
        vector<FeatureMap> output; //output feature maps

    public:

        vector<FeatureMap> getFeatureMaps();
        void displayFeatureMaps();

        virtual vector<FeatureMap> process(vector<FeatureMap> input)=0;
        virtual vector<FeatureMap> backPropagate(vector<FeatureMap>&)=0;
        virtual void gradientDescent(double)=0;
        virtual void save(string)=0;
        virtual void load(string)=0;
        virtual ~Layer(){}

};

class KernelLayer: public Layer { //layer consisting of convolution kernels

    private:

        vector<vector<Kernel>> kernels; //each first dimension represents a feature map, the second dimension represents a set of kernels for that feature map

    public:

        KernelLayer(int numKernels, int numFeatures, int kx, int ky);
        void setKernels(int numKernels, int numFeatures, int kx, int ky);
        vector<FeatureMap> process(vector<FeatureMap> input);
        vector<FeatureMap> backPropagate(vector<FeatureMap>&);
        void gradientDescent(double);
        void save(string filename);
        void load(string filename);
};

class FilterLayer: public Layer { //layer consisting of a simple filter applied to all input feature maps (pooling, relu, etc)

    private:

        Filter *filter; //this is the filter that this layer uses

    public:

        FilterLayer(Filter*);
        void setFilter(Filter*);
        vector<FeatureMap> process(vector<FeatureMap> input);
        vector<FeatureMap> backPropagate(vector<FeatureMap>&);
        void gradientDescent(double);
        void save(string filename){}
        void load(string filename){}
        ~FilterLayer();
};

class ClassificationLayer: public Layer { //This is the final classification layer.
    //This class is very messy, because it uses old feed forward neural network code

    private:

        ff::Network classifier;
        FeatureMap output;
        vector<FeatureMap> input;
        int inx; //size of the input feature maps (that will be flattened)
        int iny;

    public:

        ClassificationLayer(int s1, int s2, int numclass); //size 1, size 2, num classes
        vector<double> flatten(vector<FeatureMap>&);
        vector<FeatureMap> process(vector<FeatureMap> input); //will return n x 1 feature map at [0] of a vector
        vector<FeatureMap> deflatten(vector<double>, vector<FeatureMap>);
        vector<FeatureMap> backPropagate(vector<FeatureMap>&);
        void gradientDescent(double);
        void save(string filename);
        void load(string filename);
};

}
}

#endif
