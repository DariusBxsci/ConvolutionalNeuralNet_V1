#ifndef ADB_CONVNET_KERNEL_H
#define ADB_CONVNET_KERNEL_H

#include "featureMap.h"

namespace adb {
namespace cnn {

class Kernel { //rectangle shaped matrix for use in a convolution on a feature map

    private:

        vector<vector<double>> kernel;
        vector<vector<double>> delta;
        vector<vector<double>> lastDelta;
        vector<vector<double>> lrMod;
        vector<vector<double>> input;
        void clearDelta();
        void clearInput();
        int step;

        double bdelta;
        double bias;

    public:

        Kernel();
        Kernel(int xsize, int ysize, int stepsize);
        void init(int xsize, int ysize, int stepsize);
        double oneConv(vector<vector<double>>& image, int x, int y);
        FeatureMap process(FeatureMap&);
        void oneBprop(FeatureMap&, FeatureMap&, int, int);
        void oneDeltaConv(FeatureMap&, int, int);
        void oneBprop(FeatureMap&);
        void backPropagate(FeatureMap&, FeatureMap&);
        void gradientDescent(double);
        void flip();
        void save(string filename);
        void load(string filename);

};

}
}

#endif
