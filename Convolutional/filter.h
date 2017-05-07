#ifndef ADB_CONVNET_FILTER_H
#define ADB_CONVNET_FILTER_H

#include "featureMap.h"

namespace adb {
namespace cnn {

class Filter { //all filter classes will inherit from this class

    protected:

        vector<vector<double>> delta;

    public:

        virtual FeatureMap process(FeatureMap&)=0;
        virtual void backPropagate(FeatureMap&,FeatureMap&)=0;
        virtual ~Filter(){}

};

class Maxpool: public Filter {

    private:

    public:

        double oneMaxpool(FeatureMap&, int, int);
        FeatureMap process(FeatureMap&);
        void oneBprop(FeatureMap&, double, int, int);
        void backPropagate(FeatureMap&,FeatureMap&);

};

class Subsample: public Filter {

    private:

    public:

        double oneSubsample(FeatureMap&, int, int);
        FeatureMap process(FeatureMap&);
        void oneBprop(FeatureMap&, double, int, int);
        void backPropagate(FeatureMap&,FeatureMap&);

};

class Relu: public Filter {

    private:

    public:

        double oneRelu(FeatureMap&, int, int);
        FeatureMap process(FeatureMap&);
        void oneBprop(FeatureMap&, double, int, int);
        void backPropagate(FeatureMap&,FeatureMap&);

};

}
}

#endif
