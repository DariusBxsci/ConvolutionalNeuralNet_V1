#ifndef ADB_CONVNET_NETWORK_H
#define ADB_CONVNET_NETWORK_H

#include "layer.h"

struct xy {
    int x;
    int y;
    void set(int g, int h) {
        x = g;
        y = h;
    }
    void print() {
        cout << "x: " << x << ", y: " << y << endl;
    }
};

namespace adb {
namespace cnn {

struct TrainingElement { //training element

    FeatureMap input;
    vector<double> output;

};

using TrainingSet = vector<TrainingElement>; //training set is just a vector of training elements

class ImageTrainingSet { //class to help for loading in images into the training set

    private:

        TrainingSet tset;

    public:

        void load(string dirname, string ftype, vector<double> output) {
            vector<cv::String> fn;
            glob(dirname + "/*." + ftype, fn, false);

            vector<Mat> images;
            size_t c = fn.size(); //number of files in images folder
            for (size_t i=0; i < c; i++) {
                TrainingElement te;
                te.input.loadImage(imread(fn[i], CV_LOAD_IMAGE_COLOR));
                te.output = output;
                tset.push_back(te);
            }
        }

        TrainingSet getSet() {
            return tset;
        }

};

class Network {

    private:

        vector<Layer*> layers;
        FeatureMap output;

        int getHighest(vector<double>);

    public:

        Network();
        void addLayer(Layer*);
        FeatureMap process(FeatureMap);
        FeatureMap getOutput();
        double getError(FeatureMap, vector<double>);
        double getError(TrainingSet);
        double getClassError(FeatureMap, vector<double>);
        double getClassError(TrainingSet);
        void backPropagate(FeatureMap,vector<double>);
        void gradientDescent(double);
        void train(TrainingSet,int,double);
        void inspectLayer(int);
        int getClass();
        void displayOutput();
        xy getLayerSize(int);
        void save(string);
        void load(string);
        ~Network();


};

}
}

#endif
