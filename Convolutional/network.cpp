#include "network.h"
using namespace adb;
using namespace cnn;

Network::Network() {
    srand(time(0));
}

void Network::addLayer(Layer *l) {
    layers.push_back(l);
}

FeatureMap Network::process(FeatureMap input) {
    input.squash(); //squash the input from 0-255 to 0-1 pixel values
    vector<FeatureMap> buf = {input};
    for (unsigned int l = 0; l < layers.size(); l++) {
        buf = layers[l]->process(buf); //pass each set of feature maps to the next layer
    }

    output = buf[0];

    return output;
}

FeatureMap Network::getOutput() {
    return output;
}

double Network::getError(FeatureMap input, vector<double> target) { //get error of one input vs output
    process(input);
    vector<double> foutput = output.flatten();
    double err = 0;
    for (unsigned int x = 0; x < foutput.size(); x++) {
        double e = (target[x]*log(foutput[x])); //cross-entropy loss function
        if (isnan(e)) e = 0;
        err -= e;

    }
    return err/foutput.size();
}

double Network::getError(TrainingSet input) { //get error of entire training set
    double err = 0;
    for (unsigned int x = 0; x < input.size(); x++) {
        err += getError(input[x].input, input[x].output);
    }
    return err/input.size();
}

double Network::getClassError(FeatureMap input, vector<double> target) { //get classification error 0 or 1
    process(input);
    if (getClass() == getHighest(target)) { return 0; }
    else { return 1; }
}

double Network::getClassError(TrainingSet input) {
    double err = 0;
    for (unsigned int x = 0; x < input.size(); x++) {
        err += getClassError(input[x].input, input[x].output);
    }
    return err/input.size();
}


void Network::backPropagate(FeatureMap input,vector<double> target) { //reverse the forward propagation procedure to back propagate
    process(input);
    FeatureMap errDelt;
    errDelt.stringify(target);
    vector<FeatureMap> buf;
    buf.push_back(errDelt);
    for (int x = layers.size()-1; x >= 0; x--) {
        buf = layers[x]->backPropagate(buf);
    }
}

void Network::gradientDescent(double lr) {
    for (unsigned int x = 0; x < layers.size(); x++) {
        layers[x]->gradientDescent(lr); //apply gradient descent to each layer
    }
}

void Network::train(TrainingSet ts, int iter, double lr) { //combine gradient descent and back propagation to train model
    for (int x = 0; x < iter; x++) {
        int r = rand()%ts.size();
        backPropagate(ts[r].input, ts[r].output);
        gradientDescent(lr);
    }
}

void Network::inspectLayer(int l) {
    layers[l]->displayFeatureMaps();
}

xy Network::getLayerSize(int l) {
    xy c;
    c.set(layers[l]->getFeatureMaps()[0].image.size(),layers[l]->getFeatureMaps()[0].image[0].size());
    return c;
}

int Network::getClass() { //return which class was predicted
    return getHighest(output.flatten());
}

int Network::getHighest(vector<double> input) {
    int highest = 0;
    unsigned int x = 0;
    while (x < input.size()) {
        if (input[x] > input[highest]) highest = x;
        x++;
    }
    return highest;
}

void Network::displayOutput() {
    vector<double> o = output.flatten();
    for (unsigned int x = 0; x < o.size(); x++) {
        cout << "Class " << x << ": " << o[x] << endl;
    }
}

void Network::save(string directory) { //save model
    ofstream file(directory+"/model.txt");
    for (unsigned int x = 0; x < layers.size(); x++) {
        layers[x]->save(directory+"Layer"+to_string(x));
    }
}

void Network::load(string directory) { //load model
    for (unsigned int x = 0; x < layers.size(); x++) {
        layers[x]->load(directory+"Layer"+to_string(x));
    }
}

Network::~Network() { //delete pointer vector
    for (auto it = layers.begin(); it != layers.end(); ++it){
        delete *it;
    }
}
