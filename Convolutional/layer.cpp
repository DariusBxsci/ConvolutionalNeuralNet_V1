#include "layer.h"
using namespace adb;
using namespace cnn;


vector<FeatureMap> Layer::getFeatureMaps() {
    return featureMaps;
}

void Layer::displayFeatureMaps() {
    for (unsigned int x = 0; x < output.size(); x++) {
        output[x].display();
    }
}

KernelLayer::KernelLayer(int numKernels, int numFeatures, int kx, int ky) { //initialize kernels
    setKernels(numKernels, numFeatures, kx, ky);
}

void KernelLayer::setKernels(int numKernels, int numFeatures, int kx, int ky) {
    kernels.resize(numFeatures);
    for (unsigned int s = 0; s < kernels.size(); s++) {
        for (int x = 0; x < numKernels; x++) {
            Kernel k(kx,ky,1);
            kernels[s].push_back(k);
        }
    }
}

vector<FeatureMap> KernelLayer::process(vector<FeatureMap> input) { //process each input feature map
    featureMaps = input;
    vector<FeatureMap> out;
    for (unsigned int i = 0; i < featureMaps.size(); i++) {
        for (unsigned int k = 0; k < kernels[0].size(); k++) {
            out.push_back(kernels[i][k].process(featureMaps[i]));
        }
    }
    output = out;
    return out;
}

vector<FeatureMap> KernelLayer::backPropagate(vector<FeatureMap>& next){ //reverse processing to back propagate
    for (unsigned int f = 0; f < featureMaps.size(); f++) {
        featureMaps[f].clearDelta();
    }

    int z = 0;
    for (unsigned int x = 0; x < kernels.size(); x++) {
        for (unsigned int y = 0; y < kernels[0].size(); y++) {
            kernels[x][y].backPropagate(featureMaps[x], next[z]);
            z++;
        }
    }
    return featureMaps;
}

void KernelLayer::gradientDescent(double lr) { //apply gradient descent to each kernel
    for (unsigned int x = 0; x < kernels.size(); x++) {
        for (unsigned int y = 0; y < kernels[0].size(); y++) {
            kernels[x][y].gradientDescent(lr);
        }
    }
}

void KernelLayer::save(string filename) {
    for (unsigned int x = 0; x < kernels.size(); x++) {
        for (unsigned int y = 0; y < kernels[0].size(); y++) {
            kernels[x][y].save(filename+"-"+to_string(x)+"_"+to_string(y));
        }
    }
}

void KernelLayer::load(string filename) {
    for (unsigned int x = 0; x < kernels.size(); x++) {
        for (unsigned int y = 0; y < kernels[0].size(); y++) {
            kernels[x][y].load(filename+"-"+to_string(x)+"_"+to_string(y));
        }
    }
}



FilterLayer::FilterLayer(Filter* f) {
    setFilter(f);
}

void FilterLayer::setFilter(Filter* f) {
    filter = f;
}

vector<FeatureMap> FilterLayer::process(vector<FeatureMap> input) {
    featureMaps = input;
    vector<FeatureMap> out;
    for (unsigned int x = 0; x < featureMaps.size(); x++) {
        out.push_back(filter->process(featureMaps[x]));
    }
    output = out;
    return out;
}

FilterLayer::~FilterLayer() {
    delete filter;
}

vector<FeatureMap> FilterLayer::backPropagate(vector<FeatureMap>& next) {
    for (unsigned int x = 0; x < featureMaps.size(); x++) {
        filter->backPropagate(featureMaps[x],next[x]);
    }
    return featureMaps;
}

void FilterLayer::gradientDescent(double lr) {
    //there are no parameters for a filter layer so no gradient descent here
}


ClassificationLayer::ClassificationLayer(int s1, int s2, int numclass) {
    classifier.addLayer(s1);
    classifier.addLayer(s2);
    classifier.addLayer(numclass);
    classifier.linkLayers();
}

vector<double> ClassificationLayer::flatten(vector<FeatureMap>& input) { //flatten feature maps into a sing 1D string of values
    vector<double> f;
    for (unsigned int x = 0; x < input.size(); x++) {
        vector<double> fl = input[x].flatten();
        f.insert(f.end(), fl.begin(), fl.end());
    }
    return f;
}


vector<FeatureMap> ClassificationLayer::deflatten(vector<double> input, vector<FeatureMap> last) { //deflatten 1D string of values into a vector of feature maps
    vector<FeatureMap> fmaps;
    fmaps.resize(input.size()/(inx*iny));
    unsigned int x = 0;
    while (x < fmaps.size()) {
        vector<double> df(input.begin()+inx*iny*(x), input.begin()+inx*iny*(x+1));
        fmaps[x].size(inx,iny);
        fmaps[x].delt_destringify(df,inx,iny);

        fmaps[x].image = last[x].image;

        x++;
    }
    return fmaps;
}

vector<FeatureMap> ClassificationLayer::process(vector<FeatureMap> input) {
    featureMaps = input;
    inx = input[0].image.size();
    iny = input[0].image[0].size();
    vector<double> flattened = flatten(input);

    classifier.process(flattened);
    flattened = classifier.getOutput();

    double e = 2.71828182845;
    double sumlogit = 0;
    for (unsigned int x = 0; x < flattened.size(); x++) { //apply softmax function to output of classifier
        sumlogit += pow(e,flattened[x]);
    }
    for (unsigned int x = 0; x < flattened.size(); x++) {
        flattened[x] = pow(e,flattened[x]) / sumlogit;
        if (isnan(flattened[x])) flattened[x] = 0;
    }

    vector<FeatureMap> out;
    out.resize(1);
    out[0].stringify(flattened);
    output = out[0];
    return out;
}

vector<FeatureMap> ClassificationLayer::backPropagate(vector<FeatureMap>& target) {
    vector<double> errDelt = output.getErrorMap(target[0]).flatten();
    classifier.backPropagate(errDelt);
    errDelt = classifier.getInputDeltas();

    vector<FeatureMap> deltaMaps = deflatten(errDelt, featureMaps);

    return deltaMaps;
}

void ClassificationLayer::gradientDescent(double lr) {
    classifier.gradientDescent(lr);
}

void ClassificationLayer::save(string filename) {
    classifier.save(filename);
}

void ClassificationLayer::load(string filename) {
    classifier.load(filename);
}
