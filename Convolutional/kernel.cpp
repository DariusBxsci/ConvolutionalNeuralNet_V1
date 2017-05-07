#include "kernel.h"
using namespace adb;
using namespace cnn;

Kernel::Kernel() {
}

Kernel::Kernel(int xsize, int ysize, int stepsize) {
    init(xsize, ysize, stepsize);
}

void Kernel::clearDelta() {
    for (unsigned int x = 0; x < delta.size(); x++) {
        for (unsigned int y = 0; y < delta[0].size(); y++) {
            delta[x][y] = 0;
        }
    }
    bdelta = 0;
}

void Kernel::clearInput() {
    for (unsigned int x = 0; x < input.size(); x++) {
        for (unsigned int y = 0; y < input[0].size(); y++) {
            input[x][y] = 0;
        }
    }
}

void Kernel::init(int xsize, int ysize, int stepsize) {
    step = stepsize;
    kernel.resize(xsize);
    delta.resize(xsize);
    input.resize(xsize);
    lrMod.resize(xsize);
    lastDelta.resize(xsize);
    for (int x = 0; x < xsize; x++) {
        kernel[x].resize(ysize);
        delta[x].resize(ysize);
        input[x].resize(ysize);
        lrMod[x].resize(ysize);
        lastDelta[x].resize(ysize);
        for (unsigned int y = 0; y < kernel[x].size(); y++) {
            kernel[x][y] = (((rand()%1200)-600)/300.0);
            //kernel[x][y] = 1;

            delta[x][y] = 0;
            lrMod[x][y] = 1;
        }
    }
    bias = ((rand()%600)/60000.0);
    bdelta = 0;
}

double Kernel::oneConv(vector<vector<double>>& image, int xp, int yp) { //perform a single convolution of the image (at point xp,yp)
    double c = 0;
    for (unsigned int x = 0; x < kernel.size(); x++) {
        for (unsigned int y = 0; y < kernel[0].size(); y++) {
            double nv = 0;
            if (x+xp < image.size() && y+yp < image[0].size()) nv = image[x+xp][y+yp];
            c += kernel[x][y] * nv;
        }
    }
    c = c/(kernel.size()*kernel[0].size()) + bias;

    return c;
}

void Kernel::oneDeltaConv(FeatureMap& prev, int xp, int yp) { //perform a single convolution of the image (at point xp,yp)
    for (unsigned int x = 0; x < kernel.size(); x++) {
        for (unsigned int y = 0; y < kernel[0].size(); y++) {
            if (x+xp < prev.image.size() && y+yp < prev.image[0].size()) prev.delta[x+xp][y+yp] += delta[x][y] / (kernel.size()*kernel[0].size());
        }
    }
}

FeatureMap Kernel::process(FeatureMap& input) {
    FeatureMap out;
    out.size(input.image.size(),input.image[0].size());
    for (unsigned int x = 0; x < out.image.size(); x++) {
        for (unsigned int y = 0; y < out.image[0].size(); y++) {
            out.image[x][y] = oneConv(input.image, x-(kernel.size()-1)/2, y-(kernel[0].size()-1)/2);
        }
    }
    return out;
}

void Kernel::oneBprop(FeatureMap& prev, FeatureMap& next, int px, int py) { //px and py are coordinates on the previous map
    for (unsigned int x = 0; x < kernel.size(); x++) {
        for (unsigned int y = 0; y < kernel[0].size(); y++) {

            double pi = 0;
            if (x+px < prev.image.size() && y+py < prev.image[0].size()) pi = prev.image[x+px][y+py];
            double d =  pi * bdelta / (kernel.size()*kernel[0].size());

            delta[x][y] += d;
        }
    }
}

void Kernel::backPropagate(FeatureMap& prev, FeatureMap& next) {
    clearDelta();

    for (unsigned int x = 0; x < next.delta.size(); x++) {
        for (unsigned int y = 0; y < next.delta[0].size(); y++) {
            bdelta += next.delta[x][y];
        }
    }
    for (unsigned int x = 0; x < next.delta.size(); x++) {
        for (unsigned int y = 0; y < next.delta[0].size(); y++) {
            oneBprop(prev, next, x, y);
        }
    }

    for (unsigned int x = 0; x < prev.delta.size(); x++) {
        for (unsigned int y = 0; y < prev.delta[0].size(); y++) {
            oneDeltaConv(prev, x, y);
        }
    }
}

void Kernel::flip() {
    vector<vector<double>> nd;
    vector<vector<double>> nk;
    for (unsigned int x = 0; x < delta.size(); x++) {
        nd.resize(delta.size());
        nk.resize(kernel.size());
        for (unsigned int y = 0; y < delta[0].size(); y++) {
            nd[x].resize(delta[0].size());
            nk[x].resize(kernel[0].size());
            nd[x][y] = delta[delta.size()-1-x][delta[0].size()-1-y];
            nk[x][y] = kernel[kernel.size()-1-x][kernel[0].size()-1-y];
        }
    }
    for (unsigned int x = 0; x < delta.size(); x++) {
        for (unsigned int y = 0; y < delta[0].size(); y++) {
            delta[x][y] = nd[x][y];
            kernel[x][y] = nk[x][y];
        }
    }
}

void Kernel::gradientDescent(double lr) {
    double avglrMod = 0;
    for (unsigned int x = 0; x < delta.size(); x++) {
        for (unsigned int y = 0; y < delta[0].size(); y++) {
            //cout << "KERNEL DELTA " << delta[x][y] << endl;
            /*if (delta[x][y] > 1) delta[x][y] = 1;
            if (delta[x][y] < -1) delta[x][y] = -1;*/
            //cout << "FROM " << kernel[x][y] << " TO " << kernel[x][y]-(delta[x][y]*lr) << endl;

            if ((lastDelta[x][y] > 0 && delta[x][y] > 0) || (lastDelta[x][y] < 0 && delta[x][y] < 0)) lrMod[x][y] += 0.01;
            if ((lastDelta[x][y] < 0 && delta[x][y] > 0) || (lastDelta[x][y] > 0 && delta[x][y] < 0)) lrMod[x][y] *= 0.99;
            //if (lrMod[x][y] > 10) lrMod[x][y] = 10;
            //if (lrMod[x][y] < 0.1) lrMod[x][y] = 0.1;

            //cout << lrMod[x][y] << endl;
            kernel[x][y] = kernel[x][y]-(delta[x][y]*lr*lrMod[x][y]);
            //kernel[x][y] = kernel[x][y]-(delta[x][y]*lr);
            avglrMod += lrMod[x][y]/(delta.size()*delta[0].size());

            //lastDelta[x][y] = delta[x][y];
            //cout << "NEW KVAL " << kernel[x][y] << endl;
        }
    }
    //cout << "BDELTA " << bdelta << endl;
    //cout << "Bias " << bias << " with delta " << bdelta << endl;
    //if (bdelta > 1) bdelta = 0;
    //if (bdelta < -1) bdelta = 0;
    bias = bias-(bdelta*lr*avglrMod);
    //bias = bias-(bdelta*lr);
}

void Kernel::save(string filename) {
    string ks = "";
    for (unsigned int x = 0; x < kernel.size(); x++) {
        for (unsigned int y = 0; y < kernel[0].size(); y++) {
            ks += to_string(kernel[x][y]) + " ";
        }
    }
    ks += to_string(bias) + " ";
    ofstream out(filename);
    out << ks;
}

void Kernel::load(string filename) {
    ifstream in(filename);
    string content( (istreambuf_iterator<char>(in)),
                       (istreambuf_iterator<char>()) );
    vector<double> lkv;
    string buf = "";
    for (unsigned int x = 0; x < content.length(); x++) {
        if (content.at(x) == ' ') {
            lkv.push_back(stod(buf));
            buf = "";
        }
        else {
            buf += content.at(x);
        }
    }
    int z = 0;
    for (unsigned int x = 0; x < kernel.size(); x++) {
        for (unsigned int y = 0; y < kernel[0].size(); y++) {
            kernel[x][y] = lkv[z];
            z++;
        }
    }
    bias = lkv[z];
}

