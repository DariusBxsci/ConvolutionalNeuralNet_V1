#include "filter.h"
using namespace adb;
using namespace cnn;

//maxpool
double Maxpool::oneMaxpool(FeatureMap& input, int x, int y) {
    double highest = input.image[x][y];
    for (int ix = 0; ix < 2; ix++) {
        for (int iy = 0; iy < 2; iy++) {
            if (input.image[x+ix][y+iy] > highest) highest = input.image[x+ix][y+iy];
        }
    }
    return highest;
}

FeatureMap Maxpool::process(FeatureMap& input) {
    FeatureMap output;
    output.size(floor(input.image.size()/2),floor(input.image[0].size()/2));

    int x = 0, y = 0;
    for (unsigned int ix = 0; ix < input.image.size()-1; ix+=2) {
        y = 0;
        for (unsigned int iy = 0; iy < input.image[0].size()-1; iy+=2) {
            output.image[x][y] = oneMaxpool(input, ix, iy);
            y++;
        }
        x++;
    }
    return output;
}

void Maxpool::oneBprop(FeatureMap& input, double delta, int x, int y) {
    double highest = input.image[x][y];
    int highestx = x;
    int highesty = y;
    for (int ix = 0; ix < 2; ix++) {
        for (int iy = 0; iy < 2; iy++) {
            if (input.image[x+ix][y+iy] > highest) {
                highest = input.image[x+ix][y+iy];
                highestx = x+ix;
                highesty = y+iy;
            }
        }
    }

    input.delta[highestx][highesty] = delta;
}


void Maxpool::backPropagate(FeatureMap &prev, FeatureMap &next) {

    //find the points in the next featuremap that correspond with the next one
    // Forward prop: prev >> filter >> next
    // Back prop: next >> filter >> prev
    // prev will end up storing the deltas for it's pixels

    prev.clearDelta();
    for (unsigned int x = 0; x < next.delta.size(); x++) {
        for (unsigned int y = 0; y < next.delta[0].size(); y++) {
            //cout << "x " << x << ", y " << y << endl;
            //cout << "NDELT " << next.delta[x][y] << endl;
            oneBprop(prev, next.delta[x][y], x*2, y*2); //test this b
            //cout << "   FILTER " << next.delta[x][y] << endl;
        }
    }

}


//subsample
double Subsample::oneSubsample(FeatureMap& input, int x, int y) {
    double total = 0;
    for (int ix = 0; ix < 2; ix++) {
        for (int iy = 0; iy < 2; iy++) {
            total += input.image[x+ix][y+iy];
        }
    }
    return total/4;
}

FeatureMap Subsample::process(FeatureMap& input) {
    FeatureMap output;
    output.size(floor(input.image.size()/2),floor(input.image[0].size()/2));

    int x = 0, y = 0;
    for (unsigned int ix = 0; ix < input.image.size()-1; ix+=2) {
        y = 0;
        for (unsigned int iy = 0; iy < input.image[0].size()-1; iy+=2) {
            output.image[x][y] = oneSubsample(input, ix, iy);
            y++;
        }
        x++;
    }
    return output;
}

void Subsample::oneBprop(FeatureMap& input, double delta, int x, int y) {
    for (int ix = 0; ix < 2; ix++) {
        for (int iy = 0; iy < 2; iy++) {
            input.delta[x+ix][y+iy] = delta;
        }
    }

}


void Subsample::backPropagate(FeatureMap &prev, FeatureMap &next) {
    for (unsigned int x = 0; x < next.delta.size(); x++) {
        for (unsigned int y = 0; y < next.delta[0].size(); y++) {
            oneBprop(prev, next.delta[x][y], x*2, y*2);
        }
    }

}


//Relu
double Relu::oneRelu(FeatureMap& input, int x, int y) {
    if (input.image[x][y] <= 0) return 0;
    return input.image[x][y];
}

FeatureMap Relu::process(FeatureMap& input) {
    FeatureMap output;
    output.size(input.image.size(),input.image[0].size());
    int x = 0, y = 0;
    for (unsigned int ix = 0; ix < input.image.size()-1; ix++) {
        y = 0;
        for (unsigned int iy = 0; iy < input.image[0].size()-1; iy++) {
            output.image[x][y] = oneRelu(input, ix, iy);
            y++;
        }
        x++;
    }
    return output;
}

void Relu::oneBprop(FeatureMap& input, double delta, int x, int y) {
    if (input.image[x][y] <= 0) input.delta[x][y] = 0;
    else input.delta[x][y] = delta;
}


void Relu::backPropagate(FeatureMap &prev, FeatureMap &next) {
    prev.clearDelta();
    for (unsigned int x = 0; x < next.delta.size(); x++) {
        for (unsigned int y = 0; y < next.delta[0].size(); y++) {
            oneBprop(prev, next.delta[x][y], x, y);
        }
    }

}
