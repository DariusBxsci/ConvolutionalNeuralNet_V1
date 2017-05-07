#include "../Convolutional/network.h"
using namespace adb;

int main() {

    cnn::Network convnet;
    convnet.addLayer(new cnn::KernelLayer(3, 1, 3, 3)); // 3 5x5 per each feature map (1 feature map) kernels
    convnet.addLayer(new cnn::FilterLayer(new cnn::Maxpool));
    convnet.addLayer(new cnn::FilterLayer(new cnn::Maxpool));
    convnet.addLayer(new cnn::FilterLayer(new cnn::Maxpool));
    convnet.addLayer(new cnn::FilterLayer(new cnn::Maxpool));
    convnet.addLayer(new cnn::FilterLayer(new cnn::Maxpool));
    convnet.addLayer(new cnn::ClassificationLayer(729,360,2));

    cnn::FeatureMap in;
    in.loadImage("/mnt/programming/AxtyaNet/danq.png",450,450);
    in.display();

    //convnet.inspectLayer(6);
    convnet.process(in);
    /*convnet.process(in);
    convnet.getOutput().display();
    convnet.process(in);
    convnet.getOutput().display();*/

    xy c = convnet.getLayerSize(4);
    cout << "x: " << c.x << ", y: " << c.y << endl;


    convnet.getOutput().display();
    cout << "Error: " << convnet.getError(in,{50,150}) << endl;
    int t = 0;
    double err = 1;
    while (t < 100000 && err > 0.01) {
        convnet.backPropagate(in,{50,150});
        convnet.gradientDescent(0.4);
        err = convnet.getError(in,{50,150});
        cout << "Error: " << err << endl;
        t++;
    }
    convnet.getOutput().display();
    cout << "Training done after " << t << " iterations!" << endl;

    //vector<vector<double>> bp = { {255,255,255,255},{255,255,255,255},{255,255,255,255},{255,255,255,255} };

    cout << "DONE" << endl;
    return 0;
}
