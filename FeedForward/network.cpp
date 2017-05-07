#include "network.h"
using namespace ff;

void Network::addLayer(int s) {
  Layer l(s);
  layers.push_back(s);
}

void Network::linkLayers() {
  unsigned int x = 1;
  while (x < layers.size()) {

    layers[x].connect(&layers[x-1]);

    x++;
  }
}

void Network::process(vector<double> in) {

  layers[0].process(in);
  unsigned int x = 1;
  while (x < layers.size()) {

    layers[x].process();

    x++;
  }

}

vector<double> Network::getOutput() {
  return layers[layers.size()-1].getOutput();
}

void Network::backPropagate(TrainingExample te) {

  process(te.input);
  vector<double> errDelt;
  unsigned int x = 0;
  while (x < te.output.size()) {

    double d = getOutput()[x] - te.output[x];
    errDelt.push_back(d);

    x++;
  }

  layers[layers.size()-1].backPropagate(errDelt);

  x = layers.size()-2;
  while (x >= 0) {

    layers[x].backPropagate();

    x--;
  }

}

void Network::backPropagate(vector<double> errDelt) {

  layers[layers.size()-1].backPropagate(errDelt);

  int x = layers.size()-2;
  while (x >= 0) {

    layers[x].backPropagate();

    x--;
  }

}

vector<double> Network::getInputDeltas() {
    return layers[0].getDeltas();
}

void Network::gradientDescent(double lr) {

  unsigned int x = 0;
  while (x < layers.size()) {

    layers[x].gradientDescent(lr);

    x++;
  }

}

double Network::getError(TrainingExample te) {
  process(te.input);
  unsigned int x = 0;
  double err = 0;
  while (x < getOutput().size()) {

    err = err + pow(te.output[x]-getOutput()[x],2)/2;

    x++;
  }
  return err/getOutput().size();
}

void Network::save(string filename) {
    for (unsigned int x = 1; x < layers.size(); x++) {
        layers[x].save(filename+"layer_"+to_string(x));
    }
}

void Network::load(string filename) {
    for (unsigned int x = 1; x < layers.size(); x++) {
        layers[x].load(filename+"layer_"+to_string(x));
    }
}
