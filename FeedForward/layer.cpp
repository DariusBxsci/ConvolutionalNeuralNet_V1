#include "layer.h"
using namespace ff;

Layer::Layer() {
}

Layer::Layer(int s) {
  init(s);
}

void Layer::init(int s) {
  neurons.resize(s);
}

void Layer::process() {
  output.resize(neurons.size());
  unsigned int x = 0;
  while (x < neurons.size()) {

    neurons[x].process();
    output[x] = neurons[x].getValue();

    x++;
  }
}

void Layer::process(vector<double> in) {
  output.resize(neurons.size());
  unsigned int x = 0;
  while (x < neurons.size()) {

    neurons[x].process(in[x]);
    output[x] = neurons[x].getValue();

    x++;
  }
}

void Layer::connect(Layer* l) { //connect layer to the back of this one

  unsigned int x = 0;
  while (x < neurons.size()) {

    unsigned int y = 0;
    while (y < l->getNeurons()->size()) {

      neurons[x].connect(&l->getNeurons()->at(y));

      y++;
    }

    y = 0;
    x++;
  }

}

vector<Neuron>* Layer::getNeurons() {
  return &neurons;
}

vector<double> Layer::getOutput() {
  return output;
}

vector<double> Layer::getDeltas() {
    vector<double> d;
    for (unsigned int x = 0; x < neurons.size(); x++) {
        d.push_back(neurons[x].getDelta());
    }
    return d;
}

void Layer::backPropagate() {
  unsigned int x = 0;
  while (x < neurons.size()) {
    neurons[x].backPropagate();
    x++;
  }
}

void Layer::backPropagate(vector<double> d) {
  unsigned int x = 0;
  while (x < neurons.size()) {
    neurons[x].backPropagate(d[x]);
    x++;
  }
}

void Layer::gradientDescent(double lr) {
  unsigned int x = 0;
  while (x < neurons.size()) {
    neurons[x].gradientDescent(lr);
    x++;
  }
}

int Layer::getSize() {
  return neurons.size();
}

void Layer::save(string filename) {
    string ls = "";
    for (unsigned int x = 0; x < neurons.size(); x++) {
        ls += neurons[x].getWeightString() + "\n";
    }
    ofstream out(filename);
    out << ls;
}

void Layer::load(string filename) {
    ifstream in(filename);
    unsigned int x = 0;
    string line;
    while (getline(in, line) && x < neurons.size()) {
        neurons[x].loadWeights(line);
        x++;
    }
}
