#include "neuron.h"
using namespace ff;

Weight::Weight() {
  weight = ((rand()%20000)-10000)/50000.0;
  bias = ((rand()%20000))/1000000.0;
  input = 0;
  value = 0;
}

void Weight::process() {
  input = destination->getValue();
  //value = input * weight + bias;
  value = input * weight;

}

double Weight::getValue() {
  return value;
}

double *Weight::getWeight() {
    return &weight;
}

Neuron* Weight::getDestination() {
  return destination;
}

void Weight::backPropagate(double d) {
    lastDelta = delta;
  delta = d * input;
  bdelta = d;
}

void Weight::gradientDescent(double lr) {
    if ((lastDelta > 0 && delta > 0) || (lastDelta < 0 && delta < 0)) lrMod += 0.01;
    if ((lastDelta < 0 && delta > 0) || (lastDelta > 0 && delta < 0)) lrMod *= 0.99;
    weight -= delta*lr*lrMod;
}


Neuron::Neuron() {
  activation = new Affine; //affine means no activation function (just weights)
}

void Neuron::connect(Neuron* n) { //connect neuron to the back of this one
  Weight w;
  w.destination = n;
  backwardLinks.push_back(w);
  n->forwardLink(this);
}

void Neuron::forwardLink(Neuron* n) {
  forwardLinks.push_back(n);
}

double Neuron::getValue() {
  return value;
}

double Neuron::getDelta() {
  return delta;
}

void Neuron::process() {
  double v = 0;
  for (unsigned int x = 0; x < backwardLinks.size(); x++) {
    backwardLinks[x].process();
    v += backwardLinks[x].getValue();
  }
  input = v;
  value = activation->activate(v);
}

void Neuron::process(double in) {
  input = in;
  value = activation->activate(input);
}

void Neuron::backPropagate() {

  double d = 0;
  for (unsigned int x = 0; x < forwardLinks.size(); x++) {
    d += forwardLinks[x]->getDelta();
  }


  delta = activation->derive(input) * d;

  for (unsigned int x = 0; x < backwardLinks.size(); x++) {
    backwardLinks[x].backPropagate(delta);
  }
}

void Neuron::gradientDescent(double lr) {
  for (unsigned int x = 0; x < backwardLinks.size(); x++) {
    backwardLinks[x].gradientDescent(lr);
  }
}

void Neuron::backPropagate(double d) {

  delta = activation->derive(input) * d;

  for (unsigned int x = 0; x < backwardLinks.size(); x++) {
    backwardLinks[x].backPropagate(delta);
  }
}

string Neuron::getWeightString() {
    string wv = "";
    for (unsigned int x = 0; x < backwardLinks.size(); x++) {
        wv += to_string(*backwardLinks[x].getWeight()) + " ";
    }
    return wv;
}

void Neuron::loadWeights(string wv) {
    string buf = "";
    int w = 0;
    for (unsigned int x = 0; x < wv.length(); x++) {
        if (wv.at(x) == ' ') {
            backwardLinks[w].weight = stod(buf);
            buf = "";
            w++;
        }
        else {
            buf += wv.at(x);
        }
    }
}

Neuron::~Neuron() {
  delete activation;
}
