#ifndef FEEDFORWARD_NEURON_H
#define FEEDFORWARD_NEURON_H

#include "activation.h"
using namespace adb;

namespace ff {

  class Neuron;

  class Weight {

    private:

      double input;
      double value;
      double delta;
      double lastDelta;
      double lrMod;
      double bias;
      double bdelta;

    public:

    double weight;
      Neuron* destination;
      Weight();
      void process();
      void backPropagate(double);
      void gradientDescent(double);
      double getValue();
      double *getWeight();
      Neuron* getDestination();

  };

  class Neuron {

    private:

      vector<Neuron*> forwardLinks;
      vector<Weight> backwardLinks;

      double value;
      double delta;
      double input;
      double bias;
      double bdelta;
      Activation *activation;

    public:

      Neuron();
      void connect(Neuron*); //connect neuron to the back of this one
      void forwardLink(Neuron*);
      double getValue();
      double getDelta();
      void process();
      void process(double);
      void backPropagate();
      void backPropagate(double);
      void gradientDescent(double);
      string getWeightString();
      void loadWeights(string);
      ~Neuron();

  };

}

#endif
