#ifndef FEEDFORWARD_LAYER_H
#define FEEDFORWARD_LAYER_H

#include "neuron.h"

namespace ff {

  class Layer {

    private:

      vector<Neuron> neurons;
      vector<double> output;

    public:

      Layer();
      Layer(int);
      void init(int);
      void process();
      void process(vector<double>);
      void connect(Layer*); //connect layer to the back of this one
      vector<Neuron>* getNeurons();
      vector<double> getOutput();
      vector<double> getDeltas();
      void backPropagate();
      void backPropagate(vector<double>);
      void gradientDescent(double);
      int getSize();

      void save(string);
      void load(string);

  };

}

#endif
