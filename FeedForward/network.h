#ifndef FEEDFORWARD_NETWORK_H
#define FEEDFORWARD_NETWORK_H

#include "layer.h"

namespace ff {

  struct TrainingExample {

    vector<double> input;
    vector<double> output;

    void addInput(double in) {
      input.push_back(in);
    }
    void addOutput(double in) {
      output.push_back(in);
    }
  };

  class Network { //this is a very old class for a feed forward neural network
                    //It's not very good, but I had to use it due to a time constraint
                    // for finishing this project.

    private:

      vector<Layer> layers;

    public:

      void addLayer(int);
      void linkLayers();
      void process(vector<double>);
      vector<double> getOutput();
      void backPropagate(TrainingExample);
      void backPropagate(vector<double>);
      vector<double> getInputDeltas();
      void gradientDescent(double);
      double getError(TrainingExample);

      void save(string);
      void load(string);

  };

}

#endif
