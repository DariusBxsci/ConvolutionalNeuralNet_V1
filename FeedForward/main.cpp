#include "network.h"
using namespace std;

int main() {

  srand(time(NULL));

  ff::Network net;
  net.addLayer(2);
  net.addLayer(3);
  net.addLayer(1);
  net.linkLayers();

  ff::TrainingExample te0;
  te0.addInput(0);
  te0.addInput(0);
  te0.addOutput(0);

  ff::TrainingExample te1;
  te1.addInput(1);
  te1.addInput(0);
  te1.addOutput(1);

  ff::TrainingExample te2;
  te2.addInput(0);
  te2.addInput(1);
  te2.addOutput(0);

  ff::TrainingExample te3;
  te3.addInput(1);
  te3.addInput(1);
  te3.addOutput(1);

  int x = 0;
  double err = 1;
  while (err > 0.001) {

    int n = rand()%4;

    if (n == 0) net.backPropagate(te0);
    if (n == 1) net.backPropagate(te1);
    if (n == 2) net.backPropagate(te2);
    if (n == 3) net.backPropagate(te3);

    net.gradientDescent(0.2);

    err = ( net.getError(te0) + net.getError(te1)+ net.getError(te2)+ net.getError(te3) )/4.0;
    cout << "Error: " << err << endl;

    x++;
  }

  net.process(te0.input);
  cout << "Output: " << net.getOutput()[0] << endl;

  net.process(te1.input);
  cout << "Output: " << net.getOutput()[0] << endl;

  net.process(te2.input);
  cout << "Output: " << net.getOutput()[0] << endl;

  net.process(te3.input);
  cout << "Output: " << net.getOutput()[0] << endl;

  return 0;
}
