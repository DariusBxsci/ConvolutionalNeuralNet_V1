#ifndef ADB_ACTIVATION_H
#define ADB_ACTIVATION_H

#include "../includes.h"
using namespace std;

namespace adb{

  class Activation { //base class for all activation functions

    public:

      virtual double activate(double) = 0; //the actual function
      virtual double derive(double) = 0; //the derivative of the function
      virtual ~Activation() {};

  };

  class Sigmoid: public Activation {

    public:
      double activate(double in) {
        return 1/(1+pow(2.718,0-in));
      }
      double derive(double in) {
        return activate(in)*(1-activate(in)); //derivative of sigmoid
      }

  };

  class Tanh: public Activation {

    public:
      double activate(double in) {
        return ( (pow(2.718,in)-pow(2.718,-in))/2 )/( (pow(2.718,in)+pow(2.718,-in))/2 );
      }
      double derive(double in) {
        return 1 - pow( ((pow(2.718,in)-pow(2.718,-in))/2)/((pow(2.718,in)+pow(2.718,-in))/2),2);
      }

  };

  class Affine: public Activation {

    public:
      double activate(double in) {
        return in;
      }
      double derive(double in) {
        return 1;
      }

  };

  class Relu: public Activation {

    public:
      double activate(double in) {
        if (in > 0) return in;
        return 0;
      }
      double derive(double in) {
        if (in > 0) return 1;
        return 0;
      }

  };

}

#endif
