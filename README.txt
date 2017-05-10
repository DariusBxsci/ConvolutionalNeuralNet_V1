This is a simple implementation of a convolution neural network (CNN) in C++.
It uses Opencv (just for loading and displaying images) and the C++ standard library.
I want to use this code as a prototype for a future, highly customizeable, C++ library that will encompass many neural network architectures (similar to tensorflow)

Notes for full library:
	- Use more design patterns to make many aspects of the neural network as customizeable as possible (optimization algorithm, activation function, etc.)
	- Treat convolution layers as feed forward layers with shared weights
	- Create different many types of layers, allowing for integrating different neural network architectures into a single model
		-> eg. Generative adversarial networks
	- Allow for higher-dimensional convolutions
	- Load data in binary format, instead of through many images
	- Integrate OpenCL to allow for training on the GPU rather than just the CPU

Please note that there was a bit of a time constraint on this project, so it is very rough around the edges.
