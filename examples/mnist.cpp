#include "../Convolutional/network.h"
#include <SFML/Graphics.hpp>
using namespace adb;

cnn::FeatureMap drawDigit() { //uses the SFML library to create a simple painting program to test digit recognition

    cnn::FeatureMap digit;
    digit.size(28,28);
    sf::RenderWindow window(sf::VideoMode(560, 560), "Draw a digit!");
    //window.setSize(sf::Vector2u(560,560));
    sf::RectangleShape shape(sf::Vector2f(28, 28));
    shape.setFillColor(sf::Color(255, 255, 255));
    while (window.isOpen())
        {
            sf::Event event;
            while (window.pollEvent(event))
            {
                if (event.type == sf::Event::Closed) {
                    sf::Image display = window.capture();
                    window.close();
                    for (int y = 0; y < 560; y+=20) {
                        for (int x = 0; x < 560; x+=20) {

                            double c = 0;
                            for (int a = 0; a < 20; a++) {
                                for (int b = 0; b < 20; b++) {
                                    c += static_cast< int >(display.getPixel(x+b,y+a).b);
                                }
                            }
                            c =  c / (20*20);

                            digit.image[y/20][x/20] = c;

                        }
                    }
                }
            }

            shape.setPosition(sf::Mouse::getPosition(window).x, sf::Mouse::getPosition(window).y);
            if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
                window.draw(shape);
            }

            //window.clear();
            window.display();
        }

    return digit;
}

int main() {

    cnn::Network convnet; //create the model.
    convnet.addLayer(new cnn::KernelLayer(48, 1, 3, 3)); // 24 5x5 kernels per each feature map
    convnet.addLayer(new cnn::FilterLayer(new cnn::Relu));
    convnet.addLayer(new cnn::FilterLayer(new cnn::Maxpool));
    convnet.addLayer(new cnn::KernelLayer(2, 48, 3, 3)); //2 5x5 kernels per each of the 24 feature maps
    convnet.addLayer(new cnn::FilterLayer(new cnn::Relu));
    convnet.addLayer(new cnn::FilterLayer(new cnn::Maxpool));
    convnet.addLayer(new cnn::ClassificationLayer(7*7*48*2,150,10)); //classifier with 3 layers, ends with 10 neurons

    cout << "loading test img..." << endl; //load in a test image to confirm that the model works properly
    cnn::FeatureMap fm;
    fm.loadImage("/mnt/programming/AxtyaNet/data/mnist/testing/7/307.png", 28, 28);
    cout << "processing test image..." << endl;
    convnet.process(fm);
    convnet.displayOutput();
    cout << "done processing!" << endl;

    /*cout << "LAYER 0" << endl;
    convnet.inspectLayer(0);
    cout << "LAYER 1" << endl;
    convnet.inspectLayer(1);
    cout << "LAYER 2" << endl;
    convnet.inspectLayer(2);
    cout << "LAYER 3" << endl;
    convnet.inspectLayer(3);
    cout << "LAYER 4" << endl;
    convnet.inspectLayer(4);*/

    cout << "Loading in dataset..." << endl;
    cnn::ImageTrainingSet mnist;
    mnist.load("/mnt/programming/AxtyaNet/data/mnist/training/0/", "png", {1,0,0,0,0,0,0,0,0,0}); //load in each png image in the mnist dataset
    mnist.load("/mnt/programming/AxtyaNet/data/mnist/training/1/", "png", {0,1,0,0,0,0,0,0,0,0});
    mnist.load("/mnt/programming/AxtyaNet/data/mnist/training/2/", "png", {0,0,1,0,0,0,0,0,0,0});
    mnist.load("/mnt/programming/AxtyaNet/data/mnist/training/3/", "png", {0,0,0,1,0,0,0,0,0,0});
    mnist.load("/mnt/programming/AxtyaNet/data/mnist/training/4/", "png", {0,0,0,0,1,0,0,0,0,0});
    mnist.load("/mnt/programming/AxtyaNet/data/mnist/training/5/", "png", {0,0,0,0,0,1,0,0,0,0});
    mnist.load("/mnt/programming/AxtyaNet/data/mnist/training/6/", "png", {0,0,0,0,0,0,1,0,0,0});
    mnist.load("/mnt/programming/AxtyaNet/data/mnist/training/7/", "png", {0,0,0,0,0,0,0,1,0,0});
    mnist.load("/mnt/programming/AxtyaNet/data/mnist/training/8/", "png", {0,0,0,0,0,0,0,0,1,0});
    mnist.load("/mnt/programming/AxtyaNet/data/mnist/training/9/", "png", {0,0,0,0,0,0,0,0,0,1});

    cnn::ImageTrainingSet mnist_testing; //load in the mnist training set
    mnist_testing.load("/mnt/programming/AxtyaNet/data/mnist/testing/0/", "png", {1,0,0,0,0,0,0,0,0,0});
    mnist_testing.load("/mnt/programming/AxtyaNet/data/mnist/testing/1/", "png", {0,1,0,0,0,0,0,0,0,0});
    mnist_testing.load("/mnt/programming/AxtyaNet/data/mnist/testing/2/", "png", {0,0,1,0,0,0,0,0,0,0});
    mnist_testing.load("/mnt/programming/AxtyaNet/data/mnist/testing/3/", "png", {0,0,0,1,0,0,0,0,0,0});
    mnist_testing.load("/mnt/programming/AxtyaNet/data/mnist/testing/4/", "png", {0,0,0,0,1,0,0,0,0,0});
    mnist_testing.load("/mnt/programming/AxtyaNet/data/mnist/testing/5/", "png", {0,0,0,0,0,1,0,0,0,0});
    mnist_testing.load("/mnt/programming/AxtyaNet/data/mnist/testing/6/", "png", {0,0,0,0,0,0,1,0,0,0});
    mnist_testing.load("/mnt/programming/AxtyaNet/data/mnist/testing/7/", "png", {0,0,0,0,0,0,0,1,0,0});
    mnist_testing.load("/mnt/programming/AxtyaNet/data/mnist/testing/8/", "png", {0,0,0,0,0,0,0,0,1,0});
    mnist_testing.load("/mnt/programming/AxtyaNet/data/mnist/testing/9/", "png", {0,0,0,0,0,0,0,0,0,1});
    cout << "done." << endl;

    cout << "SIZE OF TRAINING SET IS " << mnist.getSet().size() << endl;

    cout << endl;
    cout << "INITIAL ERROR: " << convnet.getClassError(mnist_testing.getSet()) << endl;
    cout << endl;

    double errc = 1;
    double err = 1;
    int x = 0;

    //convnet.load("/mnt/programming/AxtyaNet/models/mnist1_1/"); //load in a model
    while (errc > 0.04) { //stop training when the model reaches 4% testing error

        convnet.train(mnist.getSet(), 60000, 0.05);
        convnet.save("/mnt/programming/AxtyaNet/models/mnist/");
        errc = convnet.getClassError(mnist_testing.getSet());
        err = convnet.getError(mnist_testing.getSet());
        cout << "Error " << x << ": " << err << endl;
        cout << "Class Error " << x << ": " << errc << endl;

        cout << endl;
        convnet.process(fm);
        convnet.displayOutput();
        cout << endl;

        x++;
    }

    while (true) { //test by classifying handwritten digits
        cout << endl;
        cnn::FeatureMap fm = drawDigit();
        fm.display();
        convnet.process(fm);
        convnet.displayOutput();
        cout << "THIS IS A " << convnet.getClass() << endl;
        cout << endl;
    }

    cout << "DONE!!!" << endl;
    return 0;
}
