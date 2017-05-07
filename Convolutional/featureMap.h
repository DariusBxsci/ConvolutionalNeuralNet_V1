#ifndef ADB_FEATUREMAP_H
#define ADB_FEATUREMAP_H

#include "../includes.h"

namespace adb {
namespace cnn {

class FeatureMap { //this class contains data for each feature map (uses OpenCV to load and display images)

    public:

        Mat img;
        vector<vector<double>> image; //stores pixel values
        vector<vector<double>> delta; //stores deltas for each pixel value
        vector<double> flat;

        void size(int xs, int ys) {
            image.resize(xs);
            delta.resize(xs);
            for (int x = 0; x < xs; x++) {
                image[x].resize(ys);
                delta[x].resize(ys);
            }
        }

        FeatureMap getErrorMap(FeatureMap& fm) {
            FeatureMap em;
            em.image = fm.image;
            for (unsigned int x = 0; x < fm.image.size(); x++) {
                for (unsigned int y = 0; y < fm.image[0].size(); y++) {
                    em.image[x][y] = ((image[x][y]) - (fm.image[x][y])); // out-target
                    //This is the derivative of cross entropy loss, with respect to
                    // each input value of the softmax function.
                }
            }
            return em;
        }

        void clearDelta() {
            for (unsigned int x = 0; x < delta.size(); x++) {
                for (unsigned int y = 0; y < delta[0].size(); y++) {
                    delta[x][y] = 0;
                }
            }
        }

        void squash() {
            for (unsigned int x = 0; x < image.size(); x++) {
                for (unsigned int y = 0; y < image[0].size(); y++) {
                    image[x][y] = (image[x][y])/255;
                }
            }
        }

        void unsquash() {
            for (unsigned int x = 0; x < image.size(); x++) {
                for (unsigned int y = 0; y < image[0].size(); y++) {
                    image[x][y] = (image[x][y])*255;
                }
            }
        }

        void sharpen() {
            for (unsigned int x = 0; x < image.size(); x++) {
                for (unsigned int y = 0; y < image[0].size(); y++) {
                    if (image[x][y] > 50) image[x][y] = 255;
                }
            }
        }

        vector<double> flatten() {
            vector<double> f;
            for (unsigned int x = 0; x < image.size(); x++) {
                for (unsigned int y = 0; y < image[0].size(); y++) {
                    f.push_back(image[x][y]);
                }
            }
            flat = f;
            return flat;
        }

        vector<double> dflatten() {
            vector<double> f;
            for (unsigned int x = 0; x < delta.size(); x++) {
                for (unsigned int y = 0; y < delta[0].size(); y++) {
                    f.push_back(delta[x][y]);
                }
            }
            return flat;
        }


        void stringify(vector<double> in) {
            image.resize(1);
            image[0] = in;
        }

        void destringify(vector<double> in, int xs, int ys) {
            size(xs,ys);
            int i = 0;
            for (int x = 0; x < xs; x++) {
                for (int y = 0; y < ys; y++) {
                    image[x][y] = in[i];
                    i++;
                }
            }
        }

        void delt_destringify(vector<double> in, int xs, int ys) {
            size(xs,ys);
            int i = 0;
            for (int x = 0; x < xs; x++) {
                for (int y = 0; y < ys; y++) {
                    delta[x][y] = in[i];
                    i++;
                }
            }
        }

        vector<double> operator~() {
            return flat;
        }

        void display() {

            Mat m(image.size(),image[0].size(),CV_LOAD_IMAGE_GRAYSCALE);
            img = m;

            for (unsigned int x = 0; x < image.size(); x++) {
                for (unsigned int y = 0; y < image[0].size(); y++) {
                    double d = image[x][y];
                    if (d < 0) d = 0;
                    if (d > 255) d = 255;
                    img.at<uchar>(x,y) = d;
                }
            }

            namedWindow( "Image Window", WINDOW_NORMAL );
            imshow( "Image Window", img );
            waitKey(0);
            destroyWindow("Image Window");
        }

        void displayDelta() {

            Mat m(image.size(),image[0].size(),CV_LOAD_IMAGE_GRAYSCALE);
            img = m;

            for (unsigned int x = 0; x < delta.size(); x++) {
                for (unsigned int y = 0; y < delta[0].size(); y++) {
                    double d = delta[x][y]*100;
                    if (d < 0) d = 0;
                    if (d > 255) d = 255;
                    img.at<uchar>(x,y) = d;
                }
            }

            namedWindow( "Image Window", WINDOW_NORMAL );
            imshow( "Image Window", img );
            waitKey(0);
            destroyWindow("Image Window");
        }

        void loadImage(string filename, int sx, int sy) {

            img = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

            image.resize(sx);
            delta.resize(sx);
            for (int x = 0; x < sx; x++) {
                delta[x].resize(sy);
                image[x].resize(sy);
                for (int y = 0; y < sy; y++) {
                    delta[x][y] = 0;
                    image[x][y] = (double)img.at<uchar>(x,y);
                }
            }

        }

        void loadImage(Mat img) {

            int sx = img.cols;
            int sy = img.rows;

            image.resize(sx);
            delta.resize(sx);
            for (int x = 0; x < sx; x++) {
                delta[x].resize(sy);
                image[x].resize(sy);
                for (int y = 0; y < sy; y++) {
                    delta[x][y] = 0;
                    image[x][y] = (float)img.at<Vec3b>(x,y)[0];
                }
            }

        }

};

}
}

#endif // ADB_FEATUREMAP_H
