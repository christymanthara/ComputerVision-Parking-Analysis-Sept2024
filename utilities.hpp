#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <limits>

using namespace std;
using namespace cv;

// Function declarations
Mat CannyThreshold(Mat img);
Mat masking(Mat image, int i); //for the first set of lines
Mat masking2l(Mat image,int i); //for the second set of lines left
Mat masking2r(Mat image,int i); //for the second set of lines right
Mat findlines(Mat img, int i);
Mat applyMSER(Mat img, int i);
void sobelApplied(Mat img);
void gammaCorrection(const Mat& src, Mat& dst, float gamma);
Mat constructRectangles(Mat image, vector<Vec4i> lines, int distanceParallelThreshold);

double lineLength(Vec4i line);
double lineDistance(Vec4i line1, Vec4i line2);
void filterLines(vector<Vec4i>& lines, double distanceThreshold);



#endif // UTILITIES_HPP
