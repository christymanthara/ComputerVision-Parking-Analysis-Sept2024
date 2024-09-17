// mioumap.hpp
#ifndef MIOUMAP_HPP
#define MIOUMAP_HPP

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

// Function to calculate Intersection over Union (IoU) between two rectangles
double calculateIoU(const cv::Rect& rectA, const cv::Rect& rectB);

// Function to display detected rectangles on an image
void showDetectedRectangles(const cv::Mat& image, const std::vector<cv::Rect>& detectedRects);

// Function to calculate precision and recall
void calculatePrecisionRecall(const std::vector<cv::Rect>& detectedRects, const std::vector<cv::Rect>& groundTruthRects, double& precision, double& recall, double iouThreshold = 0.5);


//overloaded functions
// Function to calculate mean average precision for multiple images

double calculateMeanAveragePrecision(const std::vector<std::string>& imagePaths, const std::string& groundTruthFolder, double iouThreshold = 0.5);
// double calculateMeanAveragePrecision(const std::vector<std::string>& imagePaths, const std::string& groundTruthFolder, double iouThreshold = 0.5);

// Overloaded function to calculate mean average precision using detected and ground truth rectangles
double calculateMeanAveragePrecision(const std::vector<cv::Rect>& detectedRects, const std::vector<cv::Rect>& groundTruthRects, double iouThreshold = 0.5);
// we use this one for average predictions
double calculateAveragePrecision(const std::vector<cv::Rect>& detectedRects, const std::vector<cv::Rect>& groundTruthRects, double iouThreshold = 0.5);

double calculateMeanAP(int imgcount, double totalAP);

#endif // MIOUMAP_HPP
