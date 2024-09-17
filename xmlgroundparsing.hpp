#ifndef PARSE_GROUND_TRUTH_HPP
#define PARSE_GROUND_TRUTH_HPP

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "pugixml.hpp"


// Function to convert a RotatedRect to a bounding box (cv::Rect)
cv::Rect rotatedRectToBoundingBox(const cv::RotatedRect& rRect);

// Function to parse the XML and extract ground truth rectangles
std::vector<cv::Rect> parseXMLGroundTruth(const std::string& xmlPath);

//Function to parse the XML and extract the occupied spaces by the label set to 1
std::vector<cv::Rect> parseOccupiedSpaces(const std::string& xmlPath);



#endif // PARSE_GROUND_TRUTH_HPP
