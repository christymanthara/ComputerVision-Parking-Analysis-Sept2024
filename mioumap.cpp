#include <iostream>
#include<filesystem>
#include <vector>
#include <opencv2/opencv.hpp>
#include "pugixml.hpp"

namespace fs = std::filesystem;
using namespace::std;

double calculateIoU(const cv::Rect& rectA, const cv::Rect& rectB) {
    cv::Rect intersection = rectA & rectB;
    int intersectionArea = intersection.area();
    
    if (intersectionArea <= 0) return 0.0;

    // double rectAarea = rectA.size.width*rectA.size.height;  // because rotated rectangles donot have a fucntion named area()
    int unionArea = rectA.area() + rectB.area() - intersectionArea;
    // int unionArea = rectAarea + rectB.area() - intersectionArea;
    return static_cast<double>(intersectionArea) / unionArea;
}


// Function to draw rectangles on the image and show it
void showDetectedRectangles(const cv::Mat& image, const std::vector<cv::Rect>& detectedRects) {
    cv::Mat outputImage = image.clone();  // Clone the image to avoid modifying the original

    // Draw each detected rectangle
    for (const auto& rect : detectedRects) {
        cv::rectangle(outputImage, rect, cv::Scalar(0, 255, 0), 2);  // Draw rectangle in green
    }

    // Display the result
    cv::imshow("Detected Rectangles", outputImage);
    cv::waitKey(0);  // Wait for a key press before closing the window
}

void calculatePrecisionRecall(const std::vector<cv::Rect>& detectedRects, const std::vector<cv::Rect>& groundTruthRects, double& precision, double& recall, double iouThreshold = 0.5) 
{
    int truePositives = 0, falsePositives = 0, falseNegatives = 0;

    std::vector<bool> matchedGroundTruth(groundTruthRects.size(), false);

    for (const auto& detectedRect : detectedRects) {
        bool foundMatch = false;
        for (size_t i = 0; i < groundTruthRects.size(); ++i) {
            double iou = calculateIoU(detectedRect, groundTruthRects[i]); // calculating iou for each of the rectangles we made

            if (iou >= iouThreshold && !matchedGroundTruth[i]) {
                truePositives++;
                matchedGroundTruth[i] = true;
                foundMatch = true;
                break;
            }
        }
        if (!foundMatch) {
            falsePositives++;
        }
    }

    falseNegatives = std::count(matchedGroundTruth.begin(), matchedGroundTruth.end(), false);

    precision = truePositives + falsePositives > 0 ? static_cast<double>(truePositives) / (truePositives + falsePositives) : 0;
    recall = truePositives + falseNegatives > 0 ? static_cast<double>(truePositives) / (truePositives + falseNegatives) : 0;
}


double calculateMeanAveragePrecision(const std::vector<std::string>& imagePaths, const std::string& groundTruthFolder, double iouThreshold = 0.5) {
    double totalPrecision = 0.0;
    int imageCount = 0;

    // for (const auto& imagePath : imagePaths) {
    //     std::string imageName = fs::path(imagePath).stem().string();
    //     std::string gtPath = groundTruthFolder + "/" + "gt_" + imageName + ".json";

    //     // cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);  // Load image in color
    //     // if (image.empty()) {
    //     //     std::cout << "Could not open or find the image: " << imagePath << std::endl;
    //     //     continue;
    //     // }

    //     // std::vector<cv::Rect> detectedRects = detectRectangles(image);
    //     std::vector<cv::Rect> groundTruthRects = loadGroundTruthRects(gtPath);

    //     double precision = 0.0, recall = 0.0;
    //     calculatePrecisionRecall(detectedRects, groundTruthRects, precision, recall, iouThreshold);

    //     // Show detected rectangles on the original image
    //     showDetectedRectangles(image, detectedRects);

    //     totalPrecision += precision;
    //     imageCount++;
    // }

    return (imageCount > 0) ? totalPrecision / imageCount : 0.0;
}

std::vector<cv::Rect> loadGroundTruthRects(const std::string& gtFilePath) {
//     std::ifstream inputFile(gtFilePath);
//     if (!inputFile.is_open()) {
//         std::cerr << "Error opening file: " << gtFilePath << std::endl;
//         return {};
//     }

//     json j;
//     try {
//         inputFile >> j;
//     } catch (const json::parse_error& e) {
//         std::cerr << "JSON parse error: " << e.what() << std::endl;
//         return {};
//     }

    std::vector<cv::Rect> groundTruthRects;
//     try {
//         for (const auto& rect : j["rectangles"]) {
//             // Convert JSON elements to integers
//             int x1 = rect[0][0].get<int>();
//             int y1 = rect[0][1].get<int>();
//             int x2 = rect[1][0].get<int>();
//             int y2 = rect[1][1].get<int>();
//             int x3 = rect[2][0].get<int>();
//             int y3 = rect[2][1].get<int>();
//             int x4 = rect[3][0].get<int>();
//             int y4 = rect[3][1].get<int>();

//             // Calculate bounding box coordinates
//             int x = std::min({x1, x2, x3, x4});
//             int y = std::min({y1, y2, y3, y4});
//             int width = std::max({x1, x2, x3, x4}) - x;
//             int height = std::max({y1, y2, y3, y4}) - y;

//             groundTruthRects.emplace_back(x, y, width, height);
//         }
//     } catch (const json::type_error& e) {
//         std::cerr << "JSON type error: " << e.what() << std::endl;
//     }

    return groundTruthRects;
}

double calculateAveragePrecision(const std::vector<cv::Rect>& detectedRects, const std::vector<cv::Rect>& groundTruthRects, double iouThreshold = 0.5) {
    double totalPrecision = 0.0;
    int imageCount = 0;

    // for (const auto& imagePath : imagePaths) {
    //     std::string imageName = fs::path(imagePath).stem().string();
    //     std::string gtPath = groundTruthFolder + "/" + "gt_" + imageName + ".json";

        // cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);  // Load image in color
        // if (image.empty()) {
        //     std::cout << "Could not open or find the image: " << imagePath << std::endl;
        //     continue;
        // }

        // std::vector<cv::Rect> detectedRects = detectRectangles(image);
        // std::vector<cv::Rect> groundTruthRects = loadGroundTruthRects(gtPath);

        double precision = 0.0, recall = 0.0;
        calculatePrecisionRecall(detectedRects, groundTruthRects, precision, recall, iouThreshold);

        // Show detected rectangles on the original image
        // showDetectedRectangles(image, detectedRects);

        totalPrecision += precision;
        // imageCount++; //dont return this for now
// }

    // return (imageCount > 0) ? totalPrecision / imageCount : 0.0;

    return totalPrecision;
}

double calculateMeanAveragePrecision(const std::vector<cv::Rect>& detectedRects, const std::vector<cv::Rect>& groundTruthRects, double iouThreshold = 0.5) {
    double totalPrecision = 0.0;
    int imageCount = 0;

    // for (const auto& imagePath : imagePaths) {
    //     std::string imageName = fs::path(imagePath).stem().string();
    //     std::string gtPath = groundTruthFolder + "/" + "gt_" + imageName + ".json";

        // cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);  // Load image in color
        // if (image.empty()) {
        //     std::cout << "Could not open or find the image: " << imagePath << std::endl;
        //     continue;
        // }

        // std::vector<cv::Rect> detectedRects = detectRectangles(image);
        // std::vector<cv::Rect> groundTruthRects = loadGroundTruthRects(gtPath);

        double precision = 0.0, recall = 0.0;
        calculatePrecisionRecall(detectedRects, groundTruthRects, precision, recall, iouThreshold);

        // Show detected rectangles on the original image
        // showDetectedRectangles(image, detectedRects);

        totalPrecision += precision;
        // imageCount++; //dont return this for now
// }

    // return (imageCount > 0) ? totalPrecision / imageCount : 0.0;

    return totalPrecision;
}


double calculateMeanAP(int imgcount, double totalSumPrecision)
{
        return totalSumPrecision/imgcount;
}