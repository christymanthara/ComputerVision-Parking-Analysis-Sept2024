#include <opencv2/opencv.hpp>
#include <vector>
#include "tinyxml2.h"

using namespace tinyxml2;

int main() {
    // Load the image
    cv::Mat image = cv::imread("/home/ms/parking-space/img1.png");
    if (image.empty()) {
        std::cout << "Could not open or find the image!\n";
        return -1;
    }
    
    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    cv::Mat equalized;
    cv::equalizeHist(gray, equalized);

    // Apply Gaussian Blur to remove noise
    cv::Mat blur;
    cv::GaussianBlur(gray, blur, cv::Size(5, 5), 0);


    // Apply edge detection
    cv::Mat edges;
    cv::Canny(blur, edges, 100, 150);
   
    cv::Mat thresh;
    // Step 4: Threshold to detect significant differences
    cv::threshold(edges, thresh, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU );

    imshow("difference", thresh);
    imshow("edges", edges);


    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    const double minContourArea = 100.0;  // Adjust this value based on your image size
    const double minAspectRatio = 1.0;    // Minimum width/height ratio for a valid car

    std::vector<cv::Rect> validBoundingBoxes;

    // Loop through contours and filter out small or irregular ones
    for (size_t i = 0; i < contours.size(); i++) {
        // Calculate bounding box for each contour
        cv::Rect boundingBox = cv::boundingRect(contours[i]);

        // Calculate contour area
        double contourArea = cv::contourArea(contours[i]);

        // Calculate aspect ratio (width/height)
        double aspectRatio = static_cast<double>(boundingBox.width) / boundingBox.height;

        // Only keep bounding boxes that meet the area and aspect ratio criteria
        if (contourArea > minContourArea && aspectRatio >= minAspectRatio) {
            validBoundingBoxes.push_back(boundingBox);
        }
    }
    // Loop through contours and draw bounding boxes
    for (const auto &boundingBox : validBoundingBoxes) {
        // cv::Rect boundingBox = cv::boundingRect(contours[i]);

        cv::Mat roi = edges(boundingBox);

        // Count the number of white pixels (non-zero pixels) in the ROI
        int nonZeroCount = cv::countNonZero(roi);

        // Calculate the area of the bounding box
        int bboxArea = boundingBox.width * boundingBox.height;

        // Calculate the ratio of white pixels to the total area
        double whitePixelRatio = (double)nonZeroCount / bboxArea;


       bool isOccupied = whitePixelRatio > 0.10?true:false;  // If more than 10% white pixels, the space is considered occupied
         // Print the ratio
       std::cout<<isOccupied; 
       std::cout << "Bounding Box: White Pixel Ratio = " << whitePixelRatio << ", Occupied: " << (isOccupied ? "Yes" : "No") << std::endl;

        // Define the color based on occupancy status
        cv::Scalar color = isOccupied ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
        cv::rectangle(image, boundingBox, color, 2);
    }

     

    // Display the result
    cv::imshow("Segmented Cars", image);
    cv::waitKey(0);

    return 0;
}
