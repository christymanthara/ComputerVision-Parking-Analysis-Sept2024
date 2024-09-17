#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include "utilities.hpp"
#include "xmlgroundparsing.hpp"
#include "mioumap.hpp"
#include "pugixml.hpp"
#include "box_extract_function.hpp"
#include "draw_rotated_rectangles.hpp"

int main() {

    //----------------------------------------------------------------work done by jayanth--------------------------------------------------------------------
    std::string folderName = "ParkingLot_dataset";
    int sequenceNumber = 0;

std::string imagePath = "../ParkingLot_dataset/sequence4/frames/2013-04-15_07_05_01.png";
    
    // Load the input image
    cv::Mat inputImage = cv::imread(imagePath);
    if (inputImage.empty()) {
        std::cout << "Could not open or find the image: " << imagePath << std::endl;
        return -1;
    }

    std::cout << "Image loaded successfully: " << imagePath << std::endl;

    // Define minimap properties
    int minimapWidth = 500;
    int minimapHeight = 300;
    int borderSize = 10;
    
    // Create a blank minimap with a white background
    cv::Mat minimap(minimapHeight, minimapWidth, CV_8UC3, cv::Scalar(255, 255, 255));

    // Draw a black border around the minimap
    cv::rectangle(minimap, cv::Point(0, 0), cv::Point(minimapWidth-1, minimapHeight-1), cv::Scalar(0, 0, 0), borderSize);

    // Define the parking rectangles properties
    int rectWidth = 20;
    int rectHeight = 40;
    int gap = 10;

    // Parking status (True = Green, False = Red)
    std::vector<bool> parkingStatus = {
        true, false, true, true, true, false, false, true, false, true,
        false, true, true, false, true, false, true, false, false, true,
        true, true, true, true, false, true, false, true, true, false,
        true, false, true, false, true, true, true
    };

    int statusIndex = 0;
    // Draw rotated rectangles on the minimap
    drawRotatedRectangles(minimap, rectWidth, rectHeight, 280, 20, gap, 5, -45.0, parkingStatus, statusIndex);
    drawRotatedRectangles(minimap, rectWidth, rectHeight, 190, 100, gap, 8, 45.0, parkingStatus, statusIndex);
    drawRotatedRectangles(minimap, rectWidth, rectHeight, 177, 130, gap, 9, -45.0, parkingStatus, statusIndex);
    drawRotatedRectangles(minimap, rectWidth, rectHeight, 137, 210, gap, 10, 45.0, parkingStatus, statusIndex);
    drawRotatedRectangles(minimap, rectWidth, rectHeight, 137, 240, gap, 10, 45.0, parkingStatus, statusIndex);

    // Resize minimap if it's larger than the input image
    if (minimap.rows > inputImage.rows / 3 || minimap.cols > inputImage.cols / 3) {
        cv::resize(minimap, minimap, cv::Size(inputImage.cols / 3, inputImage.rows / 3));
    }

    // Calculate the position for the minimap to be placed in the bottom-right corner
    int x_offset = inputImage.cols - minimap.cols - 10; // Padding of 10px
    int y_offset = inputImage.rows - minimap.rows - 10;

    // Place the minimap onto the input image
    minimap.copyTo(inputImage(cv::Rect(x_offset, y_offset, minimap.cols, minimap.rows)));

    // Save the final image
    cv::imwrite("final_image_with_minimap.png", inputImage);

    std::cout << "Minimap added to the image and saved as 'final_image_with_minimap.png'." << std::endl;

    //----------------------------------------------------------------------------------------------------------------------------------------

//{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{work done by christy}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
vector<cv::Rect> bestPArkingSpaces = bounding_box_extract(folderName, sequenceNumber);




    

    return 0;
}
