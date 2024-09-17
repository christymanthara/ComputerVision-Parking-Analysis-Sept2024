// Function to convert rotated rect to a bounding box (cv::Rect)
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "pugixml.hpp"


using namespace std;
using namespace cv;


//funciton to convet to the normal rectangle
cv::Rect rotatedRectToBoundingBox(const RotatedRect& rRect) { 
    return rRect.boundingRect();
}

// function to parse the XML and extract ground truth rectangles
std::vector<cv::Rect> parseXMLGroundTruth(const std::string& xmlPath) {
    std::vector<cv::Rect> groundTruthRects;

    // loading the XML file
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(xmlPath.c_str());

    if (!result) {
        cerr << "Failed to load XML file: " << result.description() << endl;
        return groundTruthRects;
    }

    // Get the root node (<parking>)
    pugi::xml_node parking = doc.child("parking");
    if (!parking) {
        cerr << "Invalid XML structure: <parking> not found." << endl;
        return groundTruthRects;
    }

    // Iterate over each <space> element
    for (pugi::xml_node space : parking.children("space")) {
        pugi::xml_node rotatedRect = space.child("rotatedRect");
        if (!rotatedRect) continue;

        // Extract <center>, <size>, and <angle>
        pugi::xml_node center = rotatedRect.child("center");
        pugi::xml_node size = rotatedRect.child("size");
        pugi::xml_node angle = rotatedRect.child("angle");

        if (!center || !size || !angle) continue;

        // Read values from XML
        int centerX = center.attribute("x").as_int();
        int centerY = center.attribute("y").as_int();
        int width = size.attribute("w").as_int();
        int height = size.attribute("h").as_int();
        float angleDeg = angle.attribute("d").as_float();

        // Create the RotatedRect
        Point2f centerPoint(centerX, centerY);
        Size2f rectSize(width, height);
        RotatedRect rotatedRectObj(centerPoint, rectSize, angleDeg);

        // Convert the rotated rectangle to a bounding box
        cv::Rect boundingBox = rotatedRectToBoundingBox(rotatedRectObj);
        groundTruthRects.push_back(boundingBox);
    }

    return groundTruthRects;
}



//------------------------------------------------------------Funnction fot the creating the occupied veector rectanfles
// we need to check this function with the nbounding boxes around the contours to make sure that they intersect and they are the cars, then we find the rectangledd from total rectangels that make this intersection and we change their color
std::vector<cv::Rect> parseOccupiedSpaces(const std::string& xmlPath) {
    std::vector<cv::Rect> occupiedSpaces; // the vector of rectangles that is going to store the occupied rectangles

    // Load the XML file
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(xmlPath.c_str());

    if (!result) {
        cerr << "Failed to load XML file: " << result.description() << endl;
        return occupiedSpaces;
    }

    // Get the root node (<parking>)
    pugi::xml_node parking = doc.child("parking");
    if (!parking) {
        cerr << "Invalid XML structure: <parking> not found." << endl;
        return occupiedSpaces;
    }

    // Iterate over each <space> element
    for (pugi::xml_node space : parking.children("space")) {
        // Check the occupied attribute
        int occupied = space.attribute("occupied").as_int();
        if (occupied != 1) continue; // checking if it is set to one

        pugi::xml_node rotatedRect = space.child("rotatedRect");
        if (!rotatedRect) continue;

        // Extract <center>, <size>, and <angle>
        pugi::xml_node center = rotatedRect.child("center");
        pugi::xml_node size = rotatedRect.child("size");
        pugi::xml_node angle = rotatedRect.child("angle");

        if (!center || !size || !angle) continue;

        // reading values from XML
        int centerX = center.attribute("x").as_int();
        int centerY = center.attribute("y").as_int();
        int width = size.attribute("w").as_int();
        int height = size.attribute("h").as_int();
        float angleDeg = angle.attribute("d").as_float();

        
        Point2f centerPoint(centerX, centerY);
        Size2f rectSize(width, height);
        RotatedRect rotatedRectObj(centerPoint, rectSize, angleDeg);// createing the RotatedRect

        
        cv::Rect boundingBox = rotatedRectToBoundingBox(rotatedRectObj);// Convert the rotated rectangle to a bounding box
        occupiedSpaces.push_back(boundingBox);
    }

    return occupiedSpaces;
} 