#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <cmath>
#include "utilities.hpp"
#include "xmlgroundparsing.hpp"
#include "mioumap.hpp"

namespace fs = std::filesystem;
using namespace std;
using namespace cv;


Mat grayImage, bluredImage, detected_edges, final;
int lowThreshold = 100, upperThreshold;
std::vector<cv::Point> polygon_corners;
Mat extImage;
RNG rnc;


//----------------------------------------------------------------------------------------------------------------------------------------------------
// Helper function to calculate the distance between two points
double pointDistance(Point p1, Point p2) {
    return sqrt((p2.x - p1.x) * (p2.x - p1.x) + (p2.y - p1.y) * (p2.y - p1.y));
}

// Function to calculate the length of a line
double lineLength(Vec4i line) {
    Point p1(line[0], line[1]);
    Point p2(line[2], line[3]);
    return pointDistance(p1, p2);
}

// Function to find the distance between two lines (shortest distance between endpoints)
double lineDistance(Vec4i line1, Vec4i line2) {
    Point p1(line1[0], line1[1]); //starting point line 1
    Point p2(line1[2], line1[3]); //ending point line 1
    Point p3(line2[0], line2[1]); //starting point line 2
    Point p4(line2[2], line2[3]); //ending point line 2

    // Calculate all possible endpoint-to-endpoint distances
    double d1 = pointDistance(p1, p3);
    double d2 = pointDistance(p1, p4);
    double d3 = pointDistance(p2, p3);
    double d4 = pointDistance(p2, p4);

    // Return the minimum distance
    return min({d1, d3});
    // return min({d2, d4});
}

// Function to filter lines based on distance and keep the shorter one
void filterLines(vector<Vec4i>& lines, double distanceThreshold) {
    vector<Vec4i> filteredLines;
    vector<bool> keep(lines.size(), true); // Initially keep all lines

    for (size_t i = 0; i < lines.size(); i++) {
        if (!keep[i]) continue; // Skip if the line is already discarded

        for (size_t j = i + 1; j < lines.size(); j++) {
            if (!keep[j]) continue; // Skip if the line is already discarded

            // Calculate the distance between lines
            double distance = lineDistance(lines[i], lines[j]);

            if (distance < distanceThreshold) {
                // Compare the lengths and keep the shorter line
                double length1 = lineLength(lines[i]);
                double length2 = lineLength(lines[j]);

                if (length1 < length2) {
                    keep[j] = false; // Discard the longer line
                } else {
                    keep[i] = false; // Discard the longer line
                    break;
                }
            }
        }
    }

    // Collect the lines that are kept
    for (size_t i = 0; i < lines.size(); i++) {
        if (keep[i]) {
            filteredLines.push_back(lines[i]);
        }
    }

    lines = filteredLines; // Replace the original lines with the filtered lines
}


float findLineAngle(const Vec4i& line) {
    int dx = line[2] - line[0];
    int dy = line[3] - line[1];
    return (atan2(dy, dx) * 180.0 / CV_PI +180);

}

float findAngle(const Vec4i& line) {
    int dx = line[2] - line[0];
    int dy = line[3] - line[1];
    return (atan2(dy, dx) * 180.0 / CV_PI );

}

//------------------------------------------------------------checking close and collinearity-------------------------------------------------
bool checkCloseAndCollinear(const Vec4i& l1, const Vec4i& l2, float angleThreshold, double distanceThreshold) {
    float angle1 = findLineAngle(l1);
    float angle2 = findLineAngle(l2);
    
    if ((angle1 - angle2) > angleThreshold) {
        return false;  // Angles are too different
    }

    Point p1_start(l1[0], l1[1]); //x1,y1
    Point p1_end(l1[2], l1[3]); //x2,y2
    Point p2_start(l2[0], l2[1]); //x3,y3
    Point p2_end(l2[2], l2[3]); //x4,y4

    return (pointDistance(p1_end, p2_start) < distanceThreshold ||
            pointDistance(p1_end, p2_end) < distanceThreshold ||
            pointDistance(p1_start, p2_start) < distanceThreshold ||
            pointDistance(p1_start, p2_end) < distanceThreshold);
}

// Merge two lines into a single line by connecting the farthest points
Vec4i mergeLines(const Vec4i& l1, const Vec4i& l2) {
    Point p1_start(l1[0], l1[1]);
    Point p1_end(l1[2], l1[3]);
    Point p2_start(l2[0], l2[1]);
    Point p2_end(l2[2], l2[3]);

    Point start = p1_start;
    Point end = p1_end;
    
    // Find the farthest pair of points
    vector<Point> points = {p1_start, p1_end, p2_start, p2_end};
    double maxDist = 0.0;
    for (size_t i = 0; i < points.size(); ++i) {
        for (size_t j = i + 1; j < points.size(); ++j) {
            double dist = pointDistance(points[i], points[j]);
            if (dist > maxDist) {
                maxDist = dist;
                start = points[i];
                end = points[j];
            }
        }
    }

    return Vec4i(start.x, start.y, end.x, end.y);
}

// Recursive function to merge lines that are close and collinear
void recursiveMerge(Vec4i& currentLine, vector<Vec4i>& lines, vector<bool>& merged, float angleThreshold, double distanceThreshold) {
    for (size_t i = 0; i < lines.size(); i++) {
        if (!merged[i]) {
            Vec4i nextLine = lines[i];

            if (checkCloseAndCollinear(currentLine, nextLine, angleThreshold, distanceThreshold)) {
                double llength = lineLength(currentLine);
                // if(llength<20)
                // {
                //     continue;
                // }
                // Merge the lines and mark the current line as merged
                currentLine = mergeLines(currentLine, nextLine);
                merged[i] = true;  // Mark the next line as merged

                // Recursively merge with more lines
                recursiveMerge(currentLine, lines, merged, angleThreshold, distanceThreshold);
            }
        }
    }
}

Point findmidpoint(Vec4i line)
{
    Point midpoint((line[0] + line[2]) / 2, (line[1] + line[3]) / 2);
    return midpoint;
}



//---------------------------------------------positive slopes checker--------------------------------------------------------------------------------
vector<Vec4i> checkPositiveSlope(const vector<Vec4i>& lines) {
    vector<Vec4i> positiveLines;

    for (const auto& line : lines) {
        // Extract points of the line
        Point pt1(line[0], line[1]);
        Point pt2(line[2], line[3]);

        // Calculate the slope as y2-y1 / x2-x1
        float slope = (pt2.y - pt1.y) / (float)(pt2.x - pt1.x);

        // Checking if slope is negative
        if (slope < 0) {
            // swap the points to ensure positive slope
            swap(pt1, pt2);
            // cout<<"pt1"<<pt1<<"and pt2"<<pt2<<endl;
        }

        // Store the line with positive slope
        positiveLines.push_back(Vec4i(pt1.x, pt1.y, pt2.x, pt2.y));
    }

    return positiveLines;
}

//-----------------------------------------------------------function to join the midpoints of the lines-----------------------------------




//--------------------------------------------------------------------------------------------------------------------------------------------------------

bool compareLinesByStartPointY(const Vec4i &a, const Vec4i &b) {
    // Compare by the y-coordinate of the start point
    return a[1] < b[1];
}

bool compareLinesByEndPointY(const Vec4i &a, const Vec4i &b) {
    // Compare by the y-coordinate of the end point
    return a[3] < b[3];
}

bool compareLinesByStartPointX(const Vec4i &a, const Vec4i &b) {
    // Compare by the x-coordinate of the start point
    return a[0] < b[0];
}

bool compareLinesByEndPointX(const Vec4i &a, const Vec4i &b) {
    // Compare by the x-coordinate of the end point
    return a[2] < b[2];
}


bool comparePointsByY(const Point& a, const Point& b) {
    return a.y < b.y;
}





// Function to calculate the slope between two points
double calculateSlope(const Point& p1, const Point& p2) {
    if (p2.x == p1.x) { // Avoid division by zero
        return numeric_limits<double>::infinity(); // Infinite slope (vertical line)
    }
    return static_cast<double>(p2.y - p1.y) / (p2.x - p1.x);
}

// Function to check if three points are nearly collinear
bool arePointsNearlyCollinear(const Point& A, const Point& B, const Point& C, double epsilon = 0.12) {

    // cout<<"the points are A:"<<A.x<<","<<A.y<<" B:"<<B.x<<","<<B.y<<" and C:"<<C.x<<","<<C.y<<endl; 
    double slopeAB = calculateSlope(A, B);
    // cout<<" slope ="<<slopeAB<<endl;
    double slopeBC = calculateSlope(B, C);
    // cout<<" slope ="<<slopeBC<<endl;

    double diff = fabs(slopeAB - slopeBC);

    bool slopezero = slopeAB==0;
    if(slopezero)
        return !slopezero;

    bool oppositeSigns = (slopeAB<0 && slopeBC >0 || slopeAB>0 && slopeBC<0); // if one is negative then this becomes true
    
    
    return !oppositeSigns;

}


vector<Point> removeMiddlePoints(vector<Point>& points, double epsilon = 1.0) {
    vector<Point> filteredPoints;

    // Iterate through consecutive triplets of points
    for (size_t i = 0; i < points.size(); ++i) {
        if (i == 0 || i == points.size() - 1) {
            // Always keep the first and last points
            filteredPoints.push_back(points[i]);
        } else if (i < points.size() - 1) {
            // Check if points[i-1], points[i], and points[i+1] are nearly collinear
            if (!arePointsNearlyCollinear(points[i-1], points[i], points[i+1], epsilon)) {
                filteredPoints.push_back(points[i]);
            }
        }
    }

    return filteredPoints;
}



// Function to calculate the distance between two points
double distanceBetweenPoints(const Point& p1, const Point& p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

// Function to find the line with the closest midpoint to a given point
Vec4i findClosestLine(const Point& point, const vector<Vec4i>& lines) {
    Vec4i closestLine;
    double minDistance = numeric_limits<double>::max(); // Initialize with a large value

    for (const auto& line : lines) {
        Point midpoint = findmidpoint(line);
        double distance = distanceBetweenPoints(point, midpoint);

        if (distance < minDistance) {
            minDistance = distance;
            closestLine = line;
        }
    }

    return closestLine;
}

// Function to assign each point to the line whose midpoint is closest to it
vector<pair<Point, Vec4i>> assignPointsToLines(const vector<Point>& points, const vector<Vec4i>& lines) {
    vector<pair<Point, Vec4i>> assignments;

    for (const auto& point : points) {
        Vec4i closestLine = findClosestLine(point, lines);
        assignments.push_back(make_pair(point, closestLine));
    }

    return assignments;
}
//-----------------------------------------------------------------------------------------------------------------------------------------------------




std::vector<cv::Rect> bestBoundingBoxes;



std::vector<cv::Rect> bounding_box_extract(const std::string& folderName, int sequenceNumber) {
    // Construct the sequence folder name based on the sequence number
    std::string sequenceFolder = "sequence" + std::to_string(sequenceNumber);

    // Construct the directory and groundtruthDirectory strings
    std::string directory = folderName + "/" + sequenceFolder + "/frames";
    std::string groundtruthDirectory = folderName + "/" + sequenceFolder + "/bounding_boxes";
   

    float totalAP = 0.0f;
    float maxAP = -1.0f;
    int i = 1;
    int imagecount =0;
    if(sequenceNumber == 0)//base case then you can find the base rectangles
    {    
    for (const auto& entry : fs::directory_iterator(directory)) {
        vector <cv::RotatedRect> totalRectangles; //succeeded in containing all the rectangles in one file
        
        string filepath = entry.path().string();
        string filename = entry.path().filename().string(); // getting the filename

        // matching witht the xml files 
        string baseName = filename.substr(0, filename.find_last_of('.'));

        string groundtruth = groundtruthDirectory + "/" + baseName + ".xml"; // the xml file with the ground truth

        std::vector<cv::Rect> groundTruthRects = parseXMLGroundTruth(groundtruth); //found the ground truth here

        cout<<"ground truth rectangles in "+ to_string(i) <<"is"<<groundTruthRects.size(); 

        // finding the occupied spaces from the ground truth

        std::vector<cv::Rect> occupiedSpacesGround = parseOccupiedSpaces(groundtruth);

        //the coccupied spaces are
        cout<<"ground occupied spaces are "+ to_string(i) <<"is"<<occupiedSpacesGround.size(); 

        //load the file as image
        Mat image = cv::imread(filepath);


        if (image.empty()) {
            cout << "Could not open the images: " << filepath << std::endl;
            continue;
        }

        imagecount++;

        Mat checker;
        image.copyTo(checker);
        Mat blackimg; 
        image.copyTo(blackimg);
        Mat masked;
        Mat masked2l,masked2r;

        masked = masking(image,i); //masked lines for the first set
        image.copyTo(extImage);
        // imshow("colorxyz"+ to_string(i), imgxyz);


        Mat randomcolored;
        image.copyTo(randomcolored);

        //------------------------------------------------------------trying with XYZ---------------------------------------
        Mat imgxyz;
        cvtColor(image, imgxyz, COLOR_BGR2XYZ);
        
        
        // ------------------------------------------------------applying masking and gamma to xyz to the image------------------------------
        masked = masking(imgxyz,i);
        masked2l = masking2l(imgxyz,i);
        masked2r = masking2r(imgxyz,i);
        


        

        //----------------------------------------------------------------gamma correction--------------------------------------
        float gamma = 3.0; //3.2
        Mat gammaresult, gammaresult2l,gammaresult2r;
        gammaCorrection(masked, gammaresult, gamma);
        gammaCorrection(masked2l, gammaresult2l, gamma);
        gammaCorrection(masked2r, gammaresult2r, gamma);

       


        
        //----------------------------------------------------------------trying MSER---------------------------------------------
        // applyMSER(gammaresult,i);

        //----------------------------------------------------------------converting to gray-------------------------------------
        //make into gray
        Mat gray,gray2l,gray2r;
        cvtColor(gammaresult, gray, COLOR_BGR2GRAY);
        cvtColor(gammaresult2l, gray2l, COLOR_BGR2GRAY);
        cvtColor(gammaresult2r, gray2r, COLOR_BGR2GRAY);
       
        //-------------------------------------------------------applying morphological operation-----------------------------------
        morphologyEx( gray, final,MORPH_GRADIENT, Mat());
        Mat final2l,final2r;
        morphologyEx( gray2l, final2l,MORPH_CROSS, Mat());
        morphologyEx( gray2r, final2r,MORPH_CROSS, Mat());

        
        //-----------------------------------------------------------------applying thresholding------------------------------------
        Mat thresh, thresh2l, thresh2r;
        threshold(final, thresh, 180, 200, THRESH_BINARY + THRESH_OTSU); 
        threshold(final2l, thresh2l, 130, 200, THRESH_BINARY + THRESH_OTSU); 
        // imshow("Thresh lines 2l"+ to_string(i), thresh2l);
        threshold(final2r, thresh2r, 120, 150, THRESH_BINARY +THRESH_OTSU ); 
       
        //-----------------------------------------------------------------Canny edge detector-----------------------------------------
        // Apply Canny edge detector
        Mat edges,edges2l, edges2r;
        Canny(thresh, edges, 80, 150, 5, true);
        Canny(thresh2l, edges2l, 80, 240, 3, false);
        Canny(thresh2r, edges2r, 50, 150, 7, true);
      

        //-----------------------------------------------------------------Finding Contours--------------------------------------------
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(edges, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        //---------------------------------------------------------------drawing the contours-----------------------------------------
        Mat contourImg;
        image.copyTo(contourImg);
        for (size_t i = 0; i < contours.size(); i++) {
        Scalar color = Scalar(0, 255, 0); // Green color for contours

    
    }

        //-----------------------------------------------------------------Hough transform---------------------------------------------
        vector<Vec4i> lines , lines2l, lines2r;
        HoughLinesP(edges, lines, 1, CV_PI / 180, 20, 10, 7);//HoughLinesP(edges, lines, 1, CV_PI / 180, 20, 10, 7);

        //////////////////////////////////////////////////////////////second set of lines extraction/////////////////////////////////////////////////
        HoughLinesP(edges2l, lines2l, 1,  CV_PI / 180, 8, 5, 35); //change the threshold for better results
        HoughLinesP(edges2r, lines2r, 1, CV_PI / 180, 20, 5, 35);


        // double distanceThreshold = 10.0; // adjust this threshold= best =2
        // filterLines(lines2, distanceThreshold);

        //making the slopes positive
        lines2l = checkPositiveSlope(lines2l);
        lines2r = checkPositiveSlope(lines2r);

        // cout<<"lines 2l"<<lines2l.size()<<endl; //lines 2l has the values

        //////////////////////////////////////////////////////// drawing the lines 2l////////////////////////////////////////////////////////////////
        // fix here

        vector<Vec4i> lines2lfiltered;  // to store the detected lines filtered by the angle
        for (size_t i = 0; i < lines2l.size(); i++) {
        cv::Vec4i l = lines2l[i]; //left lines

        // Draw the left lines of set 2 on the output image

        //------------------------------------finding angle--------------------------------------------
        float angle;
        int dy=l[3] - l[1];
        int dx=l[2] - l[0];
        angle = atan2(dy,dx)* 180.0 / CV_PI;
        
         if( (angle>=88 && angle <= 135))
        {
        lines2lfiltered.push_back(l);
        Point midl = findmidpoint(l);
        // Point midr = findmidpoint(l2);
        // putText(contourImg, format("The angle is %.2f", angle), midl, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 200, 255), 1, LINE_AA);
        // cv::line(contourImg, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 255, 0), 2); //second set of lines left
        }
    }






        vector<Vec4i> lines2rfiltered; // to store the detected lines filtered by the angle
            //drawing second set of lines on right
           for (size_t i = 0; i < lines2r.size(); i++) {
        // cv::Vec4i l = lines2l[i]; //left lines
        cv::Vec4i l2 = lines2r[i]; //right lines

        // Draw the left lines of set 2 on the output image

        //------------------------------------finding angle--------------------------------------------
        float angle, angle2;


        int dx=l2[3] - l2[1];
        int dy=l2[2] - l2[0];
        angle2 = atan2(dy,dx)* 180.0 / CV_PI;

     
        
        if( (angle2>=75 && angle2 <=95)) //perfect
        {

        lines2rfiltered.push_back(l2);
        
        Point midr = findmidpoint(l2);
        putText(contourImg, format("The angle is %.2f", angle2), midr, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 200, 255), 1, LINE_AA);
        cv::line(contourImg, cv::Point(l2[0], l2[1]), cv::Point(l2[2], l2[3]), cv::Scalar(0, 255, 0), 2); //second set of lines right
        }

        

    }



        //---------------------------------------------------------------Choosing lines which are smaller-------------------------------------
        double distanceThreshold = 2.0; // adjust this threshold= best =2 // first set of lines
        filterLines(lines, distanceThreshold);


        filterLines(lines2lfiltered, distanceThreshold);
        filterLines(lines2rfiltered, distanceThreshold);



//{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        //recursively merging the lines on the far left
        
        //------------------------------------------------------to check merging--------------------------------
        vector<Vec4i> mergedLines;  // for storing the merged lines
        vector<bool> merged(lines.size(), false);
        //---------------------------------------------------------------------------------------------------------                    
        // Draw the detected lines on the original image
        for (size_t i = 0; i < lines.size(); i++) 
        {
        Vec4i l = lines[i];
        //------------------------------------finding angle--------------------------------------------
        float angle;
        int dx=l[3] - l[1];
        int dy=l[2] - l[0];
        angle = atan2(dy,dx)* 180.0 / CV_PI;
        // cout<<"the angle"<<i<< "is"<<angle<<endl;
        //------------------------------------finding the length-----------------------------------------------
        double length = sqrt(pow(dy, 2) + pow(dx, 2));
        // cout<<"the length of"<<i<< "is"<<length<<endl;
        //----------    --------------------------plotting the midpoint and writing the point-------------------------------------------------
        Point midpoint((l[0] + l[2]) / 2, (l[1] + l[3]) / 2);
        double distanceThreshold = 15.0; //15
        double angleThreshold = 10.0;
        //=============================================Finding lines that are close to each other====================================================
        Vec4i l1 = lines[i];
        Point midpoint1((l1[0] + l1[2]) / 2, (l1[1] + l1[3]) / 2); //midpoint of line1
        // double llength = lineLength(l1);
        //applying recursive merging

        if (!merged[i]) {
            Vec4i currentLine = lines[i];  // Start with an unmerged line

            // Recursively merge lines that are close and collinear
            recursiveMerge(currentLine, lines, merged, angleThreshold, distanceThreshold);

            float angle;
            int dy=currentLine[3] - currentLine[1];
            int dx=currentLine[2] - currentLine[0];
            angle = atan2(dy,dx)* 180.0 / CV_PI;

            // After merging, add the combined line to the mergedLines vector
            mergedLines.push_back(currentLine);
        }
        }

               //recursively merging the lines on the second column left
        
        //------------------------------------------------------to check merging--------------------------------
        vector<Vec4i> mergedLines2l;  // for storing the merged lines
        vector<bool> merged2l(lines2lfiltered.size(), false);
        //---------------------------------------------------------------------------------------------------------                    
        // Draw the detected lines on the original image
        for (size_t i = 0; i < lines2lfiltered.size(); i++) 
        {
        Vec4i l = lines2lfiltered[i];
        //------------------------------------finding angle--------------------------------------------
        float angle;
        int dx=l[3] - l[1];
        int dy=l[2] - l[0];
        angle = atan2(dy,dx)* 180.0 / CV_PI;
        // cout<<"the angle"<<i<< "is"<<angle<<endl;
        //------------------------------------finding the length-----------------------------------------------
        double length = sqrt(pow(dy, 2) + pow(dx, 2));
        // cout<<"the length of"<<i<< "is"<<length<<endl;
        //----------    --------------------------plotting the midpoint and writing the point-------------------------------------------------
        Point midpoint((l[0] + l[2]) / 2, (l[1] + l[3]) / 2);
        double distanceThreshold = 15.0; //15
        double angleThreshold = 20.0;
        //=============================================Finding lines that are close to each other====================================================
        Vec4i l1 = lines2lfiltered[i];
        Point midpoint1((l1[0] + l1[2]) / 2, (l1[1] + l1[3]) / 2); //midpoint of line1
        // double llength = lineLength(l1);
        //applying recursive merging

        if (!merged2l[i]) {
            Vec4i currentLine = lines2lfiltered[i];  // Start with an unmerged line

            // Recursively merge lines that are close and collinear
            recursiveMerge(currentLine, lines2lfiltered, merged2l, angleThreshold, distanceThreshold);

            float angle;
            int dy=currentLine[3] - currentLine[1];
            int dx=currentLine[2] - currentLine[0];
            angle = atan2(dy,dx)* 180.0 / CV_PI;

            // After merging, add the combined line to the mergedLines vector
            mergedLines2l.push_back(currentLine);
        }
        }

                       //recursively merging the lines on the second column right
        
        //------------------------------------------------------to check merging--------------------------------
        vector<Vec4i> mergedLines2r;  // for storing the merged lines
        vector<bool> merged2r(lines2rfiltered.size(), false);
        //---------------------------------------------------------------------------------------------------------                    
        // Draw the detected lines on the original image
        for (size_t i = 0; i < lines2rfiltered.size(); i++) 
        {
        Vec4i l = lines2rfiltered[i];
        //------------------------------------finding angle--------------------------------------------
        float angle;
        int dy=l[3] - l[1];
        int dx=l[2] - l[0];
        angle = atan2(dy,dx)* 180.0 / CV_PI;
        // cout<<"the angle"<<i<< "is"<<angle<<endl;
        //------------------------------------finding the length-----------------------------------------------
        double length = sqrt(pow(dy, 2) + pow(dx, 2));
        // cout<<"the length of"<<i<< "is"<<length<<endl;
        //----------    --------------------------plotting the midpoint and writing the point-------------------------------------------------
        Point midpoint((l[0] + l[2]) / 2, (l[1] + l[3]) / 2);
        double distanceThreshold = 15.0; //15
        double angleThreshold = 5.0;
        //=============================================Finding lines that are close to each other====================================================
        Vec4i l1 = lines2rfiltered[i];
        Point midpoint1((l1[0] + l1[2]) / 2, (l1[1] + l1[3]) / 2); //midpoint of line1
        // double llength = lineLength(l1);
        //applying recursive merging

        if (!merged2r[i]) {
            Vec4i currentLine = lines2rfiltered[i];  // Start with an unmerged line

            // Recursively merge lines that are close and collinear
            recursiveMerge(currentLine, lines2rfiltered, merged2r, angleThreshold, distanceThreshold);

            float angle;
            int dy=currentLine[3] - currentLine[1];
            int dx=currentLine[2] - currentLine[0];
            angle = atan2(dy,dx)* 180.0 / CV_PI;

            // After merging, add the combined line to the mergedLines vector
            mergedLines2r.push_back(currentLine);
        }
        }

// we have all the merged lines in mergerLines, mergedLines2l, mergedLines2r


for (size_t i = 0; i < mergedLines2l.size(); i++) { //mergedLines has all the lines
        Vec4i l = mergedLines2l[i];
        float angle;
        int dy=l[3] - l[1];
        int dx=l[2] - l[0];
        angle = atan2(dy,dx)* 180.0 / CV_PI;

        if (angle < 0) 
        {
        angle += 180.0; //making +ve and in a range
        }  

        Point midpoint = findmidpoint(l);
        double llength = lineLength(l);

        
        line(checker, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 255, 0), 2, LINE_AA);
        putText(checker, format("The angle is %.2f", angle), midpoint, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 200, 255), 1, LINE_AA);
        

}




//{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}FURTHER FILTERING }}}}}}}}}}}}}}[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
        
        

        vector<Vec4i> filteredLines;
        //-------------------------------------------------------printing the recursively merged lines
        for (size_t i = 0; i < mergedLines.size(); i++) { //mergedLines has all the lines
        Vec4i l = mergedLines[i];
        float angle;
        int dy=l[3] - l[1];
        int dx=l[2] - l[0];
        angle = atan2(dy,dx)* 180.0 / CV_PI;

        if (angle < 0) 
        {
        angle += 180.0; //making +ve and in a range
        }  

        Point midpoint = findmidpoint(l);
        double llength = lineLength(l);

        if (angle >= 3 && angle <= 20)
        {
        if(llength>=20)
        {
        line(randomcolored, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 255, 0), 2, LINE_AA);
        putText(randomcolored, format("The angle is %.2f", angle), midpoint, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 200, 255), 1, LINE_AA);
        filteredLines.push_back(l);
        }
        }

        

//left, 2l,2r have all been merged

//{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{alternate: userotatedrect}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        //filteredLines has the merged lines.
       


//==============================================================draw parallel lines into rectangles======================================================


        }

        //-------------------------------------------------------checking for negattive slopes and making them positive in the filtered lines---------

        // cout<<"size of filtered lines"<< filteredLines.size()<<endl;
        vector<Vec4i> positiveLines = checkPositiveSlope(filteredLines); //works
        vector<Vec4i> positiveLines2l = checkPositiveSlope(mergedLines2l); 
        vector<Vec4i> positiveLines2r = checkPositiveSlope(mergedLines2r);
            //-----------------------------------------------------------using the sort function-------------------------------------------------------------

        sort(positiveLines.begin(), positiveLines.end(), compareLinesByEndPointY);
        sort(positiveLines2l.begin(), positiveLines2l.end(), compareLinesByEndPointY);
        sort(positiveLines2r.begin(), positiveLines2r.end(), compareLinesByEndPointY);

        //working till heres
      

//-----------------------------------------------------------------------------------------------------------------------------

        vector<Point> midPos;
        for (size_t i = 0; i < positiveLines.size(); i++)
        {
             midPos.push_back(findmidpoint(positiveLines[i]));
        }


        vector<Point> midPos2l; // midppints of the line on the 2nd row left
        for (size_t i = 0; i < positiveLines2l.size(); i++)
        {
             midPos2l.push_back(findmidpoint(positiveLines2l[i]));
        }

        vector<Point> midPos2r; // midppints of the line on the 2nd row right
        for (size_t i = 0; i < positiveLines2r.size(); i++)
        {
             midPos2r.push_back(findmidpoint(positiveLines2r[i]));
        }




        vector<Point> filteredPoints = removeMiddlePoints(midPos);
      
            vector<Point> filteredPoints2l = midPos2l; // not removing middlepoints because the arrangement is different
    //    vector<Point> filteredPoints2r = removeMiddlePoints(midPos2r);
            vector<Point> filteredPoints2r = midPos2r;

//-------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------sorting filtered collinear midpoints and printing the result--------------------------------------------
sort(filteredPoints.begin(), filteredPoints.end(), comparePointsByY);
sort(filteredPoints2l.begin(), filteredPoints2l.end(), comparePointsByY);
sort(filteredPoints2r.begin(), filteredPoints2r.end(), comparePointsByY);

for (size_t i = 0; i < filteredPoints.size()-1; i++)
        {
            Point p = filteredPoints[i];
            Point pn = filteredPoints[i+1];



            Point midomid((p.x+pn.x)/2, (p.y+pn.y)/2); // midpoint of the filtered midpoints
        
        //finding the slope
        // Calculate the slope as y2-y1 / x2-x1
        float slope = (pn.y - p.y) / (float)(pn.x - p.x);
        line(extImage,p, pn, Scalar(0, 255, 0), 2, LINE_AA);
        putText(extImage, format("The angle is %.2f", slope), midomid, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 200, 255), 1, LINE_AA);
        circle(extImage, p, 5, Scalar(180, 255, 0), -1);


        }

        // not printing these for the 2l and 2r for now

        //--------------------------------------------------joining the midpoints in order == straight lines-------------------------------------------------------
            //you get  the center and the height of the lines from here
            vector<Point> rectmid;
            vector<float> midlength;
                for (size_t i = 0; i < filteredPoints.size()-2; i++)
                    {
                        Point p = filteredPoints[i];
                        Point pn = filteredPoints[i+2];


                        midlength.push_back(pointDistance(p,pn));
                        Point midomid((p.x+pn.x)/2, (p.y+pn.y)/2); // midpoint of the filtered midpoints
                        rectmid.push_back(midomid);

                    //finding the slope
                    // Calculate the slope as y2-y1 / x2-x1
                    float slope = (pn.y - p.y) / (float)(pn.x - p.x);
                   
                    }

                 vector<Point> rectmid2l;
            vector<float> midlength2l;
                for (size_t i = 0; i < filteredPoints2l.size()-2; i++)
                    {
                        Point p = filteredPoints2l[i];
                        Point pn = filteredPoints2l[i+2];


                        midlength2l.push_back(pointDistance(p,pn));
                        Point midomid((p.x+pn.x)/2, (p.y+pn.y)/2); // midpoint of the filtered midpoints
                        rectmid2l.push_back(midomid);

                    //finding the slope
                    // Calculate the slope as y2-y1 / x2-x1
                    float slope = (pn.y - p.y) / (float)(pn.x - p.x);
                    
                    }

                    vector<Point> rectmid2r;
                    vector<float> midlength2r;
                   for (size_t i = 0; i < filteredPoints2r.size()-2; i++)
                    {
                        Point p = filteredPoints2r[i];
                        Point pn = filteredPoints2r[i+2];


                        midlength2r.push_back(pointDistance(p,pn));
                        Point midomid((p.x+pn.x)/2, (p.y+pn.y)/2); // midpoint of the filtered midpoints
                        rectmid2r.push_back(midomid);

                    //finding the slope
                    // Calculate the slope as y2-y1 / x2-x1
                    float slope = (pn.y - p.y) / (float)(pn.x - p.x);
                  
                    }



//-----------------------------------------------------------------------finding the lines on which the midpoints exists=successsss------------------------------

                    vector<pair<Point, Vec4i>> assignments = assignPointsToLines(filteredPoints, positiveLines);
                    vector<pair<Point, Vec4i>> assignments2l = assignPointsToLines(filteredPoints2l, positiveLines2l); // fix these
                    vector<pair<Point, Vec4i>> assignments2r = assignPointsToLines(filteredPoints2r, positiveLines2r); //fix these
                    cout<<" size of filtered mid lines is"<<assignments2l.size()<<endl;

//=================================================== building the rotated rectangles=============================================================================
//{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                //{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                //{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                vector<RotatedRect> rec;
                for (int i = 0; i < assignments.size() - 2; i++) 
                {
                const Vec4i& line1 = assignments[i].second;
                const Vec4i& line2 = assignments[i + 2].second; 

                double l1 = lineLength(line1); 
                double l2 = lineLength(line2); 

                float bestAngle = findAngle(line1) > findAngle(line2) ? findAngle(line1) : findAngle(line2); 
                float bestLength = l1 > l2 ? l1 * 2 : l2 * 2; // Width of the rectangle (scaled)
                // float bestLength = l1 > l2 ? l1  : l2 ; // Width of the rectangle (scaled)
                
                float bestMid = midlength[i]*0.8;  //height of the rectangle

                

                Point2f center = rectmid[i]; 
                Size2f dim(bestLength, bestMid); //dimensions for the rotated rectangle

                rec.push_back(RotatedRect(center, dim, bestAngle)); 
                }

                // Append detected rectangles to totalRectangles
                totalRectangles.insert(totalRectangles.end(), rec.begin(), rec.end()); //
//-------------------------------------------------------------------------------for second lines left 
                vector<RotatedRect> rec2l;
                for (int i = 0; i < assignments2l.size() - 2; i++) 
                {
                const Vec4i& line1 = assignments2l[i].second;
                const Vec4i& line2 = assignments2l[i + 2].second; 

                double l1 = lineLength(line1); 
                double l2 = lineLength(line2); 

                float bestAngle = findAngle(line1) > findAngle(line2) ? findAngle(line1) : findAngle(line2); 
                float bestLength = l1 > l2 ? l1 : l2; // Width of the rectangle (scaled)
                // float bestLength = l1 > l2 ? l1  : l2 ; // Width of the rectangle (scaled)
                
                float bestMid = midlength2l[i]*0.8;  //height of the rectangle

                

                Point2f center = rectmid2l[i]; 
                Size2f dim(bestLength, bestMid); //dimensions for the rotated rectangle

                rec2l.push_back(RotatedRect(center, dim, bestAngle)); 
                }
                totalRectangles.insert(totalRectangles.end(), rec2l.begin(), rec2l.end());

//-------------------------------------------------------------------------------for second lines right
                vector<RotatedRect> rec2r;
                for (int i = 0; i < assignments2r.size() - 2; i++) 
                {
                const Vec4i& line1 = assignments2r[i].second;
                const Vec4i& line2 = assignments2r[i + 2].second; 

                double l1 = lineLength(line1); 
                double l2 = lineLength(line2); 

                float bestAngle = findAngle(line1) > findAngle(line2) ? findAngle(line1) : findAngle(line2); 
                float bestLength = l1 > l2 ? l1  : l2; // Width of the rectangle (scaled)
                // float bestLength = l1 > l2 ? l1  : l2 ; // Width of the rectangle (scaled)
                
                float bestMid = midlength2r[i]*0.5;  //height of the rectangle

                

                Point2f center = rectmid2r[i]; 
                Size2f dim(bestLength, bestMid); //dimensions for the rotated rectangle

                rec2r.push_back(RotatedRect(center, dim, bestAngle)); 
                }


                totalRectangles.insert(totalRectangles.end(), rec2r.begin(), rec2r.end());
                
            //{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            //{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

            //convert each of the rotated rectangles to the corresponding bounding rectangles

            std::vector<cv::Rect> totalBoundingBoxRectangles;
            for(const auto& rectangle : totalRectangles)
                {

                    cv::Rect boundingBox = rectangle.boundingRect();
                    totalBoundingBoxRectangles.push_back(boundingBox);

                }
            

    

             double AP = calculateAveragePrecision(totalBoundingBoxRectangles, groundTruthRects , 0.5);
    
            std::cout << "Average Precision (AP) at IoU 0.5: " << AP << std::endl;
            totalAP += AP;
            if (AP > maxAP) {
            maxAP = AP;
            bestBoundingBoxes = totalBoundingBoxRectangles;
            }
            imshow("Final parking spot"+ to_string(i), checker);

        //----------------------------------------------------------eliminating the lines which are off angle in the ones that are connected
    }

    calculateMeanAP(imagecount,totalAP);




        // imshow("Detected White Lines and Merged", randomcolored);
        int parallelthreshold = 200;


        i++;
        waitKey(0); 
    }

return bestBoundingBoxes;

}