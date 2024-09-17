#ifndef BOX_EXTRACT_FUNCTION_HPP
#define BOX_EXTRACT_FUNCTION_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;
using namespace std;
using namespace cv;

// declarations of functions
double pointDistance(Point p1, Point p2);
double lineLength(Vec4i line);
double lineDistance(Vec4i line1, Vec4i line2);
void filterLines(vector<Vec4i>& lines, double distanceThreshold);
float findLineAngle(const Vec4i& line);
float findAngle(const Vec4i& line);
bool checkCloseAndCollinear(const Vec4i& l1, const Vec4i& l2, float angleThreshold, double distanceThreshold);
Vec4i mergeLines(const Vec4i& l1, const Vec4i& l2);
void recursiveMerge(Vec4i& currentLine, vector<Vec4i>& lines, vector<bool>& merged, float angleThreshold, double distanceThreshold);
Point findmidpoint(Vec4i line);
vector<Vec4i> checkPositiveSlope(const vector<Vec4i>& lines);
vector<Point> removeMiddlePoints(vector<Point>& points, double epsilon = 1.0);
double distanceBetweenPoints(const Point& p1, const Point& p2);
Vec4i findClosestLine(const Point& point, const vector<Vec4i>& lines);
vector<pair<Point, Vec4i>> assignPointsToLines(const vector<Point>& points, const vector<Vec4i>& lines);
vector<cv::Rect> bounding_box_extract(const std::string& folderName, int sequenceNumber);

// Global variables (if needed)
// extern std::vector<cv::Rect> bestBoundingBoxes;

#endif // BOX_EXTRACT_FUNCTION_HPP
