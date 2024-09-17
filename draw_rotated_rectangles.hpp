
// draw_rotated_rectangles.hpp
#ifndef DRAW_ROTATED_RECTANGLES_HPP
#define DRAW_ROTATED_RECTANGLES_HPP

#include <opencv2/opencv.hpp>
#include <vector>

void drawRotatedRectangles(cv::Mat& image, int rectWidth, int rectHeight, int startX, int startY, int gap, int numRects, double angle, const std::vector<bool>& parkingStatus, int& statusIndex);

#endif
