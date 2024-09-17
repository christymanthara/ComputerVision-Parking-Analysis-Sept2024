
// draw_rotated_rectangles.cpp

#include "draw_rotated_rectangles.hpp"

void drawRotatedRectangles(cv::Mat& image, int rectWidth, int rectHeight, int startX, int startY, int gap, int numRects, double angle, const std::vector<bool>& parkingStatus, int& statusIndex) {
    for(int i = 0; i < numRects; ++i) {
        int rectX1 = startX + i * (rectWidth + gap);

        cv::Point2f rect_center(rectX1 + rectWidth / 2.0, startY + rectHeight / 2.0);
        cv::RotatedRect rotated_rect(rect_center, cv::Size2f(rectWidth, rectHeight), angle);

        cv::Scalar color = parkingStatus[statusIndex] ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        statusIndex++;

        cv::Point2f vertices2f[4];
        rotated_rect.points(vertices2f);

        std::vector<cv::Point> vertices;
        for (int j = 0; j < 4; j++) {
            vertices.push_back(vertices2f[j]);
        }

        cv::fillConvexPoly(image, vertices, color);
        cv::polylines(image, vertices, true, cv::Scalar(0, 0, 0), 2);
    }
}
