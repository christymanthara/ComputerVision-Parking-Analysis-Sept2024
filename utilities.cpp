#include <opencv2/opencv.hpp> // Include OpenCV if you are working with OpenCV

using namespace std;
using namespace cv;


// Mat extImage;
// RNG rnc;
Mat rotrectoutput;


//this is the header file for the fucntion definitions
static Mat CannyThreshold(Mat img)
{
    Mat grayImage, bluredImage, detected_edges, final;
    int lowThreshold = 100, upperThreshold;

//step1: grayscale
cvtColor(img,grayImage, COLOR_BGR2GRAY);

//step2: we apply gaussian blur
GaussianBlur( grayImage, bluredImage, Size( 3,3), 0, 0 );

//step3:Apply Canny edge detection
// Canny( bluredImage, detected_edges, lowThreshold, lowThreshold*3, kernel_size );

Canny( bluredImage, detected_edges, lowThreshold, lowThreshold*3);

return detected_edges;
}


Mat masking(Mat image,int i) //works perfectly
{
    std::vector<cv::Point> polygon_corners;
    Mat masked_image;

    //create a black image of the same size as the original image
        Mat blackimg = Mat::zeros(image.size(),CV_8UC1);
        // imshow("Blackimage" + to_string(i),blackimg);

    //taking the polygon area (hard coding)
        polygon_corners.push_back(cv::Point(243, 46));  
        polygon_corners.push_back(cv::Point(649, 705)); 
        polygon_corners.push_back(cv::Point(911, 596)); 
        polygon_corners.push_back(cv::Point(412, 19));  
        

        fillPoly(blackimg, polygon_corners, Scalar(255, 255, 255)); //fill the mask with white
        // imshow("Blackimage filled" + to_string(i),blackimg);

        //applyinng the mask to the original image
        bitwise_and(image,image,masked_image,blackimg);
        // imshow("And Mask Applied" + to_string(i),masked_image);
        // bitwise_not(image,masked_image,blackimg);
        // imshow("Not Mask Applied" + to_string(i),masked_image); //using the not mask also removes the white lines into black

        return masked_image;
        
}

Mat masking2r(Mat image,int i) //works perfectly for the second set of lines
{
    std::vector<cv::Point> polygon_corners;
    Mat masked_image;

    //create a black image of the same size as the original image
        Mat blackimg = Mat::zeros(image.size(),CV_8UC1);
        // imshow("Blackimage" + to_string(i),blackimg);

    //taking the polygon area (hard coding)
        polygon_corners.push_back(cv::Point(695, 1)); 
        polygon_corners.push_back(cv::Point(777, 4)); 
        polygon_corners.push_back(cv::Point(798, 4));  
        
        polygon_corners.push_back(cv::Point(1260, 311));  
        polygon_corners.push_back(cv::Point(1268, 372)); 
        polygon_corners.push_back(cv::Point(1192, 359));  
        

        fillPoly(blackimg, polygon_corners, Scalar(255, 255, 255)); //fill the mask with white
        // imshow("Blackimage filled" + to_string(i),blackimg);

        //applyinng the mask to the original image
        bitwise_and(image,image,masked_image,blackimg);
        // imshow("And Mask Applied" + to_string(i),masked_image);
        // bitwise_not(image,masked_image,blackimg);
        // imshow("Not Mask Applied" + to_string(i),masked_image); //using the not mask also removes the white lines into black

        return masked_image;
        
}

Mat masking2l(Mat image,int i) //works perfectly for the second set of lines
{
    std::vector<cv::Point> polygon_corners;
    Mat masked_image;

    //create a black image of the same size as the original image
        Mat blackimg = Mat::zeros(image.size(),CV_8UC1);
        // imshow("Blackimage" + to_string(i),blackimg);

    //taking the polygon area (hard coding)
          
        polygon_corners.push_back(cv::Point(599, 29));  
        polygon_corners.push_back(cv::Point(663, 5)); 
        polygon_corners.push_back(cv::Point(1225,415 )); 
        polygon_corners.push_back(cv::Point(1162,488)); 

        fillPoly(blackimg, polygon_corners, Scalar(255, 255, 255)); //fill the mask with white
        // imshow("Blackimage filled" + to_string(i),blackimg);

        //applyinng the mask to the original image
        bitwise_and(image,image,masked_image,blackimg);
        // imshow("And Mask Applied" + to_string(i),masked_image);
        // bitwise_not(image,masked_image,blackimg);
        // imshow("Not Mask Applied" + to_string(i),masked_image); //using the not mask also removes the white lines into black

        return masked_image;
        
}

static Mat findlines(Mat img,int i)
{
    Mat finallines,grayImage;
    
    //step1: convert to grayscale 
    cvtColor(img, grayImage, COLOR_BGR2GRAY);
    imshow("Grayimage" + to_string(i),grayImage);
    // Step 2: applying threshhold
    threshold(grayImage, grayImage,130 , 255, cv::THRESH_BINARY); 
    // Step 3: bluring
    medianBlur(grayImage,grayImage,5);
    imshow("gray blur image",grayImage);


    grayImage.copyTo(finallines);
    return finallines;

}

 Mat applyMSER(Mat img, int i)
{
        Mat output(img.size(),CV_8UC3);
        output = cv::Scalar(255,255,255);
        // cvtColor(img,img, COLOR_BGR2GRAY);
        vector<std::vector<cv::Point>> mpoints;
        std::vector< Rect > bboxes;
        //detect MSER features
        //using MSER
        Ptr<MSER> mser;
        mser = cv::MSER::create(5,800,8000,0.2,5,1000);//15,1000,8000,0.5,0.5,1000
        mser->detectRegions(img,mpoints,bboxes);

        //drawing the detected regions
        for (size_t i = 0; i < bboxes.size(); i++) {
        rectangle(img, bboxes[i], Scalar(0, 255, 0));  // Draw rectangle around the detected region
        }
        for (size_t i = 0; i < mpoints.size(); i++) {
        polylines(img, mpoints[i], true, Scalar(0, 255, 0), 2);
        }
        //drawing with random colors
        RNG rng;
        for(vector<vector<Point>>::iterator it=mpoints.begin();it!=mpoints.end();++it)
        {
            //generating random color
            cv::Vec3b c(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255));

            //for each point in MSER set
            for (auto itPts = it->begin(); itPts != it->end(); ++itPts) {
            // If the point is part of the foreground (white in grayscale, for example)
            // we color it with the random color
            output.at<Vec3b>(*itPts) = c;
        }


        }




        // Display the image with detected regions
        imshow("MSER Regions"+ to_string(i), img);
        imshow("MSER Regions with random colors"+ to_string(i), output);
return img;

}


void sobelApplied(Mat img)
{
    Mat sobelx, sobely, sobelxy;
    Sobel(img, sobelx, CV_64F, 1, 0, 5);
    Sobel(img, sobely, CV_64F, 0, 1, 5);
    Sobel(img, sobelxy, CV_64F, 1, 1, 5);

    // Display Sobel edge detection images
    imshow("Sobel X", sobelx);
    
    imshow("Sobel Y", sobely);
 
    imshow("Sobel XY using Sobel() function", sobelxy);
    


}

void gammaCorrection(const cv::Mat& src, cv::Mat& dst, float gamma) {
    CV_Assert(gamma >= 0);
    cv::Mat lut(1, 256, CV_8UC1);
    for (int i = 0; i < 256; i++) {
        lut.at<uchar>(i) = cv::saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
    }
    cv::LUT(src, lut, dst);
}




