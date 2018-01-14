#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/stitching.hpp>
#include <vector>

std::vector<cv::Rect> getMSER(cv::Mat img, int thresDelta, int minArea, int maxArea);