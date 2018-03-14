#include "getMSER.hpp"
#include <opencv2/stitching.hpp>
using namespace cv;
using namespace std;

std::vector<cv::Rect> getMSER(cv::Mat img, int thresDelta, int minArea, int maxArea) {
	Ptr<MSER> mserpos = MSER::create(thresDelta, minArea, maxArea);
	vector<vector<Point>> posContours;
	vector<Rect> posbox;
	mserpos->detectRegions(img, posContours, posbox);
	return posbox;
}