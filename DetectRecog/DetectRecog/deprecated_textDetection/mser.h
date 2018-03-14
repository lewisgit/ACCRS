#ifndef __OPENCV_PRECOMP_H__
#define __OPENCV_PRECOMP_H__

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

//#include <opencv2/core/utility.hpp>
//#include <opencv2/core/private.hpp>
//#include <opencv2/core/ocl.hpp>
//#include <opencv2/core/hal/hal.hpp>

#include <algorithm>
using std::vector;
using cv::KeyPoint;
using cv::Point;
using cv::Mat;
#ifdef HAVE_TEGRA_OPTIMIZATION
#include "opencv2/features2d/features2d_tegra.hpp"
#endif

class CV_EXPORTS_W MYMSER : public cv::FeatureDetector
{
public:
	//! the full constructor
	CV_WRAP explicit MYMSER(int _delta = 5, int _min_area = 60, int _max_area = 14400,
		double _max_variation = 0.25, double _min_diversity = .2,
		int _max_evolution = 200, double _area_threshold = 1.01,
		double _min_margin = 0.003, int _edge_blur_size = 5);

	//! the operator that extracts the MSERs from the image or the specific part of it
	CV_WRAP_AS(detect) void operator()(const cv::Mat& image, CV_OUT std::vector<std::vector<cv::Point> >& msers,
		const cv::Mat& mask = cv::Mat()) const;
	//AlgorithmInfo* info() const;

protected:
	void detectImpl(const cv::Mat& image, vector<cv::KeyPoint>& keypoints, const cv::Mat& mask = cv::Mat()) const;

	int delta;
	int minArea;
	int maxArea;
	double maxVariation;
	double minDiversity;
	int maxEvolution;
	double areaThreshold;
	double minMargin;
	int edgeBlurSize;
};

//typedef MSER MserFeatureDetector;



#endif