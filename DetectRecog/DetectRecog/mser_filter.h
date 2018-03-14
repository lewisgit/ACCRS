#ifndef _MSER_FILTER_H
#define _MSER_FILTER_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <cmath>
#include "utils.h"
#include <iostream>

using namespace cv;
using namespace std;

#define ROW_MODE 1
#define COL_MODE 2
#define WRONG 0

class MserFilter
{
public:
	MserFilter(Mat srcImg, Rect deeplabBbox);
	MserFilter(Mat srcImg, vector<Rect> bboxes, Rect deeplabBbox);
	void drawMsers(string outputPath);
	void drawBboxes(string outputPath);
	void drawClusters(string outputPath, vector<vector<Rect>> clusters);
	void filter(void);
	void patch(void);
	int judgeSide(void);
	virtual ~MserFilter();
public:
	vector<vector<Rect>> rowClusters;
	vector<vector<Rect>> colClusters;

private:
	Mat srcImg;
	Rect deeplabBbox;
	vector<vector<Point>> msers;
	vector<Rect> bboxes;

	// 阈值
	double mserHeightSmall, mserHeightHuge, mserWidthHuge;

private:
	//一些供调用的功能函数
	bool isClose(Rect r1, Rect r2, int disThres, int mode);			//用于聚类时判断是否属于一类
	void deeplabFilter(void);
	void singleBboxFilter(void);
	void clusterFilter(void);
	int rowClusterFilter(void);
	int colClusterFilter(void);
	void cluster(int disThres, int mode);
};

#endif
