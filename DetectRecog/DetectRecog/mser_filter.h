#ifndef _MSER_FILTER_H
#define _MSER_FILTER_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <cmath>
#include "utils.h"
#include <iostream>

using namespace std;

#define WRONG 0
#define ROW_MODE 1
#define ROW_LOOSE_MODE 2	//第二次行聚类时放宽了一些条件
#define COL_MODE 3
#define COL_LOOSE_MODE 4

class MserFilter
{
public:
	MserFilter(cv::Mat srcImg, cv::Rect deeplabBbox);
	void drawMsers(string outputPath);
	void drawBboxes(string outputPath, vector<cv::Rect> bboxes);
	void drawClusters(string outputPath, vector<vector<cv::Rect>> clusters);
	void drawResult(string outputPath, vector<cv::Rect> result);
	void filter(void);
	int judgeSide(void);
	virtual ~MserFilter();
public:
	vector<vector<cv::Rect>> rowClusters;
	vector<vector<cv::Rect>> colClusters;
	vector<cv::Rect> rowResult;
	vector<cv::Rect> colResult;

private:
	cv::Mat srcImg;
	cv::Rect deeplabBbox;
	vector<vector<cv::Point>> msers;
	vector<cv::Rect> bboxes;

	// 阈值
	double mserHeightSmall, mserHeightHuge, mserWidthHuge;

private:
	//一些供调用的功能函数
	bool isClose(cv::Rect r1, cv::Rect r2, int disThres, int mode);			//用于聚类时判断是否属于一类
	void deeplabFilter(void);
	void singleBboxFilter(void);
	void clusterFilter(void);
	int clusterProcess(int mode);
	void findMainRowCluster(vector<cv::Rect>& mainRowCluster);
	void delByMainRow(vector<cv::Rect> mainRowCluster);
	void buildRowResult(void);
	void findMainColCluster(vector<cv::Rect>& mainColCluster);
	void delByMainCol(vector<cv::Rect> mainColCluster);
	void buildColResult(void);
	void MserFilter::cluster(int disThres, vector<cv::Rect> bboxes, vector<vector<cv::Rect>>& clusters, int mode);
};

#endif
