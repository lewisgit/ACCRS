#pragma once
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "MyTypes.h"
#include "colors.h"
//#include "colors.h"
using std::string;
using cv::Mat;
using cv::Rect;
using std::vector;

extern bool mostSimMatch(const string &iStr, string &oStr);
extern bool conNuMostSimMatch(const string &iStr, string &oStr);

class NumberDetector {
public:
	NumberDetector(string& imgpath,vector<Rect>& ctpn_boxes, vector<east_bndbox>& east_boxes, vector<Rect>& mser_boxes,Rect deeplab_box);
	string detectVerticalNumber(string savepath);
	void detectHorizontalNumber(vector<vector<Rect>>& dst_rects, string savepath);
	string detectRowNumber_front(string savepath);

	int judgeSide();
	int judgeMode_east();
private:
	string _imgpath;
	Mat _org_img;
	vector<Rect> _ctpn_boxes;
	vector<east_bndbox> _east_boxes;
	vector<Rect> _mser_boxes;
	Rect _deeplab_box;
private:
	void filterByDeeplab(vector<Rect>& input_boxes, vector<Rect>& output_boxes);
	void filterByCTPNVertical(vector<Rect>& input_boxes, vector<Rect>& ctpn_filter, int distThres, int midlineThres);
	void geometryFilter(vector<Rect>& input_boxes, vector<Rect>& geometry_filter, int strict_mode);
	void boxClusteringVertical(vector<Rect>& input_boxes, vector<vector<Rect>>& clusters, int midlineThres, int heightThres, int borderThres);

	int getRectClusterLength(vector<Rect>& clusters, int direction);

	void filterEASTbyDeeplab(vector<Rect>& src, vector<Rect>& dst);
	void simpleClusterByHorizon(vector<Rect>& src,vector<vector<Rect>>& clusters, int thres);

	void adjustEastByMser(vector<Rect> rects);
	int findVerifyNumber(Rect& rect, Rect& verifyRect, int midthres, int heightThres);

	void locateNumber(vector<east_bndbox>& east_boxes, vector<Rect> cluster, vector<Rect>& boxes);
	void filterByEast(vector<Rect>& mser_boxes, vector<east_bndbox>& east_boxes, vector<Rect>& filtered_boxes, float filterHeightThres, float filterYThres, float filterDistThres);

	void eastBoderRefine(vector<Rect>& boxes, vector<Rect>& filtered_mser);

	void filterEastByPosition(vector<Rect>& src, vector<Rect>& dst);
};
