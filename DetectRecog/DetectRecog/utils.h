#pragma once
#include <vector>
#include <opencv2/core.hpp>
//#include <cmath>
using cv::Rect;
using std::vector;
void clusteringRects(vector<Rect> rects, vector<vector<Rect>>& clusters, int xthres, int ythres, int heightthres);
void removeOverlap(vector<Rect>& rects, vector<Rect>& slim_rects);

bool isOverlap(const Rect &rc1, const Rect &rc2);