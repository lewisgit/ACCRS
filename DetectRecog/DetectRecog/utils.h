#pragma once
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <math.h>
#include "MyTypes.h"
//#include <dirent.h>
using std::string;
//#include <cmath>
using cv::Rect;
using std::vector;
void clusteringRects(vector<Rect> rects, vector<vector<Rect>>& clusters, int xthres, int ythres, int heightthres);
void removeOverlap(vector<Rect>& rects, vector<Rect>& slim_rects);

bool isOverlap(const Rect &rc1, const Rect &rc2);
void scaleSingleRect(Rect& rect, float scalex, float scaley, int maxx, int maxy);
bool sortByX(const Rect& lhs, const Rect& rhs);
bool sortByY(const Rect& lhs, const Rect& rhs);
void mergeVerifyNum(vector<Rect>& rects);
void findClusterAttr(vector<vector<Rect>>& clusters, vector<int> &attr);

int findCompanyNumber(vector<vector<Rect>>& clusters, int mid, int yThres, int xThres);
int findContainerType(vector<vector<Rect>>& clusters, int mid, int yThres, int xThres);
void splitRectsByAverageHeight(vector<Rect>& src, vector<Rect>& dst, float avrageHeight);
void mergeSameHorizon(vector<Rect>& src, vector<Rect>& dst);

Rect eastbox2rect(east_bndbox box);
void showClustering(cv::Mat& drawimg, vector<vector<Rect>>& clusters);

bool verifyContainerCode(string containerCode);
void getAllFiles(string rootpath, vector<string>& files);