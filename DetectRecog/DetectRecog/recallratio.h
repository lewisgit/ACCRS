#pragma once
#include <rapidxml/rapidxml.hpp>
#include <rapidxml/rapidxml_utils.hpp>
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <dirent.h>
using cv::Rect;
using std::vector;
using std::string;
using std::min;
using std::max;
using namespace rapidxml;

void calRatio1(vector<Rect> rects, string filename, float* ratio1, float* ratio2,int *num);
void calRatio2(vector<vector<Rect>> rects, string filename, float* ratio1, float* ratio2, int* num);
void iterDir(string path, vector<string>& filenames);