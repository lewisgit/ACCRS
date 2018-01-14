//#include "pythonCall.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <string>

#include "socketcpp/tcpClient.h"
#include "colors.h"

#include "textDetection/TextDetection.h"
#include <fstream>
#include "getMSER.hpp"

#include "utils.h"

using cv::Mat;
using std::cout;
using std::endl;
using std::vector;
//EAST_Detector* east_detector;
string east_trained;
string east_pyfile;

string east_path;
/*
void init() {
	string charname = "D:/Lewis/projects/container_ocr/ctpn_vs2015_py36/windows/../Build/x64/Release/classification.exe";
	//std::wstring wcharname(charname.begin(),charname.end());
	wchar_t** args = new wchar_t*;
	wchar_t* proName = new wchar_t[charname.size()+1];
	for (int i = 0; i < charname.size() + 1; i++) {
		proName[i] = charname[i];
	}
	args[0] = proName;
	Py_SetProgramName(proName);
	Py_Initialize();
	//wchar_t** wStr = new wchar_t*;
	//wStr[0] = new wchar_t[10];
	//wStr[0][0] = 'c';
	PySys_SetArgv(1, args);
	//Py_SetProgramName(wStr);
	//PyRun_SimpleString("import sys");
	//string seteastpath = "sys.path.append('" + east_path + "')";
	//PyRun_SimpleString(seteastpath.c_str());
	east_detector = new EAST_Detector(east_trained,east_pyfile);
}
*/
/*
void detect_back(string imgpath) {
	vector<bndbox> east_boxes;
	east_detector->detect(imgpath, east_boxes);
	for (auto box : east_boxes) {
		cout << box.x0 << endl;
	}
}
*/

void testMSERnSWT(string imgpath) {
	vector<string> gtset;
	string line;
	cv::Mat image = cv::imread(imgpath);
	ifstream fin("D:\\Lewis\\projects\\container_ocr\\ctpn_vs2015_py36\\windows\\classification\\textDetection\\resource\\gt_demo");
	while (getline(fin, line)) {
		gtset.push_back(line);
	}
	TextDetection td;
	cout << td.detectText(image, gtset) << endl;
	cv::waitKey(0);
}


#include "config.h"

void EastMserFilter(vector<Rect>& boxes, vector<east_bndbox>& east_boxes, vector<Rect>& filter_boxes, float heightThres, float yThres) {
	for (auto box : boxes) {
		for (auto east_bndbox : east_boxes) {
			float avgheight = (east_bndbox.y3 - east_bndbox.y0 + east_bndbox.y2 - east_bndbox.y1) / 2.0;
			float avgy= (east_bndbox.y0 + east_bndbox.y3 + east_bndbox.y1 + east_bndbox.y2) / 4.0;
			float midline = box.x + box.height / 2.0;
			float dist;
			if (midline < east_bndbox.x0)
				dist = east_bndbox.x0 - midline;
			else if (midline > east_bndbox.x1)
				dist = midline - east_bndbox.x1;
			else
				dist = 0;

			if (abs(box.height - avgheight) < heightThres && abs(box.y+ box.height/2.0- avgy) < yThres && dist<3*box.width && box.height>box.width) {
				filter_boxes.push_back(box);
			}
		}
	}
}



void locateNumber(vector<east_bndbox>& east_boxes, vector<Rect> cluster, vector<Rect>& boxes) {
	int eastnum = east_boxes.size();
	for (int i = 0; i < eastnum; i++) {
		east_bndbox box = east_boxes[i];
		int x0 = std::min(box.x0, box.x3);
		int x1 = std::max(box.x1, box.x2);
		int y0 = std::min(box.y0, box.y1);
		int y1 = std::max(box.y2, box.y3);
		Rect rect0(x0, y0, x1 - x0, y1 - y0);
		for (auto rect1 : cluster) {
			if (isOverlap(rect0, rect1)) {
				boxes.push_back(rect0);
				break;
			}
		}
	}
}


void detectRegion(string& imgpath, vector<east_bndbox>& east_boxes) {
	Mat img_org = imread(imgpath);
	Mat img_gray;
	cvtColor(img_org, img_gray, CV_BGR2GRAY);
	vector<Rect> boxes;
	boxes=getMSER(img_gray,mserThres,mserMinArea,mserMaxArea);
	Mat showimg;
	img_org.copyTo(showimg);

	vector<Rect> filter_boxes;
	EastMserFilter(boxes, east_boxes, filter_boxes, filterHeightThres, filterYThres);

	vector<Rect> slim_rects;
	removeOverlap(filter_boxes, slim_rects);

	vector<vector<Rect>> clusters;
	clusteringRects(slim_rects, clusters,50,20,15);

	int clusterNum = clusters.size();

	int mid=-1,up=-1,down=-1;
	for (int i = 0; i < clusterNum; i++) {
		if (clusters[i].size() > 5) {
			mid = i;
			break;
		}
	}
	if (mid > 0) {
		Rect rect = clusters[mid - 1][0];
		if (clusters[mid][0].y - (rect.y + rect.height) < 15) {
			up = mid - 1;
		}
	}
	if (mid < clusterNum - 1) {
		Rect rect = clusters[mid + 1][0];
		if (rect.y - (clusters[mid][0].y + clusters[mid][0].height) < 15) {
			down = mid + 1;
		}
	}

	vector<Rect> midboxes;
	locateNumber(east_boxes, clusters[mid], midboxes);
	vector<Rect> upboxes;
	if ( up != -1)
		locateNumber(east_boxes, clusters[up], upboxes);
	vector<Rect> downboxes;
	if (down != -1)
		locateNumber(east_boxes, clusters[down], downboxes);

	Rect verifyDigit;
	int verifyid = 0;

	int area = 0;
	for (auto rect1 : clusters[mid]) {
		int overlap = 0;
		for (auto rect2 : midboxes) {
			if (isOverlap(rect1, rect2)) {
				overlap = 1;
				break;
			}
		}
		if (overlap == 0) {
			if (rect1.width*rect1.height > area) {
				verifyDigit = rect1;
				verifyid = 1;
				area = rect1.width*rect1.height;
			}
		}
	}
	if (verifyid == 1)
		midboxes.push_back(verifyDigit);


	string imgsavepath = "D:/Lewis/projects/numbers/";
	int count = 0;
	for (auto rect : upboxes) {
		Mat img = img_org(rect);
		string path = imgsavepath + to_string(count) + ".jpg";
		cv::imwrite(path, img);
		count++;
	}
	for (auto rect : midboxes) {
		Mat img = img_org(rect);
		string path = imgsavepath + to_string(count) + ".jpg";
		cv::imwrite(path, img);
		count++;
	}
	for (auto rect : downboxes) {
		Mat img = img_org(rect);
		string path = imgsavepath + to_string(count) + ".jpg";
		cv::imwrite(path, img);
		count++;
	}

	vector<string> strs;
	crnn_request(imgsavepath, strs);

	for (int i = 0; i < clusterNum; i++) {
		for (auto rect : clusters[i]) {
			float v = 255.0*i / clusterNum;
			cv::Scalar rectColor(v, v, v);
			rectangle(showimg, rect, rectColor);
		}
	}

	

	//for (auto box : filter_boxes) {
	//	rectangle(showimg, box, red);
	//}
	imshow("filter", showimg);
	waitKey(0);
}

int main(int argc,char** argv) {
	if (*argv[5] == '1')
		testMSERnSWT(argv[2]);
	initSocket();
	string imgpath = argv[2];
	vector<cv::Rect> ctpn_rects;
	vector<east_bndbox> east_boxes;
	//string imgpath = argv[2];
	if(*argv[3]=='1')
		ctpn_request(argv[2],ctpn_rects);
	if (*argv[4] == '1')
		east_request(argv[2],east_boxes);
	
	detectRegion(imgpath, east_boxes);

	Mat orig_img = cv::imread(imgpath);
	cout << orig_img.cols << endl;
	cout << orig_img.rows << endl;
	Mat draw_img;
	orig_img.copyTo(draw_img);
	for (auto rect : ctpn_rects)
		cv::rectangle(draw_img, rect, red);
	for (auto box : east_boxes) {
		vector<cv::Point> points;
		points.push_back(cv::Point(box.x0, box.y0));
		points.push_back(cv::Point(box.x1, box.y1));
		points.push_back(cv::Point(box.x2, box.y2));
		points.push_back(cv::Point(box.x3, box.y3));
		cv::polylines(draw_img, points, 1, green,2);
	}
	cv::imshow("nn detection", draw_img);
	cv::waitKey(0);

	/*printf("%s\n", argv[0]);
	east_pyfile = argv[1];
	east_path = "D:/Lewis/projects/ACCRS/EAST";
	east_trained = "D:/Lewis/projects/container_ocr/EAST-master/east_icdar2015_resnet_v1_50_rbox";

	init();
	detect_back("D:/OneDrive/Lewis/projects/BoxNumberDetect/imgs/126_0062_170214_165717_Rear_03.jpg");
	*/
}