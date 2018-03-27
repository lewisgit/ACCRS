//#include "pythonCall.h"

#define DLL_GEN

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <string>

#include "socketcpp/tcpClient.h"

#include <time.h>
//#include "textDetection/TextDetection.h"
#include <fstream>
#include "getMSER.hpp"
#include "NumberDetector.h"
#include "utils.h"
#include "mser_filter.h"
#include "detect.h"
#include "config.h"

//#include <windows.h>
extern bool mostSimMatch(const string &iStr, string &oStr);
extern bool conNuMostSimMatch(const string &iStr, string &oStr);
using std::to_string;
using cv::Mat;
using std::cout;
using std::endl;
using std::vector;

string east_trained;
string east_pyfile;

string east_path;


void DeeplabEastMserFilter(vector<Rect>& boxes, vector<east_bndbox>& east_boxes,Rect deeplab_box, vector<Rect>& filter_boxes, float heightThres, float yThres) {
	for (auto box : boxes) {
		for (auto east_bndbox : east_boxes) {
			float avgheight = (east_bndbox.y3 - east_bndbox.y0 + east_bndbox.y2 - east_bndbox.y1) / 2.0;
			float avgy= (east_bndbox.y0 + east_bndbox.y3 + east_bndbox.y1 + east_bndbox.y2) / 4.0;
			float midline = box.x + box.height / 2.0;
			float dist;
			if (midline < east_bndbox.x0) {
				dist = east_bndbox.x0 - midline;
				if (dist > box.width * 3)
					dist = 1000;
			}
			else if (midline > east_bndbox.x1)
				dist = midline - east_bndbox.x1;
			else
				dist = 0;
			//box width largers than 3 

			float whratio = box.height*1.0 / box.width;
			if (whratio > 1.3 && box.width > 3 && abs(box.height - avgheight) < heightThres && abs(box.y + box.height / 2.0 - avgy) < yThres && dist<2 * box.height && box.height>box.width && isOverlap(deeplab_box, box) && box.x > (deeplab_box.x + deeplab_box.width*0.5) && box.y < deeplab_box.y + deeplab_box.height*0.7) {
				filter_boxes.push_back(box);
			}
		}
	}
}



void locateNumber(vector<east_bndbox>& east_boxes, vector<Rect> cluster, vector<Rect>& boxes) {
	int eastnum = east_boxes.size();
	for (int i = 0; i < eastnum; i++) {
		east_bndbox box = east_boxes[i];
		int x0 = min(box.x0, box.x3);
		int x1 = max(box.x1, box.x2);
		int y0 = min(box.y0, box.y1);
		int y1 = max(box.y2, box.y3);
		if (x0 < 0)
			x0 = 0;
		if (y0 < 0)
			y0 = 0;
		Rect rect0(x0, y0, x1 - x0, y1 - y0);
		for (auto rect1 : cluster) {
			if (isOverlap(rect0, rect1)) {
				int width = min(rect0.x + rect0.width, rect1.x + rect1.width) - max(rect0.x, rect1.x);
				int height= min(rect0.y + rect0.height, rect1.y + rect1.height) - max(rect0.y, rect1.y);
				if (width > 0 && height > 0) {
					int area = rect1.width*rect1.height;
					if (width*height*1.0 / area > 0.9) {
						boxes.push_back(rect0);
						break;
					}
				}
			}
		}
	}
}


string detectRegion(string& imgpath, vector<east_bndbox>& east_boxes, Rect deeplab_box, vector<Rect>& mser_boxes, NumberDetector* detector, long start) {
	Mat img_org = cv::imread(imgpath);
	Mat img_gray;
	cvtColor(img_org, img_gray, CV_BGR2GRAY);
	//vector<Rect> boxes;
	//boxes = getMSER(img_gray, mserThres, mserMinArea, mserMaxArea);
	Mat showimg, showimg2, showimg3,showimg4,showimg5;
	img_org.copyTo(showimg);
	img_org.copyTo(showimg2);
	img_org.copyTo(showimg3);
	img_org.copyTo(showimg4);
	img_org.copyTo(showimg5);




	vector<Rect> filter_boxes;
	DeeplabEastMserFilter(mser_boxes, east_boxes, deeplab_box, filter_boxes, eastFilterHeightThres, eastFilterYThres);

	vector<Rect> slim_rects;
	removeOverlap(filter_boxes, slim_rects);

	vector<vector<Rect>> clusters;
	clusteringRects(slim_rects, clusters, 100, 20, 15);
	 
#if 0
	for (auto rect : filter_boxes) {
		rectangle(showimg3, rect, red, 1);
	}
	imshow("mser", showimg3);
	cv::waitKey(0);


	Mat draw_img;
	img_org.copyTo(draw_img);
	for (auto box : east_boxes) {
		vector<cv::Point> points;
		points.push_back(cv::Point(box.x0, box.y0));
		points.push_back(cv::Point(box.x1, box.y1));
		points.push_back(cv::Point(box.x2, box.y2));
		points.push_back(cv::Point(box.x3, box.y3));
		cv::polylines(draw_img, points, 1, green, 2);
	}
	cv::imshow("nn detection", draw_img);
	cv::waitKey(0);
#endif 

	int clusterNum = clusters.size();

	vector<int> attrs;
	findClusterAttr(clusters, attrs);
	int mid = -1, up = -1, down = -1;
	for (int i = 0; i < clusterNum; i++) {
		float sumheight = 0;
		int clustersize = clusters[i].size();
		for (auto rect : clusters[i])
			sumheight += rect.height;
		float avgheight = sumheight / clustersize;

		float clusterlength = clusters[i][clustersize - 1].x + clusters[i][clustersize - 1].width - clusters[i][0].x;
		float lhratio = clusterlength / avgheight;
		if (attrs[i] > 5 ) {
			mid = i;
			break;
		}
	}


	//add processing code for "mid=-1" here
	//---
	//end
	vector<Rect> midboxes, upboxes, downboxes;
	if (mid != -1) {
		//find "company code" and "container type" 
		up = findCompanyNumber(clusters, mid, 30, 200);
		down = findContainerType(clusters, mid, 100, 200);

		/*if (mid > 0) {
			Rect rect = clusters[mid - 1][0];
			if (clusters[mid][0].y - (rect.y + rect.height) < 25  && abs(clusters[mid][0].x- rect.x)<100) {
				up = mid - 1;
			}
		}
		if (mid < clusterNum - 1) {
			Rect rect = clusters[mid + 1][0];
			if (rect.y - (clusters[mid][0].y + clusters[mid][0].height) < 25) {
				down = mid + 1;
			}
		}*/

		

		locateNumber(east_boxes, clusters[mid], midboxes);
	
		if (up != -1) {
			/*for (auto rect : clusters[up]) {
				rectangle(showimg5, rect, red, 1);
			}
			imshow("up",showimg5);
			cv::waitKey(0);*/
			locateNumber(east_boxes, clusters[up], upboxes);

			if (midboxes.size() == 1) {
				Rect rect = midboxes[0];
				midboxes.pop_back();
				
				Rect rect1(rect.x,rect.y,rect.width*4/7.0,rect.height);
				Rect rect2(rect.x+rect.width * 5 / 7.0, rect.y, rect.width * 2 / 7.0, rect.height);
				midboxes.push_back(rect1);
				midboxes.push_back(rect2);
			}

		}
		if (down != -1)
			locateNumber(east_boxes, clusters[down], downboxes);

		Rect verifyDigit;
		int verifyid = 0;


		//rectangle(showimg2, midboxes[0], blue,1);
		//rectangle(showimg2, midboxes[1], blue, 1);
		//rectangle(showimg2, downboxes[0], blue,1);
		//cv::putText(showimg2, "TGHU", cv::Point(midboxes[0].x, midboxes[0].y + midboxes[0].height), 1.8, 1.3, green);
		//cv::putText(showimg2, "213783 5", cv::Point(midboxes[1].x, midboxes[1].y + midboxes[1].height), 1.8, 1.3, green);
		//cv::putText(showimg2, "22G1", cv::Point(downboxes[0].x, downboxes[0].y + downboxes[0].height), 1.8, 1.3, green);
		//for (auto box : filter_boxes) {
		//	rectangle(showimg, box, red);
		//}
		//imshow("filter", showimg2);
		//cv::waitKey(0);

		//



		Rect rightMostRect = clusters[mid][clusters[mid].size() - 1];
		int leftBorder = rightMostRect.x + rightMostRect.width;


		sort(midboxes.begin(), midboxes.end(), sortByX);
		vector<Rect> newmidBoxes;
		if (up == -1) {
			/*Rect rect;
			rect.x = midboxes[1].x;
			rect.y = midboxes[1].y;
			rect.height = midboxes[1].height;
			rect.width = leftBorder - rect.x;
			newmidBoxes.push_back(midboxes[0]);
			newmidBoxes.push_back(rect);*/
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
		}
		else {
			Rect rect;
			int nn = midboxes.size() - 1;
			rect.x = midboxes[nn].x;
			rect.y = midboxes[nn].y;
			rect.height = midboxes[nn].height;
			rect.width = leftBorder - rect.x;
			//newmidBoxes.push_back(midboxes[0]);
			for (int i = 0;i < nn;i++)
				newmidBoxes.push_back(midboxes[i]);
			newmidBoxes.push_back(rect);
			midboxes = newmidBoxes;
		}

	}
	else {
		vector<vector<Rect>> dst_rects;
		detector->detectHorizontalNumber(dst_rects,"");
		upboxes = dst_rects[0];
		midboxes = dst_rects[1];
		downboxes = dst_rects[2];

		if (upboxes.size() > 0)
			up = 1;
		if (midboxes.size() > 0)
			mid = 1;
		if (downboxes.size() > 0)
			down = 1;
		//NumberDetector detector(imgpath,);
	}
	//int area = 0;
	//for (auto rect1 : clusters[mid]) {
	//	int overlap = 0;
	//	for (auto rect2 : midboxes) {
	//		if (isOverlap(rect1, rect2)) {
	//			overlap = 1;
	//			break;
	//		}
	//	}
	//	if (overlap == 0) {
	//		if (rect1.width*rect1.height > area) {
	//			verifyDigit = rect1;
	//			verifyid = 1;
	//			area = rect1.width*rect1.height;
	//		}
	//	}
	//}
	//if (verifyid == 1)
	//	midboxes.push_back(verifyDigit);


	string imgsavepath = "D:/savedimgs/";

	

	int count = 0;
	for (auto rect : upboxes) {
		Mat img = img_org(rect);
		string path = imgsavepath + to_string(count) + ".jpg";
		cv::imwrite(path, img);
		count++;
	}
	
	//mergeVerifyNum(midboxes);
	for (auto rect : midboxes) {
		

		scaleSingleRect(rect, 0.1, 0.1, img_org.cols, img_org.rows);
		Mat img = img_org(rect);
		if (rect.width < rect.height) {
			Mat newimg(rect.height, rect.width * 2, img.type());
			cv::hconcat(img, img, newimg);
			img = newimg;
		}
		string path = imgsavepath + to_string(count) + ".jpg";

		cv::imwrite(path, img);
		count++;
	}
	for (auto rect : downboxes) {
		scaleSingleRect(rect, 0.2, 0.1, img_org.cols, img_org.rows);
		Mat img = img_org(rect);
		string path = imgsavepath + to_string(count) + ".jpg";
		cv::imwrite(path, img);
		count++;
	}

	vector<string> strs;
	crnn_request(imgsavepath, strs);


	long end = clock();

	double dur = (double)(end - start);
	printf("duration Time:%f\n", (dur / CLOCKS_PER_SEC));

	//print recognized text on the image;
	//Mat showimg4;
	//img_org.copyTo(showimg4);
	count = 0;

	string output_str="";

	for (auto rect : upboxes) {
		rectangle(showimg4, rect, green, 2);
		
		
		string chechdatabase;
		conNuMostSimMatch(strs[count], chechdatabase);
		output_str = output_str + chechdatabase + " ";

		cv::putText(showimg4, chechdatabase, cv::Point(rect.x, rect.y), 1.9, 1.7, red, 2);
		count++;
	}
	for (int i = 0;i < midboxes.size();i++) {
		Rect rect = midboxes[i];
		rectangle(showimg4, rect, green, 2);
		if (up == -1 && i==0) {
			char firstchar = strs[count].c_str()[0];
			string restchars = strs[count].c_str() + 1;
			//string newstr;
			
			string checkdatabase;
			conNuMostSimMatch(strs[count], checkdatabase);
			output_str = output_str + checkdatabase + " ";

			/*if (firstchar == 'i')
				newstr = "t" + restchars;
			else
				newstr = strs[count];*/
			cv::putText(showimg4, checkdatabase, cv::Point(rect.x, rect.y), 1.9, 1.7, red, 2);

		}
		else if (midboxes[i].width<midboxes[i].height) {
			int lastid = strs[count].size();
			string lastdigit = strs[count].c_str() + lastid - 1;
			if (lastdigit[0] == 'z')
				lastdigit = '2';
			else if (lastdigit[0] == 'o' || lastdigit[0] == 'c')
				lastdigit = '0';
			else if (lastdigit[0] == 'b')
				lastdigit = '3';
			else if (lastdigit[0] == 'd' || lastdigit[0] == 'q')
				lastdigit = '1';

			output_str = output_str +  lastdigit;
			cv::putText(showimg4, lastdigit, cv::Point(rect.x, rect.y), 1.9, 1.7, red, 2);
		}
		
		/*else if (i == midboxes.size() - 1 && up == -1 && midboxes.size()>2) {
			int lastid = strs[count].size();
			string lastdigit = strs[count].c_str()+lastid-1;
			if (lastdigit[0] == 'z')
				lastdigit = '2';
			else if (lastdigit[0] == 'o')
				lastdigit = '0';
			cv::putText(showimg4, lastdigit, cv::Point(rect.x, rect.y), 1.9, 1.7, red, 2);
		}*/
		else {
			cv::putText(showimg4, strs[count], cv::Point(rect.x, rect.y), 1.9, 1.7, red, 2);
			output_str = output_str + strs[count];
		}
		/*if (i == 0 && up == -1) {
			string chechdatabase;
			conNuMostSimMatch(strs[count], chechdatabase);
			output_str = output_str + chechdatabase + " ";
		}
		else {
			output_str += strs[count];
		}*/

		count++;
	}
	for (auto rect : downboxes) {
		rectangle(showimg4, rect, green, 2);

		string findDatabase;

		mostSimMatch(strs[count],findDatabase);
		cv::putText(showimg4, findDatabase, cv::Point(rect.x, rect.y), 1.9, 1.7, red, 2);

		output_str = output_str +  " " + findDatabase;
		/*if(strs[count][0]=='2')
			cv::putText(showimg4, "22g1", cv::Point(rect.x, rect.y ), 1.9, 1.7, red, 2);
		else if(strs[count][0] == '4')
			if (strs[count][1] == '2')
				if (strs[count][2] == '6'|| strs[count][2] == 'g')
					cv::putText(showimg4, "42g1", cv::Point(rect.x, rect.y), 1.9, 1.7, red, 2);
				else
					cv::putText(showimg4, "42u1", cv::Point(rect.x, rect.y), 1.9, 1.7, red, 2);
			else
				cv::putText(showimg4, "45g1", cv::Point(rect.x, rect.y ), 1.9, 1.7, red, 2);
		else
			cv::putText(showimg4, strs[count], cv::Point(rect.x, rect.y), 1.9, 1.7, red, 2);*/
		count++;
	}

	//cout << output_str << endl;
#if 0
	cv::imshow(imgpath, showimg4);
	cv::waitKey(0);
#endif 
#if 0
	for (int i = 0; i < clusterNum; i++) {
		cv::Scalar rectColor(rand()*1.0 / RAND_MAX * 255, rand()*1.0 / RAND_MAX * 255, rand()*1.0 / RAND_MAX * 255);
		for (auto rect : clusters[i]) {
			float v = 255.0*i / clusterNum;
			
			rectangle(showimg5, rect, rectColor);
		}
	}
	cv::imshow("CLUSTER", showimg5);
	cv::waitKey(0);
#endif
	std::transform(output_str.begin(), output_str.end(),output_str.begin(), ::toupper);
	return output_str;
}


bool  CheckFolderExist(const string &strPath)
{
	WIN32_FIND_DATA  wfd;
	bool rValue = false;
	HANDLE hFind = FindFirstFile(strPath.c_str(), &wfd);
	if ((hFind != INVALID_HANDLE_VALUE) && (wfd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
	{
		rValue = true;
	}
	FindClose(hFind);
	return rValue;
}


bool detectSide(char *output,const char *input, int mode)
{
	string filePath(input);
	string tmpSaveDir("D:/SideSaveImg/");
	Rect deeplabBbox;			// deeplabµÄ½á¹û
	vector<string> rowOutput;	// ÏäºÅÊÇºá×ÅµÄµÄ½á¹û
	string colOutput = "???? ??????? ????";			// ÏäºÅÊÇÊú×ÅµÄ½á¹û
	string tmp;
	strcpy(output, colOutput.c_str());
	Mat srcImg = cv::imread(filePath);

	if (!CheckFolderExist(tmpSaveDir)) {
		CreateDirectory(tmpSaveDir.c_str(), NULL);
	}
	//clearDir(tmpSaveDir);		// Çå¿ÕÒÔÏÂ´æ´¢ÖÐ¼äÊý¾ÝµÄÎÄ¼þ¼Ð
	vector<string> files;
	getAllFiles(tmpSaveDir, files);
	for (auto file : files) {
		DeleteFile(file.c_str());
	}

	initSocket();
	deeplab_request(filePath, deeplabBbox); //¸ù¾Ýdeeplab½á¹ûcropÔ­Ê¼Í¼Æ¬£¬ÏÈcropÊÇÎªÁË½µµÍÖ®ºóËã·¨µÄ¼ÆËãÁ¿



	string deeplabImgPath = filePath.substr(0, filePath.length() - 4) + "_dl.jpg";
	cv::imwrite(deeplabImgPath, srcImg(deeplabBbox));


	MserFilter mf(srcImg, deeplabBbox);
	mf.filter();

	if (mode == 0)	// AUTO
	{
		if (mf.rowClusters.size() != 0){
			vector<east_bndbox> east_boxes;
			vector<Rect> mser_boxes;
			vector<Rect> ctpn_boxes;
			east_request(deeplabImgPath, east_boxes);
			Mat deeplabimg = cv::imread(deeplabImgPath);
			Mat deeplabgrayimg;
			cv::cvtColor(deeplabimg, deeplabgrayimg, CV_BGR2GRAY);
			mser_boxes = getMSER(deeplabgrayimg, mserThres, mserMinArea, mserMaxArea);

			NumberDetector* detector = new NumberDetector(deeplabImgPath, ctpn_boxes, east_boxes, mser_boxes, deeplabBbox);

			string output_str = detector->detectRowNumber_front(tmpSaveDir);
			strcpy(output, output_str.c_str());
			return true;
		}
		else if (mf.colClusters.size() != 0)
		{
			for (int j = 0; j < mf.colResult.size(); j++)
			{
				Mat tmpImg(srcImg, mf.colResult[j]);
				imwrite(tmpSaveDir + to_string(j) + ".jpg", tmpImg);
			}
			alex_request(tmpSaveDir, tmp);

			if (tmp.size() == 0)
				return false;
			else {
				int companyLen = 3;
				for (; companyLen >= 0; companyLen--)
				{
					if (tmp[companyLen] < '0' || tmp[companyLen] > '9')
						break;
				}
				for (int i = 0; i <= companyLen; i++)
					colOutput[i] = tmp[i];

				int contLen = 0;
				if (mf.colClusters.size() == 1)
					contLen = tmp.size() - companyLen - 1;
				else
					contLen = tmp.size() - companyLen - 1 - mf.colClusters[1].size();
				for (int i = 0; i < contLen; i++)
					colOutput[i + 5] = tmp[companyLen + 1 + i];

				if (mf.colClusters.size() > 1)
				{
					for (int j = 0; j < 4 && j < mf.colClusters[1].size(); j++)
						colOutput[j + 13] = tmp[contLen + companyLen + j + 1];
				}

				strcpy(output, colOutput.c_str());
				return true;
			}

		}
		else {
			return false;
		}
	}
	if (mode == 1)	//FCOL_MODE
	{
		if (mf.colClusters.size() != 0)
		{
			for (int j = 0; j < mf.colResult.size(); j++)
			{
				Mat tmpImg(srcImg, mf.colResult[j]);
				imwrite(tmpSaveDir + to_string(j) + ".jpg", tmpImg);
			}
			alex_request(tmpSaveDir, tmp);

			if (tmp.size() == 0)
				return false;
			else {
				int companyLen = 3;
				for (; companyLen >= 0; companyLen--)
				{
					if (tmp[companyLen] < '0' || tmp[companyLen] > '9')
						break;
				}
				for (int i = 0; i <= companyLen; i++)
					colOutput[i] = tmp[i];

				int contLen = 0;
				if (mf.colClusters.size() == 1)
					contLen = tmp.size() - companyLen - 1;
				else
					contLen = tmp.size() - companyLen - 1 - mf.colClusters[1].size();
				for (int i = 0; i < contLen; i++)
					colOutput[i + 5] = tmp[companyLen + 1 + i];

				if (mf.colClusters.size() > 1)
				{
					for (int j = 0; j < 4 && j < mf.colClusters[1].size(); j++)
						colOutput[j + 13] = tmp[contLen + companyLen + j + 1];
				}

				strcpy(output, colOutput.c_str());
				return true;
			}
		}
		else
			return false;
	}
	if (mode == 2)	// FROW_MODE
	{
		vector<east_bndbox> east_boxes;
		vector<Rect> mser_boxes;
		vector<Rect> ctpn_boxes;
		east_request(deeplabImgPath, east_boxes);
		Mat deeplabimg = cv::imread(deeplabImgPath);
		Mat deeplabgrayimg;
		cv::cvtColor(deeplabimg, deeplabgrayimg, CV_BGR2GRAY);
		mser_boxes = getMSER(deeplabgrayimg, mserThres, mserMinArea, mserMaxArea);

		NumberDetector* detector = new NumberDetector(deeplabImgPath, ctpn_boxes, east_boxes, mser_boxes, deeplabBbox);

		string output_str = detector->detectRowNumber_front(tmpSaveDir);
		strcpy(output, output_str.c_str());
		return true;
	}

	return false;
}


bool detect(char* output, const char* input, int side) {
	string imgpath(input);
	//添加文件是否存在的判断代码
	//if(!fileExist(imgpath))
	Mat org_img = cv::imread(imgpath);
	

	vector<east_bndbox> east_boxes;
	vector<Rect> ctpn_boxes;
	Rect deeplab_box;
	vector<Rect> mser_boxes;

	string deeplab_imgpath(imgpath);
	deeplab_imgpath.replace(deeplab_imgpath.find(".jpg"), 4, "_dl.jpg");

	
	clock_t start, end;
	start = clock();

	initSocket();
	deeplab_request(imgpath, deeplab_box);

	//利用基于Deeplab的结果再做处理
	Mat deeplab_img = org_img(deeplab_box);
	cv::imwrite(deeplab_imgpath, deeplab_img);
	Mat deeplab_grayimg;
	cvtColor(deeplab_img, deeplab_grayimg, CV_BGR2GRAY);

	east_request(deeplab_imgpath, east_boxes);
	mser_boxes= getMSER(deeplab_grayimg, mserThres, mserMinArea, mserMaxArea);

	//添加路径是否存在的判断，若不存在则创建路径
	string chars_savepath = "D:/savedImgs/";
	if (!CheckFolderExist(chars_savepath)) {
		CreateDirectory(chars_savepath.c_str(), NULL);
	}

	vector<string> files;
	getAllFiles(chars_savepath, files);

	for (auto file : files) {
		DeleteFile(file.c_str());
	}

	string output_str;
	NumberDetector* detector = new NumberDetector(deeplab_imgpath, ctpn_boxes, east_boxes, mser_boxes, deeplab_box);
	MserFilter* filter = new MserFilter(org_img, deeplab_box);
	switch (side) {
	case AUTO:

		if (filter->judgeSide() == COL_MODE) {
			output_str = detector->detectVerticalNumber(chars_savepath);
		}
		else {
			output_str = detector->detectRowNumber_front(chars_savepath);
		}
		
		/*if (detector->judgeSide() == FCOL_MODE || detector->judgeMode_east()== FCOL_MODE){
			output_str=detector->detectVerticalNumber(chars_savepath);
		}
		else {
			output_str = detector->detectRowNumber_front(chars_savepath);
		}*/
		break;
	case FCOL_MODE:
		output_str = detector->detectVerticalNumber(chars_savepath);
		break;
	case FROW_MODE:
		output_str = detector->detectRowNumber_front(chars_savepath);
		break;
	default:
		return false;
	}

	end = clock();
	
	cout << "duration time: " << (end - start) / 1000.0 << "s" << endl;

	strcpy(output, output_str.c_str());
	return true;
}


#if 0
bool detect(char* output1,char* output2,const char* input1,const char* input2, int mode) {
	string img1 = input1;
	string img2 = input2;

	int flag = 0;
	ifstream fin1(img1);
	ifstream fin2(img2);

	if (!fin1) {
		cout << "file 1 does not exist." << endl;
		*output1 = '\0';
		*output2 = '\0';
		flag = 1;
	}
	if (!fin2) {
		cout << "file 2 does not exist." << endl;
		*output1 = '\0';
		*output2 = '\0';
		flag = 1;
	}
	if (flag == 1)
		return false;

	clock_t start, end;
	start = clock();

	vector<cv::Rect> ctpn_rects_1, ctpn_rects_2;
	vector<east_bndbox> east_boxes_1, east_boxes_2;
	Rect deeplabBox_1, deeplabBox_2;
	vector<Rect> mser_boxes_1, mser_boxes_2;

	initSocket();
	deeplab_request(img1, deeplabBox_1);
	east_request(img1, east_boxes_1);

	

	//initSocket();
	east_request(img2, east_boxes_2);
	deeplab_request(img2, deeplabBox_2);

	Mat orgimg1 = cv::imread(img1);
	Mat orgimg2 = cv::imread(img2);
	Mat grayimg1, grayimg2;
	cv::cvtColor(orgimg1, grayimg1, CV_BGR2GRAY);
	cv::cvtColor(orgimg2, grayimg2, CV_BGR2GRAY);



	Mat img1_deeplab = orgimg1(deeplabBox_1);
	Mat img2_deeplab = orgimg2(deeplabBox_2);

	char *img1_cstr = new char[img1.size() + 10];
	char *img2_cstr = new char[img2.size() + 10];
	strcpy(img1_cstr, img1.c_str());
	strcpy(img2_cstr, img2.c_str());
	string img1_dup = img1_cstr;
	string img2_dup = img2_cstr;
	img1_dup.replace(img1_dup.find(".jpg"), 4, "_dl.jpg");
	img2_dup.replace(img2_dup.find(".jpg"), 4, "_dl.jpg");
	cv::imwrite(img1_dup, img1_deeplab);
	cv::imwrite(img2_dup, img2_deeplab);

	mser_boxes_1 = getMSER(grayimg1, mserThres, mserMinArea, mserMaxArea);
	mser_boxes_2 = getMSER(grayimg2, mserThres, mserMinArea, mserMaxArea);


	NumberDetector *detector1 = new NumberDetector(img1, ctpn_rects_1, east_boxes_1, mser_boxes_1, deeplabBox_1);
	NumberDetector *detector2 = new NumberDetector(img2, ctpn_rects_2, east_boxes_2, mser_boxes_2, deeplabBox_2);
	
	string frontsavepath = "D:/frontSaveImg/";
	string backsavepath = "D:/savedimgs/";


	vector<string> files;
	getAllFiles(backsavepath, files);
	getAllFiles(frontsavepath, files);

	for (auto file : files) {
		DeleteFile(file.c_str());
	}

	//MserFilter sidejudge1(grayimg1, mser_boxes_1, deeplabBox_1);
	//MserFilter sidejudge2(grayimg2, mser_boxes_2, deeplabBox_2);

	string output_str_1 = "", output_str_2 = "";
	if (mode == 1) {
		output_str_1 = detector1->detectVerticalNumber(frontsavepath);
		output_str_2 = detectRegion(img2, east_boxes_2, deeplabBox_2, mser_boxes_2, detector2, start);
	}
	else {

		if (detector1->judgeSide() == 0) {
			output_str_1 = detectRegion(img1, east_boxes_1, deeplabBox_1, mser_boxes_1, detector1, start);
		}
		else {
			output_str_1 = detector1->detectVerticalNumber(frontsavepath);
		}

		if (detector2->judgeSide() == 0) {
			output_str_2 = detectRegion(img2, east_boxes_2, deeplabBox_2, mser_boxes_2, detector2, start);
		}
		else {
			output_str_2 = detector2->detectVerticalNumber(frontsavepath);
		}

	}
	//int ttmp;
	//cout << output_str_1 << endl;
	//cout << output_str_2 << endl;

	strcpy(output1, output_str_1.c_str());
	strcpy(output2, output_str_2.c_str());
	//output1 = output_str_1.c_str();
	//output2 = output_str_2.c_str();
	//std::cin >> ttmp;
	end = clock();

	return true;
}
#endif 

#if 0
//using namespace std;
string img1 = "D:\\col_img\\35.jpg";
string img2 = "D:\\testIMG\\253.jpg";

char* output1 = new char[100];
char* output2 = new char[100];

int main() {

	string testPath1= "D:/testIMG/";
	vector<string> files1;
	getAllFiles(testPath1, files1);
	/*string testPath2 = "D:/TESTimg/front";
	vector<string> files2;
	getAllFiles(testPath2, files2);*/

	for (int i = 0;i < files1.size();i++) {
	detectSide(output1, files1[i].c_str(), AUTO);
		//detect(output1, output2, img1.c_str(), img2.c_str(), 0);

	cout << output1 << endl;
	cv::destroyAllWindows();
		//cout << output2 << endl;
	}
	int tmp;
	scanf_s("%d", &tmp);
	return 0;
}
#endif

