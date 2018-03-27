#include "NumberDetector.h"
#include "getMSER.hpp"
#include "utils.h"
#include "socketcpp\tcpClient.h"
#include <regex>
#include "config.h"
using namespace std;
//#include <iostream>
NumberDetector::NumberDetector(string& imgpath, vector<Rect>& ctpn_boxes, vector<east_bndbox>& east_boxes, vector<Rect>& mser_boxes, Rect deeplab_box) {
	_org_img = cv::imread(imgpath);
	_imgpath = imgpath;
	_ctpn_boxes = ctpn_boxes;
	_east_boxes = east_boxes;
	_mser_boxes = mser_boxes;
	_deeplab_box = deeplab_box;
}


void NumberDetector::filterByDeeplab(vector<Rect>& input_boxes, vector<Rect>& output_boxes) {
	float middleline = _deeplab_box.x + _deeplab_box.width*0.5;
	for (auto rect : input_boxes)
		if (isOverlap(rect, _deeplab_box) && rect.x>middleline && rect.y+rect.height<_deeplab_box.y+ _deeplab_box.height && rect.y>_deeplab_box.y)
			output_boxes.push_back(rect);
}

void NumberDetector::filterByCTPNVertical(vector<Rect>& input_boxes, vector<Rect>& filtered_boxes, int distThres, int midlineThres) {
	for (auto rect : input_boxes) {
		for (auto box : _ctpn_boxes) {
			float midlineDist = abs(rect.x + rect.width / 2.0 - box.x - box.width / 2.0) < midlineThres;
			if (isOverlap(rect, box) && midlineDist<midlineThres) {
				filtered_boxes.push_back(rect);
				break;
			}
			int dist = abs(rect.y+rect.height/2.0-box.y-box.height/2.0)- rect.height / 2.0- box.height / 2.0;
			if (dist<distThres && midlineDist<midlineThres) {
				filtered_boxes.push_back(rect);
				break;
			}
		}
	}
}

void NumberDetector::geometryFilter(vector<Rect>& input_boxes, vector<Rect>& geometry_filter, int strict_mode) {
	for (auto rect : input_boxes) {
		if (rect.height > rect.width && rect.height *1.0 /rect.width<10 && rect.x>0.5*_org_img.cols
			&& abs(_org_img.cols-rect.x-rect.height)>20) {
			if(strict_mode==1 && rect.height *1.0 / rect.width>1.3)
				geometry_filter.push_back(rect);
			else
				geometry_filter.push_back(rect);
		}
	}
}

//2: row_mode
//1: col_mode
int NumberDetector::judgeSide() {
	vector<Rect> geometry_filter;
	geometryFilter(_mser_boxes, geometry_filter,1);

	//vector<Rect> deeplab_filter;
	//filterByDeeplab(geometry_filter, deeplab_filter);

	vector<vector<Rect>> clusters;
	boxClusteringVertical(geometry_filter, clusters, 10, 10, 80);

	Mat drawimg;
#if 0
	_org_img.copyTo(drawimg);
	for (auto rects : clusters) {
		cv::Scalar color(1.0*rand() / RAND_MAX * 255, 1.0*rand() / RAND_MAX * 255, 1.0*rand() / RAND_MAX * 255);
		for (auto rect : rects)
			cv::rectangle(drawimg, rect, color, 1);
	}
	cv::imshow("clusters", drawimg);
	cv::waitKey(0);
#endif


	int col_flag = 2;
	for (auto cluster : clusters) {
		int n = cluster.size();
		Rect first_rect = cluster[0];
		Rect last_rect = cluster[n - 1];

		int length = last_rect.y + last_rect.height - first_rect.y;
		int container_height = _deeplab_box.height;

		float ratio = 1.0*length / container_height;
		cout << ratio << endl;
		if (ratio> 0.43)
			col_flag = 1;
	}
	return col_flag;
}

//聚类代码存在的问题
//由于阈值设定的问题，可能会因为纳入无关区域而使得
//vector<float> midlines;
//vector<float> heights;
//vector<int> downBorders;
//的值发生变化
//从而使得后面正确的区域无法被纳入聚类

//聚类考虑的依据有：
//高度差在阈值范围内
//水平方向的坐标差在阈值范围内（以rect的垂直中线为准）
//如果两rect重叠，但重叠比例不大的，不被当作是同一cluster
//相互之间在y轴方向的距离
void NumberDetector::boxClusteringVertical(vector<Rect>& input_boxes, 
	vector<vector<Rect>>& clusters, 
	int midlineThres, 
	int heightThres, 
	int borderThres) {

	sort(input_boxes.begin(),input_boxes.end(),sortByY);
	vector<float> midlines;
	vector<float> midlineThreses;
	vector<float> heights;
	vector<float> widths;
	vector<int> downBorders;
	int clusterNum = 0;
	for (auto rect : input_boxes) {
		int clusteringFlag = -1;
		for (int i = 0;i <clusterNum;i++) {
			Rect lastrect = clusters[i][clusters[i].size() - 1];

			
			//以重叠面积占比来判断是否为同一聚类
			if (isOverlap(rect, lastrect)) {
				float x0 = std::max(rect.x, lastrect.x);
				float x1 = std::min(rect.x + rect.width, lastrect.x + lastrect.width);
				float y0 = std::max(rect.y, lastrect.y);
				float y1 = std::min(rect.y + rect.height, lastrect.y + lastrect.height);
				float overlaparea = (x1 - x0)*(y1 - y0);
				float area1 = rect.width*rect.height;
				float area2= lastrect.width*lastrect.height;
				float ratio1 = overlaparea / area1;
				float ratio2 = overlaparea / area2;
				if (ratio1 > 0.1 && ratio1<0.9 && ratio2>0.1 && ratio2 < 0.9)
					continue;
			}

			//分别利用中线位置，高度，距离判断是否聚类
			float midline = rect.x + rect.width / 2.0;
			if( abs(midline-midlines[i])<midlineThreses[i] && 
				abs(rect.height-heights[i])<heightThres &&
				abs(rect.y - downBorders[i])<borderThres) {
				clusteringFlag = i;
				break;
			}
		}
		if (clusteringFlag == -1) {
			vector<Rect> rects;
			rects.push_back(rect);
			clusterNum++;
			midlines.push_back(rect.x+rect.width/2.0);
			heights.push_back(rect.height);
			widths.push_back(rect.width);
			downBorders.push_back(rect.y + rect.height);
			midlineThreses.push_back(rect.width*0.7);
			clusters.push_back(rects);
		}
		else {
			int clusterSize = clusters[clusteringFlag].size();
			//midlines[clusteringFlag] = (midlines[clusteringFlag]* clusterSize + rect.x + rect.width / 2.0)/(clusterSize+1.0);
			midlines[clusteringFlag] =   rect.x + rect.width / 2.0;
			//midlineThreses[clusteringFlag] = rect.width*0.9;
			
			widths[clusteringFlag]= (widths[clusteringFlag] * clusterSize + rect.width) / (clusterSize + 1.0);
			midlineThreses[clusteringFlag] = widths[clusteringFlag]*0.7;
			//widths[clusteringFlag] = rect.width;
			heights[clusteringFlag]= (heights[clusteringFlag] * clusterSize + rect.height) / (clusterSize + 1.0);
			downBorders[clusteringFlag] = std::max(downBorders[clusteringFlag], rect.y+rect.height);
			clusters[clusteringFlag].push_back(rect);
		}

	}
}
 
string NumberDetector::detectVerticalNumber(string savepath) {
	int mserSize = _mser_boxes.size();

	Mat drawimg;
	

	vector<Rect> geometry_filter;
	geometryFilter(_mser_boxes, geometry_filter,0);

	//vector<Rect> ctpn_filter;
	//filterByCTPNVertical(geometry_filter, ctpn_filter, 20, 20);

	//vector<Rect> deeplab_filter;
	//filterByDeeplab(geometry_filter, deeplab_filter);

	

	vector<vector<Rect>> clusters;
	boxClusteringVertical(geometry_filter, clusters, 10,10,80);

#if 0
	_org_img.copyTo(drawimg);
	for (auto rect : geometry_filter) {
		cv::rectangle(drawimg, rect, (0, 0, 255), 1);
	}
	cv::imshow("mser", drawimg);
	cv::waitKey(0);
#endif

	//the cluster with longest length in vertical direction
	//is chosen as the conatiner code region
	int n = clusters.size();
	int maxlen = 0;
	int maxLenClusterID;
	for (int i = 0;i < n;i++) {
		int newlen = getRectClusterLength(clusters[i], 1);
		if (newlen > maxlen && getAverageHeight(clusters[i])>20) {
			maxlen = newlen;
			maxLenClusterID = i;
		}
	}
#if 0
	_org_img.copyTo(drawimg);
	for (auto rects : clusters) {
		cv::Scalar color(1.0*rand()/RAND_MAX*255, 1.0*rand()/ RAND_MAX*255, 1.0*rand()/ RAND_MAX*255);
		for(auto rect: rects)
			cv::rectangle(drawimg, rect, color, 1);
	}
	cv::imshow("clusters", drawimg);
	cv::waitKey(0);
#endif
	vector<Rect> numberVertical = clusters[maxLenClusterID];

#ifdef SHOW_CLUSTERING
	_org_img.copyTo(drawimg);
	for (auto rect : numberVertical) {
		cv::rectangle(drawimg, rect, (0, 0, 255), 1);
	}
	cv::imshow("cluster", drawimg);
	cv::waitKey(0);
	
	
#endif
	
	//vector<Rect> slimRect;
	//removeOverlap(numberVertical, slimRect);


	float average_height=getAverageHeight(numberVertical);
	/*for (auto rect : numberVertical)
		average_height += rect.height;
	average_height=average_height / numberVertical.size();*/


	//利用平均高度信息进行分割，分割出单个字符
	vector<Rect> splitRect;
	splitRectsByAverageHeight(numberVertical, splitRect, average_height);


	//如果有相同水平线上的rect出现，则将其合并
	vector<Rect> mergedRect;
	mergeSameHorizon(splitRect, mergedRect);

#ifdef SHOW_FINAL
	_org_img.copyTo(drawimg);
	for (auto rect : mergedRect) {
		cv::rectangle(drawimg, rect, (0,0,255), 1);
	}
	cv::imshow("vertical numbers", drawimg);
	cv::waitKey(0);
#endif
	//string savepath = "D:/frontSaveImg/";

	int firstGap, lastGap;
	findFirstAndLastGap(mergedRect, firstGap, lastGap);
	vector<Rect> fillGapRects;
	fillGap(mergedRect,fillGapRects,firstGap,lastGap);
	int nn = fillGapRects.size();
	for (int i = 0; i < nn; i++) {
		Rect rect = fillGapRects[i];
		scaleSingleRect(rect, 0.3, 0.1, _org_img.cols, _org_img.rows);
		
		string imgpath = savepath + std::to_string(i) + ".jpg";
		cv::imwrite(imgpath, _org_img(rect));
	}
	string recog_result;
	alex_request(savepath, recog_result);

	


#if 0
	_org_img.copyTo(drawimg);
	for (int i = 0;i < fillGapRects.size();i++) {
		
		Rect rect = fillGapRects[i];
		scaleSingleRect(rect, 0.2, 0.1, _org_img.cols, _org_img.rows);
		cv::rectangle(drawimg, rect, cv::Scalar(0, 0, 255), 1);
		char digit[2];
		digit[0]=recog_result[i];
		digit[1] = '\0';
		cv::putText(drawimg, digit, cv::Point(rect.x+rect.width, rect.y+rect.height), 1, 1.6, cv::Scalar(0, 255, 255),2);
	}
	cv::imshow(_imgpath, drawimg);
	cv::waitKey(0);
#endif
	//recog_result.insert(4, " ");
	std::transform(recog_result.begin(), recog_result.end(), recog_result.begin(), ::tolower);
	string company_code;
	string container_code;
	int flag = -1;
	for (int i = 0; i < recog_result.size(); i++) {
		if (recog_result[i] >= '0'&& recog_result[i] <= '9') {
			flag = i;
			company_code = string(recog_result, 0, i);
			container_code = string(recog_result, i, recog_result.size()-i);
			break;
		}
	}
	string final_str;
	if (flag <= 0) {
		company_code = "????";
		final_str = company_code + recog_result;
	}
	else {
		string new_company_str;
		conNuMostSimMatch(company_code, new_company_str);
		final_str = new_company_str + container_code;
		
	}
	std::transform(final_str.begin(), final_str.end(), final_str.begin(), ::toupper);
	addUnknownMark(final_str, 11, 11);
	final_str.insert(4, " ");
	return final_str;
};

int NumberDetector::getRectClusterLength(vector<Rect>& cluster, int direction) {
	int n = cluster.size()-1;
	if (direction == 0)
		return cluster[n].x + cluster[n].width - cluster[0].x;
	else
		return cluster[n].y + cluster[n].height - cluster[0].y;
}

void NumberDetector::filterEastByPosition(vector<Rect>& src, vector<Rect>& dst) {
	float midline_x = _org_img.cols*0.5;
	for (auto rect : src) {
		if (rect.x > midline_x)
			dst.push_back(rect);
	}
}

//#define SHOW_FINAL
//#define SHOW_FILTER
//#define SHOW_CLUSTER
void NumberDetector::detectHorizontalNumber(vector<vector<Rect>>& dst_rects, string savepath) {

	Mat draw_img;
	vector<Rect> east_rects;
	for (auto box : _east_boxes) {
		east_rects.push_back(eastbox2rect(box));
	}

	//sort(east_rects.begin(), east_rects.end(), sortByY);

	vector<Rect> filtered_rects;
	filterEastByPosition(east_rects, filtered_rects);

#ifdef SHOW_FILTER
	_org_img.copyTo(draw_img);
	for (auto rect : filtered_rects)
		cv::rectangle(draw_img, rect, cv::Scalar(0, 0, 255), 1);
	cv::imshow("filter", draw_img);
	cv::waitKey(0);
#endif


	vector<vector<Rect>> clusters;
	simpleClusterByHorizon(filtered_rects, clusters, 20);

#ifdef SHOW_CLUSTER
	_org_img.copyTo(draw_img);
	showClustering(draw_img, clusters);
#endif


	for (int i = 0;i < clusters.size();i++) {
		sort(clusters[i].begin(), clusters[i].end(), sortByX);
	}

	//确定集装箱箱号的位置
	vector<Rect> final_rects;
	int size = clusters.size();
	if (size >= 3) {
		int rownum;
		int mainrow;
		if (getRectClusterLength(clusters[0], 0) < getRectClusterLength(clusters[1], 0)) {
			rownum = 3;
			mainrow = 1;
		}
		else {
			rownum = 2;
			mainrow = 0;
		}


		//后面改成将lastRect和verifynumber合并
		//并加入 对east的选框调整的代码
		Rect lastRect = clusters[mainrow][clusters[mainrow].size() - 1];
		Rect verifyNumber;
		if(findVerifyNumber(lastRect, verifyNumber, 20, 20)==1)
			clusters[mainrow].push_back(verifyNumber);

		if (rownum == 2) {
			vector<Rect> empty_rects;
			dst_rects.push_back(empty_rects);
			dst_rects.push_back(clusters[mainrow]);
			dst_rects.push_back(clusters[mainrow+1]);
		}
		else {
			dst_rects.push_back(clusters[mainrow-1]);
			dst_rects.push_back(clusters[mainrow]);
			dst_rects.push_back(clusters[mainrow + 1]);
		}
		for (int i = 0;i < rownum;i++) {
			for (auto rect : clusters[i])
				final_rects.push_back(rect);
		}
	}
	else if(size==2){
		int mainrow;
		if (getRectClusterLength(clusters[0], 0) < getRectClusterLength(clusters[1], 0)) {
			mainrow = 1;
		}
		else {
			mainrow = 0;
		}

		Rect lastRect = clusters[mainrow][clusters[mainrow].size() - 1];
		Rect verifyNumber;
		if (findVerifyNumber(lastRect, verifyNumber, 20, 20) == 1)
			clusters[mainrow].push_back(verifyNumber);

		if (mainrow == 1) {
			vector<Rect> empty_rects;

			dst_rects.push_back(clusters[0]);
			dst_rects.push_back(clusters[1]);
			dst_rects.push_back(empty_rects);
		}
		else {
			vector<Rect> empty_rects;
			dst_rects.push_back(empty_rects);
			dst_rects.push_back(clusters[0]);
			dst_rects.push_back(clusters[1]);
		}
		for (auto rects : clusters) {
			for (auto rect : rects) {
				final_rects.push_back(rect);
			}
		}
	}
	else if(size==1){
		vector<Rect> empty_rects;

		dst_rects.push_back(empty_rects);
		dst_rects.push_back(clusters[0]);
		dst_rects.push_back(empty_rects);

		for (auto rects : clusters) {
			for (auto rect : rects) {
				final_rects.push_back(rect);
			}
		}
	}
	else {
		vector<Rect> empty_rects;

		dst_rects.push_back(empty_rects);
		dst_rects.push_back(empty_rects);
		dst_rects.push_back(empty_rects);
	}
	
#ifdef SHOW_FINAL
	_org_img.copyTo(draw_img);
	for (auto rect : final_rects)
		cv::rectangle(draw_img, rect, cv::Scalar(0, 0, 255), 1);
	cv::imshow("final",draw_img);
	cv::waitKey(0);
#endif
}
void NumberDetector::simpleClusterByHorizon(vector<Rect>& src, vector<vector<Rect>>& clusters, int thres){
	vector<float> horizon_line;
	for (auto rect : src) {
		int clusternum = clusters.size();
		int flag = -1;
		float rectline = rect.y + rect.height*0.5;
		for (int i = 0;i < clusternum;i++) {
			if (abs(rectline - horizon_line[i]) < thres) {
				flag = i;
				break;
			}
		}
		if (flag == -1) {
			vector<Rect> newrects;
			newrects.push_back(rect);

			horizon_line.push_back(rectline);
			clusters.push_back(newrects);
		}
		else {
			int size = clusters[flag].size();
			horizon_line[flag] = (horizon_line[flag] * size + rectline) / (size + 1);
			clusters[flag].push_back(rect);
		}
	}
	sort(clusters.begin(), clusters.end(), sortClusterByY);
}
void NumberDetector::filterEASTbyDeeplab(vector<Rect>& src, vector<Rect>& dst) {
	sort(src.begin(), src.end(), sortByY);
	float deeplab_midline= _deeplab_box.x + _deeplab_box.width*0.5;
	for (auto rect : src) {
		if (rect.x > deeplab_midline && rect.y > _deeplab_box.y && rect.x+rect.width<_deeplab_box.x+_deeplab_box.width)
			dst.push_back(rect);
	}
}

int NumberDetector::findVerifyNumber(Rect& rect, Rect& verifyRect, int midthres,int heightThres) {
	float midline = rect.y + rect.height*0.5;
	float xdist = rect.height*2;
	float xmax = rect.x + rect.width;
	float height = rect.height;
	float xborder = _deeplab_box.x + _deeplab_box.width;

	int success=-1;

	int maxarea = 0;

	for (auto mserrect : _mser_boxes) {
		if (isOverlap(mserrect, rect))
			continue;
		float msermid = mserrect.y + mserrect.height*0.5;
		float dist = mserrect.x - xmax;
		float mserxmax = mserrect.x + mserrect.width;

		
		if (dist>0 && abs(msermid - midline) < midthres && dist < xdist && abs(height - mserrect.height) < heightThres&& mserxmax<xborder && mserrect.height>mserrect.width) {
			int area = mserrect.width*mserrect.height;
			if (area > maxarea) {
				maxarea = area;
				verifyRect = mserrect;
			}
			success = 1;
		}
	}
	return success;
}


//如果出现高宽比大于3的east box，则判定为col mode
int NumberDetector::judgeMode_east() {
	for (auto box : _east_boxes) {
		float width = box.x1 - box.x0;
		float height = box.y3 - box.y0;
		if (height / width > 4)
			return 1;
	}
	return 2;
}


//利用east的结果对mser进行过滤
void NumberDetector::filterByEast(vector<Rect>& mser_boxes, vector<east_bndbox>& east_boxes, vector<Rect>& filtered_boxes,float filterHeightThres,float filterYThres, float filterDistThres) {
	for (auto box : mser_boxes) {
		for (auto east_bndbox : east_boxes) {
			float avgheight = (east_bndbox.y3 - east_bndbox.y0 + east_bndbox.y2 - east_bndbox.y1) / 2.0;
			float avgy = (east_bndbox.y0 + east_bndbox.y3 + east_bndbox.y1 + east_bndbox.y2) / 4.0;
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
			if (whratio > 1.3 && box.width > 3 && abs(box.height - avgheight) < filterHeightThres && abs(box.y + box.height / 2.0 - avgy) < filterYThres && dist < filterDistThres && box.height>box.width) {
				filtered_boxes.push_back(box);
			}
		}
	}
};

string NumberDetector::detectRowNumber_front(string imgsavepath) {
	
	Mat img_gray;
	cvtColor(_org_img, img_gray, CV_BGR2GRAY);
	
	Mat showimg, showimg2, showimg3, showimg4, showimg5;
	_org_img.copyTo(showimg);
	_org_img.copyTo(showimg2);
	_org_img.copyTo(showimg3);
	_org_img.copyTo(showimg4);
	_org_img.copyTo(showimg5);




	vector<Rect> filter_boxes;
	filterByEast(_mser_boxes, _east_boxes, filter_boxes, eastFilterHeightThres, eastFilterYThres,eastFilterDistThres);

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
	_org_img.copyTo(draw_img);
	for (auto box : _east_boxes) {
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
		if (attrs[i] > 5) {
			mid = i;
			break;
		}
	}



	vector<Rect> midboxes, upboxes, downboxes;
	if (mid != -1 && clusters[mid][0].y<_org_img.rows*0.5) {
		//find "company code" and "container type" 
		up = findCompanyNumber(clusters, mid, 30, 200);
		down = findContainerType(clusters, mid, 100, 200);

		locateNumber(_east_boxes, clusters[mid], midboxes);

		//修正EAST检测框的边界
		eastBoderRefine(midboxes, slim_rects);



		if (up != -1) {

			locateNumber(_east_boxes, clusters[up], upboxes);

			if (midboxes.size() == 1) {
				Rect rect = midboxes[0];
				midboxes.pop_back();

				Rect rect1(rect.x, rect.y, rect.width * 4 / 7.0, rect.height);
				Rect rect2(rect.x + rect.width * 5 / 7.0, rect.y, rect.width * 2 / 7.0, rect.height);
				midboxes.push_back(rect1);
				midboxes.push_back(rect2);
			}

		}
		if (down != -1)
			locateNumber(_east_boxes, clusters[down], downboxes);

		Rect verifyDigit;
		int verifyid = 0;


		Rect rightMostRect = clusters[mid][clusters[mid].size() - 1];
		int leftBorder = rightMostRect.x + rightMostRect.width;


		sort(midboxes.begin(), midboxes.end(), sortByX);
		vector<Rect> newmidBoxes;
		if (up == -1) {

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
			for (int i = 0; i < nn; i++)
				newmidBoxes.push_back(midboxes[i]);
			newmidBoxes.push_back(rect);
			midboxes = newmidBoxes;
		}

	}
	else {
		vector<vector<Rect>> dst_rects;
		detectHorizontalNumber(dst_rects, "");
		upboxes = dst_rects[0];
		midboxes = dst_rects[1];
		downboxes = dst_rects[2];

		if (upboxes.size() > 0)
			up = 1;
		if (midboxes.size() > 0)
			mid = 1;
		if (downboxes.size() > 0)
			down = 1;
	}


	//处理EAST漏检的情况
	if (up != -1 && midboxes.size()>=2) {
		float dist = midboxes[1].x - midboxes[0].x - midboxes[0].width;
		if (dist > 20)
			midboxes[0].width += 20;
	}



	int count = 0;
	for (auto rect : upboxes) {
		Mat img = _org_img(rect);
		string path = imgsavepath + to_string(count) + ".jpg";
		cv::imwrite(path, img);
		count++;
	}

	//mergeVerifyNum(midboxes);
	for (auto rect : midboxes) {


		scaleSingleRect(rect, 0.1, 0.1, _org_img.cols, _org_img.rows);
		Mat img = _org_img(rect);
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
		scaleSingleRect(rect, 0.2, 0.1, _org_img.cols, _org_img.rows);
		Mat img = _org_img(rect);
		string path = imgsavepath + to_string(count) + ".jpg";
		cv::imwrite(path, img);
		count++;
	}

	vector<string> strs;
	crnn_request(imgsavepath, strs);

	count = 0;

	string output_str = "";

	for (auto rect : upboxes) {
		rectangle(showimg4, rect, green, 2);


		string chechdatabase;
		conNuMostSimMatch(strs[count], chechdatabase);
		output_str = output_str + chechdatabase;

		cv::putText(showimg4, chechdatabase, cv::Point(rect.x, rect.y), 1.9, 1.7, red, 2);
		count++;
	}
	for (int i = 0; i < midboxes.size(); i++) {
		Rect rect = midboxes[i];
		rectangle(showimg4, rect, green, 2);
		if (up == -1 && i == 0) {
			char firstchar = strs[count].c_str()[0];
			string restchars = strs[count].c_str() + 1;
			//string newstr;

			string checkdatabase;
			conNuMostSimMatch(strs[count], checkdatabase);
			
			
			if (checkdatabase.compare("") == 0) {
				output_str = output_str + strs[count];
				cv::putText(showimg4, strs[count], cv::Point(rect.x, rect.y), 1.9, 1.7, red, 2);
				
			}
			else {
				output_str = output_str + checkdatabase;
				cv::putText(showimg4, checkdatabase, cv::Point(rect.x, rect.y), 1.9, 1.7, red, 2);
			}

		}
		else if (midboxes[i].width<midboxes[i].height) {
			addUnknownMark(output_str, 10, 8);

			int lastid = strs[count].size();
			string lastdigit = strs[count].c_str() + lastid - 1;
			if (lastdigit[0] == 'z')
				lastdigit = '2';
			else if (lastdigit[0] == 'o' || lastdigit[0] == 'c')
				lastdigit = '0';
			else if (lastdigit[0] == 'b')
				lastdigit = '3';
			else if (lastdigit[0] == 'd' || lastdigit[0] == 'q' || lastdigit[0] == 'e' || lastdigit[0] == 'i')
				lastdigit = '1';

			output_str = output_str + lastdigit;
			cv::putText(showimg4, lastdigit, cv::Point(rect.x, rect.y), 1.9, 1.7, red, 2);
		}
		else if (up != -1 && midboxes.size() == 1 && strs[count].size()>7) {
			string contNum(strs[count]);
			contNum.erase(4, 1);
			cv::putText(showimg4, contNum, cv::Point(rect.x, rect.y), 1.9, 1.7, red, 2);
			output_str = output_str + contNum;
		}
		else {
			cv::putText(showimg4, strs[count], cv::Point(rect.x, rect.y), 1.9, 1.7, red, 2);
			output_str = output_str + strs[count];
		}

		count++;
	}
	addUnknownMark(output_str, 11, 8);
	for (auto rect : downboxes) {
		rectangle(showimg4, rect, green, 2);

		string findDatabase;

		mostSimMatch(strs[count], findDatabase);
		cv::putText(showimg4, findDatabase, cv::Point(rect.x, rect.y), 1.9, 1.7, red, 2);

		output_str = output_str + findDatabase;
		count++;
	}
	addUnknownMark(output_str, 15, 8);
	output_str.insert(11, " ");
	output_str.insert(4, " ");
	//cout << output_str << endl;
#if 0
	cv::imshow(_imgpath, showimg4);
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
	std::transform(output_str.begin(), output_str.end(), output_str.begin(), ::toupper);
	return output_str;
}

void NumberDetector::locateNumber(vector<east_bndbox>& east_boxes, vector<Rect> cluster, vector<Rect>& boxes) {
	int eastnum = east_boxes.size();
	for (int i = 0; i < eastnum; i++) {
		east_bndbox box = east_boxes[i];
		int x0 = std::min(box.x0, box.x3);
		int x1 = std::max(box.x1, box.x2);
		int y0 = std::min(box.y0, box.y1);
		int y1 = std::max(box.y2, box.y3);
		if (x0 < 0)
			x0 = 0;
		if (y0 < 0)
			y0 = 0;
		Rect rect0(x0, y0, x1 - x0, y1 - y0);
		for (auto rect1 : cluster) {
			if (isOverlap(rect0, rect1)) {
				int width = std::min(rect0.x + rect0.width, rect1.x + rect1.width) - std::max(rect0.x, rect1.x);
				int height = std::min(rect0.y + rect0.height, rect1.y + rect1.height) - std::max(rect0.y, rect1.y);
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
void NumberDetector::eastBoderRefine(vector<Rect>& boxes, vector<Rect>& filtered_mser) {
	for (int i = 0; i < boxes.size(); i++) {
		int rightmost_border = 0;
		for (auto rect : filtered_mser) {
			if (isOverlap(boxes[i], rect)) {
				int newborder = rect.x + rect.width;
				if (newborder > boxes[i].x + boxes[i].width && newborder > rightmost_border) {
					rightmost_border = newborder;
				}
			}
		}
		if (rightmost_border != 0)
			boxes[i].width = rightmost_border - boxes[i].x;
	}


}