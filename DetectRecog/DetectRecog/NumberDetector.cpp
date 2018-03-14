#include "NumberDetector.h"
#include "getMSER.hpp"
#include "utils.h"
#include "socketcpp\tcpClient.h"

using namespace std;
//#include <iostream>
NumberDetector::NumberDetector(string& imgpath, vector<Rect>& ctpn_boxes, vector<east_bndbox>& east_boxes, vector<Rect>& mser_boxes, Rect deeplab_box) {
	_org_img = cv::imread(imgpath);
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
		if (rect.height > rect.width && rect.height *1.0 /rect.width<10) {
			if(strict_mode==1 && rect.height *1.0 / rect.width>1.3)
				geometry_filter.push_back(rect);
			else
				geometry_filter.push_back(rect);
		}
	}
}

//0: row_mode
//1: col_mode
int NumberDetector::judgeSide() {
	vector<Rect> geometry_filter;
	geometryFilter(_mser_boxes, geometry_filter,1);

	vector<Rect> deeplab_filter;
	filterByDeeplab(geometry_filter, deeplab_filter);

	vector<vector<Rect>> clusters;
	boxClusteringVertical(deeplab_filter, clusters, 10, 10, 80);

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


	int col_flag = 0;
	for (auto cluster : clusters) {
		int n = cluster.size();
		Rect first_rect = cluster[0];
		Rect last_rect = cluster[n - 1];

		int length = last_rect.y + last_rect.height - first_rect.y;
		int container_height = _deeplab_box.height;

		float ratio = 1.0*length / container_height;
		//cout << ratio << endl;
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
 
string NumberDetector::detectVerticalNumber(long start, string savepath) {
	int mserSize = _mser_boxes.size();

	Mat drawimg;
	

	vector<Rect> geometry_filter;
	geometryFilter(_mser_boxes, geometry_filter,0);

	//vector<Rect> ctpn_filter;
	//filterByCTPNVertical(geometry_filter, ctpn_filter, 20, 20);

	vector<Rect> deeplab_filter;
	filterByDeeplab(geometry_filter, deeplab_filter);

	

	vector<vector<Rect>> clusters;
	boxClusteringVertical(deeplab_filter, clusters, 10,10,80);

#ifdef SHOW_FILTERED_MSER
	_org_img.copyTo(drawimg);
	for (auto rect : deeplab_filter) {
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
		if (newlen > maxlen) {
			maxlen = newlen;
			maxLenClusterID = i;
		}
	}
#ifdef SHOW_CLUSTERING
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


	float average_height=0;
	for (auto rect : numberVertical)
		average_height += rect.height;
	average_height=average_height / numberVertical.size();


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
	int nn = mergedRect.size();
	for (int i = 0; i < nn; i++) {
		string imgpath = savepath + std::to_string(i) + ".jpg";
		cv::imwrite(imgpath, _org_img(mergedRect[i]));
	}
	string recog_result;
	alex_request(savepath, recog_result);

	
	//显示检测识别所需的时间
	long end = clock();
	double dur = (double)(end - start);
	printf("duration Time:%f\n", (dur / CLOCKS_PER_SEC));

#if 0
	_org_img.copyTo(drawimg);
	for (int i = 0;i < mergedRect.size();i++) {
		Rect rect = mergedRect[i];
		scaleSingleRect(rect, 0.2, 0.1, _org_img.cols, _org_img.rows);
		cv::rectangle(drawimg, rect, cv::Scalar(0, 0, 255), 1);
		char digit[2];
		digit[0]=recog_result[i];
		digit[1] = '\0';
		cv::putText(drawimg, digit, cv::Point(rect.x+rect.width, rect.y+rect.height), 1, 1.6, cv::Scalar(0, 255, 255),2);
	}
	cv::imshow("recogn", drawimg);
	cv::waitKey(0);
#endif
	recog_result.insert(4, " ");
	std::transform(recog_result.begin(), recog_result.end(), recog_result.begin(), ::toupper);
	return recog_result;
};

int NumberDetector::getRectClusterLength(vector<Rect>& cluster, int direction) {
	int n = cluster.size()-1;
	if (direction == 0)
		return cluster[n].x + cluster[n].width - cluster[0].x;
	else
		return cluster[n].y + cluster[n].height - cluster[0].y;
}



//#define SHOW_FINAL
//#define SHOW_FILTER
//#define SHOW_CLUSTER
void NumberDetector::detectHorizontalNumber(long start, vector<vector<Rect>>& dst_rects, string savepath) {

	Mat draw_img;
	vector<Rect> east_rects;
	for (auto box : _east_boxes) {
		east_rects.push_back(eastbox2rect(box));
	}

	vector<Rect> filtered_rects;
	filterEASTbyDeeplab(east_rects, filtered_rects);

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