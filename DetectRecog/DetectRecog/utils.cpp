#include "utils.h"
#include "MyTypes.h"
//#include <dirent.h>
using namespace std;

bool sortByX(const Rect& lhs, const Rect& rhs) {
	if (lhs.x<rhs.x)
		return true;
	else return false;
}
bool sortByY(const Rect& lhs, const Rect& rhs) {
	if (lhs.y<rhs.y)
		return true;
	else return false;
}

bool sortClusterByY(const vector<Rect>& lhs, const vector<Rect>& rhs) {
	if (lhs[0].y<rhs[0].y)
		return true;
	else return false;
}
int isClusterExist(Rect rect, vector<vector<Rect>> clusters, vector<float> rectsx, vector<float> rectsy, vector<float> rectsheight, int xthres, int ythres, int heightthres) {
	int n = rectsx.size();
	float x = rect.x, y = rect.y + rect.height / 2.0;
	for (int i = 0; i<n; i++) {
		if (abs(x - rectsx[i])<xthres)
			if (abs(y - rectsy[i])<ythres)
				if (abs(rect.height - rectsheight[i])<heightthres)
					return i;
	}
	return -1;
}

void clusteringRects(vector<Rect> rects, vector<vector<Rect>>& clusters, int xthres, int ythres, int heightthres) {
	//vector<vector<Rect>> clusters;
	vector<float> rectsx, rectsy, rectsheight;
	sort(rects.begin(), rects.end(), sortByX);
	for (auto rect : rects) {

		int index = isClusterExist(rect, clusters, rectsx, rectsy, rectsheight, xthres, ythres, heightthres);
		if (index == -1) {
			vector<Rect> newcluster;
			newcluster.push_back(rect);
			clusters.push_back(newcluster);
			float newx = rect.x + rect.width, newy = rect.y + rect.height / 2.0;
			rectsx.push_back(newx);
			rectsy.push_back(newy);
			rectsheight.push_back(rect.height);
		}
		else {

			rectsx[index] = rect.x + rect.width;
			rectsy[index] = (clusters[index].size()*rectsy[index] + rect.y + rect.height / 2.0) / (clusters[index].size() + 1);
			rectsheight[index] = (clusters[index].size()*rectsheight[index] + rect.height) / (clusters[index].size() + 1);
			clusters[index].push_back(rect);
		}
	}

	sort(clusters.begin(), clusters.end(), sortClusterByY);
	//sortCluster(clusters);
	//sort(clusters.begin(),clusters.end(),sortByClusterSize);
	//return clusters;
}




void removeOverlap(vector<Rect>& rects,vector<Rect>& slim_rects) {
	sort(rects.begin(), rects.end(), sortByX);
	int n = rects.size();
	int* sign = new int[n] {1};
	for (int i = 0; i < n; i++)
		sign[i] = 1;
	for (int i = 0; i < n; i++) {
		Rect rect1 = rects[i];
		for (int j = i+1; j < n; j++) {
			Rect rect2 = rects[j];
			float x0 = max(rect1.x, rect2.x);
			float x1 = min(rect1.x + rect1.width, rect2.x + rect2.width);
			float y0 = max(rect1.y, rect2.y);
			float y1 = min(rect1.y + rect1.height, rect2.y + rect2.height);
			if (x1 <= x0 || y1 <= y0)
				break;
			float overlapArea = (x1 - x0)*(y1 - y0);
			float area1 = rect1.width*rect1.height;
			float area2 = rect2.width*rect2.height;
			if (overlapArea / area1 > 0.8 && overlapArea / area2 > 0.8) {
				sign[j] = 0;
			}
		}
	}
	for (int i = 0; i < n; i++) {
		if (sign[i] == 1)
			slim_rects.push_back(rects[i]);
	}
}

bool isOverlap(const Rect &rc1, const Rect &rc2)
{
	if (rc1.x + rc1.width  > rc2.x &&
		rc2.x + rc2.width  > rc1.x &&
		rc1.y + rc1.height > rc2.y &&
		rc2.y + rc2.height > rc1.y
		)
		return true;
	else
		return false;
}


void scaleSingleRect(Rect& rect, float scalex, float scaley, int maxx, int maxy) {
	int x = rect.x;
	int y = rect.y;
	int width = rect.width;
	int height = rect.height;
	x = x - ceil((float)width*scalex / 2.0);
	y = y - ceil((float)height*scaley / 2.0);
	width += width*scalex;
	height += height*scaley;
	rect.x = x < 0 ? 0 : x;
	rect.y = y < 0 ? 0 : y;
	rect.width = width;
	rect.height = height;
	if (x + width + 2>maxx)
		rect.width = maxx - x - 2;
	if (y + height + 2>maxy)
		rect.height = maxy - y - 2;
}


void mergeVerifyNum(vector<Rect>& rects) {
	int n = rects.size();
	if (n > 2) {
		Rect rect1 = rects[n - 1];
		Rect rect2 = rects[n - 2];
		rects.pop_back();
		rects.pop_back();
		Rect rect = rect1 | rect2;
		rects.push_back(rect);
	}
	

}

int clusterInCluster(vector<Rect> cluster) {
	vector<vector<Rect>> clusters;
	for(auto rect: cluster){
		int flag = 0;
		int n = clusters.size();
		for (int i = 0; i < n;i++) {
			int isOverlapped = 0;
			for (auto rect2 : clusters[i]) {
				if (isOverlap(rect, rect2)) {
					isOverlapped = 1;
					break;
				}
			}
			if (isOverlapped == 1) {
				clusters[i].push_back(rect);
				flag = 1;
				break;
			}
		}
		if (flag == 0) {
			vector<Rect> newcluster;
			newcluster.push_back(rect);
			clusters.push_back(newcluster);
		}

	}
	return clusters.size();
}

void findClusterAttr(vector<vector<Rect>>& clusters, vector<int> &attr) {
	for (auto cluster : clusters) {
		attr.push_back(clusterInCluster(cluster));
	}
}

//this function finds the most closest Rect in cluster
//by comparing x coordinate
int findClosestRectInCluster(vector<Rect> &rects, Rect rect) {
	int mindist = 999;
	int id=-1;
	int n = rects.size();
	for (int i = 0;i < rects.size();i++) {
		int dist = abs(rects[i].x - rect.x);
		if (dist < mindist) {
			id = i;
			mindist = dist;
		}
	}
	return id;
}


//this function calculate the distance betweent two Rect clusters
//which assumes the rects are sorted by x coordinates, from 0 to xmax
int verticalDisBetweenClusters(vector<Rect> rects1, vector<Rect> rects2) {
	vector<Rect> uprects, downrects;
	if (rects1[0].y < rects2[0].y) {
		uprects = rects1;
		downrects = rects2;
	}
	else {
		uprects = rects2;
		downrects = rects1;
	}

	int id = findClosestRectInCluster(uprects, downrects[0]);
	return downrects[0].y - (uprects[id].y + uprects[id].height);


}


//Code for finding company number
int findCompanyNumber(vector<vector<Rect>>& clusters, int mid, int yThres, int xThres) {
	for (int i = mid - 1;i >= 0;i--) {
		int ydist=verticalDisBetweenClusters(clusters[i],clusters[mid]);
		int xdist = abs(clusters[i][0].x - clusters[mid][0].x);
		if (ydist < yThres && xdist < xThres && ydist > 0) {
			return i;
		}
	}
	return -1;
}
int findContainerType(vector<vector<Rect>>& clusters, int mid, int yThres, int xThres) {
	for (int i = mid + 1;i < clusters.size();i++) {
		int ydist = verticalDisBetweenClusters(clusters[i], clusters[mid]);
		int xdist = abs(clusters[i][0].x - clusters[mid][0].x);
		if (ydist < yThres && xdist < xThres && ydist > 0) {
			return i;
		}
	}
	return -1;
}


void splitRectsByAverageHeight(vector<Rect>& src,vector<Rect>& dst, float avrageHeight) {
	vector<Rect> tmp;

	int top, bottom;
	int left, right;
	for (auto rect : src) {
		int n = tmp.size()-1;
		if (n < 0) {
			top = rect.y;
			bottom=rect.y + rect.height;
			left = rect.x;
			right = left + rect.width;
			tmp.push_back(rect);
			continue;
		}
		bool overlap=false;
		for (auto recttmp : tmp)
			overlap = overlap | isOverlap(rect, recttmp);
		if (overlap) {
			top = min(top, rect.y);
			bottom = max(bottom, rect.y + rect.height);
			left = min(left, rect.x);
			right = max(right, rect.x + rect.width);

			tmp.push_back(rect);
			continue;
		}
		else {
			int numChar = round((bottom - top) / avrageHeight);
			float height = (bottom - top)*1.0 / numChar;
			for (int i = 0; i < numChar; i++) {
				Rect splitrect(left, top+i*height, right - left, height);
				dst.push_back(splitrect);
			}
			/*if(abs(2*avrageHeight-bottom+top)<abs(avrageHeight - bottom + top)){
				Rect splitrect1(left, top, right - left, (bottom - top)*0.5);
				Rect splitrect2(left, top+ (bottom - top)*0.5, right - left, (bottom - top)*0.5);
				dst.push_back(splitrect1);
				dst.push_back(splitrect2);
			}
			else {
				Rect splitrect(left, top, right - left, bottom - top);
				dst.push_back(splitrect);
			}*/

			tmp.clear();
			top = rect.y;
			bottom = rect.y + rect.height;
			left = rect.x;
			right = left + rect.width;
			tmp.push_back(rect);
		}
		

	}
	if (tmp.size()>0 && abs(2 * avrageHeight - bottom + top)<abs(avrageHeight - bottom + top)) {
		Rect splitrect1(left, top, right - left, (bottom - top)*0.5);
		Rect splitrect2(left, top + (bottom - top)*0.5, right - left, (bottom - top)*0.5);
		dst.push_back(splitrect1);
		dst.push_back(splitrect2);
	}
	else {
		Rect splitrect(left, top, right - left, bottom - top);
		dst.push_back(splitrect);
	}
}

void mergeSameHorizon(vector<Rect>& src, vector<Rect>& dst) {
	Rect rect=src[0];
	for (int i = 1;i < src.size();i++) {
		float line = abs(rect.y + rect.height*0.5 - (src[i].y + src[i].height*0.5));
		if (line < 10)
			rect = rect | src[i];
		else {
			dst.push_back(rect);
			rect = src[i];
		}
	}
	dst.push_back(rect);
}


Rect eastbox2rect(east_bndbox box) {
	int x = std::min(box.x0,box.x3);
	int y = std::min(box.y0, box.y1);
	int width = std::max(box.x1, box.x2) - x;
	int height= std::max(box.y2, box.y3) - y;
	return Rect(x, y, width, height);
}

void showClustering(cv::Mat& drawimg,vector<vector<Rect>>& clusters) {
	//_org_img.copyTo(drawimg);
	for (auto rects : clusters) {
		cv::Scalar color(1.0*rand() / RAND_MAX * 255, 1.0*rand() / RAND_MAX * 255, 1.0*rand() / RAND_MAX * 255);
		for (auto rect : rects)
			cv::rectangle(drawimg, rect, color, 1);
	}
	cv::imshow("clusters", drawimg);
	cv::waitKey(0);
}


int verifyTable[26] = {10,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,34,35,36,37,38};

bool verifyContainerCode(string containerCode) {
	int sum = 0;
	for (int i = 0;i < 10;i++) {
		const char ch = containerCode.c_str()[i];
		if (ch >= 'A' && ch < 'a') {
			sum += verifyTable[ch - 'A']*pow(2,i);
		}
		else if (ch >= 'a') {
			sum += verifyTable[ch - 'a'] * pow(2, i);
		}
		else {
			sum += (ch - '0')*pow(2, i);
		}
	}
	int div = sum / 11;
	int newsum = div * 11;
	int verifyNum = sum - newsum;

	const char ch = containerCode.c_str()[10];

	if (verifyNum == (ch - '0')) {
		return true;
	}
	else {
		return false;
	}

}

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
void getAllFiles(string rootpath, vector<string>& files) {
	namespace fs = boost::filesystem;
	fs::path fullpath(rootpath, fs::native);
	if (!fs::exists(fullpath)) { return; }
	fs::directory_iterator end_iter;

	for (fs::directory_iterator iter(fullpath);iter != end_iter;iter++) {
		files.push_back(iter->path().string());
		//cout << iter->path().string() << endl;
	}
}

void findFirstAndLastGap(vector<Rect>& rects, int& first, int& last) {
	int n = rects.size();
	first = -1;
	last = -1;
	for (int i = 0; i < n-1; i++) {
		if (rects[i+1].y - rects[i].y-rects[i].height > rects[i].height*0.5) {
			first = i;
			break;
		}
	}
	for (int i = n-1; i >0; i--) {
		if (rects[i].y - rects[i-1].y- rects[i-1].height > rects[i-1].height*0.5) {
			last = i-1;
			break;
		}
	}
}
void fillGap(vector<Rect>& src, vector<Rect>& dst, int& first, int& last) {
	for (int i = 0; i < src.size()-1; i++) {
		bool isGap = (src[i+1].y - src[i].y - src[i].height) > src[i].height*0.5;
		if (i == first && isGap && i<4) {
			src[i].height = src[i + 1].y - src[i].y - src[i].height*0.5;
			dst.push_back(src[i]);
		}
		else if (i != last && isGap) {
			int numChar = round((src[i + 1].y - src[i].y)*1.0/ src[i].height);
			int x = src[i].x;
			int y = src[i].y;
			int width = src[i].width;
			int height = (src[i + 1].y - src[i].y)*1.0/numChar;
			for (int j = 0; j < numChar; j++) {
				dst.push_back(Rect(x, y + j*height, width, height));
			}
		}/*else if(i == last && isGap){
			if ((src[i + 1].y - src[i].y) * 1.0 / src[i].height > 2.5) {
				int numChar = round((src[i + 1].y - src[i].y- src[i].height*0.5)*1.0 / src[i].height);
				int x = src[i].x;
				int y = src[i].y;
				int width = src[i].width;
				int height = (src[i + 1].y - src[i].y)*1.0 / numChar;
				for (int j = 0; j < numChar; j++) {
					dst.push_back(Rect(x, y + j*height, width, height));
				}
			}
		}*/
		else {
			dst.push_back(src[i]);
		}
	}
	dst.push_back(src[src.size() - 1]);
}

float getAverageHeight(vector<Rect>& rects) {
	float average_height = 0;
	for (auto rect : rects)
		average_height += rect.height;
	average_height = average_height / rects.size();
	return average_height;
}
bool sortByLen(vector<cv::Rect> v1, vector<cv::Rect> v2)
{
	return v1.size() > v2.size();
}
void getClusterBbox(vector<cv::Rect> &cluster, cv::Rect &bbox)
{
	int minX = 10000, minY = 10000, maxX = 0, maxY = 0;
	for (int i = 0; i < cluster.size(); i++)
	{
		minX = min(minX, cluster[i].x);
		minY = min(minY, cluster[i].y);
		maxX = max(maxX, cluster[i].x + cluster[i].width);
		maxY = max(maxY, cluster[i].y + cluster[i].height);
	}
	bbox.x = minX;
	bbox.y = minY;
	bbox.width = maxX - minX;
	bbox.height = maxY - minY;
}

void getClusterAvgHW(vector<cv::Rect> &cluster, int &avgHeight, int &avgWidth)
{
	avgHeight = avgWidth = 0;
	for (int j = 0; j < cluster.size(); j++)
	{
		avgHeight += cluster[j].height;
		avgWidth += cluster[j].width;
	}
	avgHeight = avgHeight / cluster.size();
	avgWidth = avgWidth / cluster.size();
}
void getClusterMaxHW(vector<cv::Rect> &cluster, int &maxHeight, int &maxWidth)
{
	maxHeight = maxWidth = 0;
	for (int j = 0; j < cluster.size(); j++)
	{
		maxHeight = max(cluster[j].height, maxHeight);
		maxWidth = max(cluster[j].height, maxWidth);
	}
}
void addUnknownMark(string& str, int targetLen,int removePos) {
	int n = str.size();
	if (n < targetLen) {
		while (str.size() < targetLen) {
			str += '?';
		}
	}
	if (n > targetLen) {
		while (str.size() > targetLen) {
			str.erase(removePos, 1);
		}
	}
}
//void clearDir(string path)
//{
//	boost::filesystem::path rootPath(path, boost::filesystem::native);
//	boost::filesystem::directory_iterator end_iter;
//	for (boost::filesystem::directory_iterator iter(rootPath); iter != end_iter; iter++)
//		boost::filesystem::remove(iter->path());
//}
bool clusterSortByY(vector<cv::Rect>& c1, vector<cv::Rect>& c2)
{
	sort(c1.begin(), c1.end(), sortByY);
	sort(c2.begin(), c2.end(), sortByY);
	return c1[0].y < c2[0].y;
}
//static void
//list_directory(
//	const char *dirname)
//{
//	struct dirent **files;
//	int i;
//	int n;
//
//	/* Scan files in directory */
//	n = scandir(dirname, &files, NULL, alphasort);
//	if (n >= 0) {
//
//		/* Loop through file names */
//		for (i = 0; i < n; i++) {
//			struct dirent *ent;
//
//			/* Get pointer to file entry */
//			ent = files[i];
//
//			/* Output file name */
//			switch (ent->d_type) {
//			case DT_REG:
//				printf("%s\n", ent->d_name);
//				break;
//
//			case DT_DIR:
//				printf("%s/\n", ent->d_name);
//				break;
//
//			case DT_LNK:
//				printf("%s@\n", ent->d_name);
//				break;
//
//			default:
//				printf("%s*\n", ent->d_name);
//			}
//
//		}
//
//		/* Release file names */
//		for (i = 0; i < n; i++) {
//			free(files[i]);
//		}
//		free(files);
//
//	}
//	else {
//		printf("Cannot open directory %s\n", dirname);
//	}
//}
//void getAllFiles(const char* dirname, vector<string> filenames) {
//	struct dirent **files;
//	int n = scandir(dirname, &files, NULL, alphasort);
//}