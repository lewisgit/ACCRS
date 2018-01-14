#include "recallratio.h"



void readXml(string filename, vector<Rect>& rects) {
	xml_document<> doc;
	rapidxml::file<> xmlfile(filename.c_str());
	doc.parse<0>(xmlfile.data());
	xml_node<> *boxnode = doc.first_node("annotation");
	boxnode = boxnode->first_node("object");
	for (xml_node<> *rectinfo = boxnode->first_node("bndbox"); boxnode; boxnode = boxnode->next_sibling()) {
		
		int xmin = atoi(rectinfo->first_node("xmin")->value());
		int xmax = atoi(rectinfo->first_node("xmax")->value());
		int ymin = atoi(rectinfo->first_node("ymin")->value());
		int ymax = atoi(rectinfo->first_node("ymax")->value());
		Rect rect(xmin,ymin,xmax-xmin,ymax-ymin);
		rects.push_back(rect);
	}

}


int overlapArea(Rect rect1,Rect rect2) {
	int x1 = rect1.x, x2 = rect2.x;
	int y1 = rect1.y, y2 = rect2.y;
	int w1 = rect1.width, w2 = rect2.width;
	int h1 = rect1.height, h2 = rect2.height;
	int width = min(x1+w1,x2+w2) - max(x1,x2);
	int height= min(y1 + h1, y2 + h2) - max(y1, y2);
	if (width > 0 && height > 0) {
		return width*height;
	}
	else {
		return 0;
	}
}

void calRatio1(vector<Rect> rects, string filename, float* ratio1, float* ratio2, int* num ) {
	vector<Rect> groundtruth;
	readXml(filename, groundtruth);
	int n = rects.size();
	*num += groundtruth.size();
	for (int i = 0; i < n; i++) {
		int area1 = rects[i].width*rects[i].height;
		int area2 = groundtruth[i].width*groundtruth[i].height;
		int overlaparea = overlapArea(rects[i], groundtruth[i]);
		*ratio1 += 1.0*overlaparea / area1;
		*ratio2 += 1.0*overlaparea / area2;
	}

}
void calRatio2(vector<vector<Rect>> rects, string filename, float* ratio1, float* ratio2,int* num) {
	vector<Rect> groundtruth;
	readXml(filename, groundtruth);
	int n = rects.size();
	*num += groundtruth.size();
	for (int i = 0; i < n; i++) {
		int area1 = 0, overlaparea = 0;
		for (int j = 0; j < rects[i].size(); j++) {
			area1 += rects[i][j].width*rects[i][j].height;
			overlaparea += overlapArea(rects[i][j], groundtruth[i]);
		}
		int area2= groundtruth[i].width*groundtruth[i].height;
		*ratio1 += 1.0*overlaparea / area1;
		*ratio2 += 1.0*overlaparea / area2;
	}
}

void iterDir(string path,vector<string>& filenames) {
	DIR *dir;
	struct dirent *ent;
	//string path = "C:\\Users\\sudol\\Desktop\\labelimg\\boxlabel";
	if ((dir = opendir(path.c_str())) != NULL) {
		/* print all the files and directories within directory */
		while ((ent = readdir(dir)) != NULL) {

			string filename = ent->d_name;
			if(filename.compare(".")!=0 && filename.compare("..") != 0)
				filenames.push_back(filename);
			//cout << filename << endl;
			//printf("%s\n", ent->d_name);
		}
		closedir(dir);
	}
	else {
		/* could not open directory */
		perror("");
	}
}