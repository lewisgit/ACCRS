#include "mser_filter.h"

extern double charHeightSmall, charHeightHuge, charWidthHuge;

MserFilter::MserFilter(Mat srcImg, Rect deeplabBbox)
{
	Mat grayImg = Mat::zeros(srcImg.size(), CV_8UC1);
	cvtColor(srcImg, grayImg, COLOR_BGR2GRAY);

	this->srcImg = srcImg;
	this->deeplabBbox = deeplabBbox;
	Ptr<MSER> mser = MSER::create(5, 50, 500, 0.4);
	mser->detectRegions(grayImg, msers, bboxes);

	//设定阈值
	double heightRatio = (double(deeplabBbox.height)) / srcImg.rows;
	//对于另一个码头摄像机位置有时候过高进行一次校正
	if (deeplabBbox.y+deeplabBbox.height > 0.99*srcImg.rows && deeplabBbox.y > srcImg.rows*0.5)
		heightRatio += 0.2;
	//摄像头位置过低的校正
	switch ((int)(heightRatio*10))
	{
	case 1:
	case 2:
	case 3: 
	case 4: mserHeightSmall = 15; mserHeightHuge = 27; mserWidthHuge = 20; break;
	case 5: mserHeightSmall = 18; mserHeightHuge = 35; mserWidthHuge = 25; break;
	case 6: mserHeightSmall = 20; mserHeightHuge = 50; mserWidthHuge = 32; break;
	case 7: mserHeightSmall = 22; mserHeightHuge = 52; mserWidthHuge = 38; break;
	case 8:
	case 9: mserHeightSmall = 25; mserHeightHuge = 55; mserWidthHuge = 42; break;
	}
}

MserFilter::MserFilter(Mat srcImg, vector<Rect> bboxes, Rect deeplabBbox)
{
	this->srcImg = srcImg;
	this->deeplabBbox = deeplabBbox;
	this->bboxes = bboxes;
	//设定阈值
	double heightRatio = (double(deeplabBbox.height)) / srcImg.rows;
	//对于另一个码头摄像机位置有时候过高进行一次校正
	if (deeplabBbox.y + deeplabBbox.height > 0.99*srcImg.rows && deeplabBbox.y > srcImg.rows*0.5)
		heightRatio += 0.2;
	//摄像头位置过低的校正
	switch ((int)(heightRatio * 10))
	{
	case 1:
	case 2:
	case 3:
	case 4: mserHeightSmall = 15; mserHeightHuge = 27; mserWidthHuge = 20; break;
	case 5: mserHeightSmall = 18; mserHeightHuge = 35; mserWidthHuge = 25; break;
	case 6: mserHeightSmall = 20; mserHeightHuge = 50; mserWidthHuge = 32; break;
	case 7: mserHeightSmall = 22; mserHeightHuge = 52; mserWidthHuge = 38; break;
	case 8:
	case 9: mserHeightSmall = 25; mserHeightHuge = 55; mserWidthHuge = 42; break;
	}
}

void MserFilter::drawMsers(string outputPath)
{
	Mat mserImg = Mat::zeros(srcImg.size(), CV_8UC1);
	for (int i = 0; i < msers.size(); i++)
	{
		vector<Point>& r = msers[i];
		for (int j = 0; j < (int)r.size(); j++)
		{
			Point pt = r[j];
			mserImg.at<unsigned char>(pt) = 255;
		}
	}
	imwrite(outputPath, mserImg);
}

// 在全黑图上画bbox
void MserFilter::drawBboxes(string outputPath)
{
	Mat mserImg = Mat::zeros(srcImg.size(), CV_8UC1);
	for (int i = 0; i < bboxes.size(); i++)
	{
		Rect bbox = bboxes[i];
		rectangle(mserImg, bbox, 255, 1, 8, 0);
	}
	rectangle(mserImg, Rect(0, 0, int(mserWidthHuge), int(mserHeightSmall)), 255, 1, 8, 0);
	rectangle(mserImg, Rect(0, 0, int(mserWidthHuge), int(mserHeightHuge)), 255, 1, 8, 0);
	rectangle(mserImg, deeplabBbox, 255, 1, 8, 0);
	imwrite(outputPath, mserImg);
}

// 在一张全黑图中画聚类情况
void MserFilter::drawClusters(string outputPath, vector<vector<Rect>> clusters)
{
	Mat mserImg = Mat::zeros(srcImg.size(), CV_8UC1);
	int clusterNum = 0;
	for (int i = 0; i < clusters.size(); i++)
		if (clusters[i].size() != 0)
			clusterNum++;

	int clusterCnt = 0;
	for (int i = 0; i < clusters.size(); i++)
	{
		if(clusters[i].size() != 0)
		{
			clusterCnt++;
			int color = (220 / clusterNum) * clusterCnt;
			for (int j = 0; j < clusters[i].size(); j++)
			{
				rectangle(mserImg, clusters[i][j], color, CV_FILLED, 8, 0);
			}
		}
	}
	rectangle(mserImg, Rect(0, 0, int(mserWidthHuge), int(mserHeightSmall)), 255, 1, 8, 0);
	rectangle(mserImg, Rect(0, 0, int(mserWidthHuge), int(mserHeightHuge)), 255, 1, 8, 0);
	rectangle(mserImg, deeplabBbox, 255, 1, 8, 0);
	imwrite(outputPath, mserImg);
}

int MserFilter::judgeSide(void)
{
	deeplabFilter();
	singleBboxFilter();

	cluster(mserWidthHuge*2.5, ROW_MODE);
	cluster(mserHeightHuge * 2, COL_MODE);

	int rowMaxLen = rowClusterFilter();
	int colMaxLen = colClusterFilter();

	if (rowMaxLen <= 3 && colMaxLen <= 3)
		return WRONG;
	else if (colMaxLen > rowMaxLen)	// col聚类得到的结果更好，保留col类
		return COL_MODE;
	else if (rowMaxLen > colMaxLen)
		return ROW_MODE;
	else	// hard
		return WRONG;
}

void MserFilter::filter(void)
{
	deeplabFilter();
	singleBboxFilter();
	clusterFilter();
}

void MserFilter::deeplabFilter(void) 
{
	for (int i = 0; bboxes.begin() + i != bboxes.end(); i++)	//超出deeplab bbox的删除
	{
		if (bboxes[i].x < deeplabBbox.x || bboxes[i].x + bboxes[i].width > deeplabBbox.x + deeplabBbox.width
			|| bboxes[i].y < deeplabBbox.y || bboxes[i].y + bboxes[i].height > deeplabBbox.y + deeplabBbox.height)
		{
			bboxes.erase(bboxes.begin() + i);
			i--;
		}
	}
}

void MserFilter::singleBboxFilter(void)
{
	for (int i = 0; bboxes.begin() + i != bboxes.end(); i++)
	{
		// size明显不是文字的不要
		if (bboxes[i].height < bboxes[i].width || bboxes[i].height <= mserHeightSmall
			|| bboxes[i].height >= mserHeightHuge || bboxes[i].width >= mserWidthHuge
			|| bboxes[i].height > 10 * bboxes[i].width)
		{
			bboxes.erase(bboxes.begin() + i);
			i--;
		}
	}

	// 删除重叠矩形
	for (int i = 0; bboxes.begin() + i != bboxes.end(); i++)
		for (int j = i + 1; bboxes.begin() + j != bboxes.end(); j++)
			if (((double)(bboxes[i] & bboxes[j]).area()) / bboxes[i].area() > 0.25
				|| ((double)(bboxes[i] & bboxes[j]).area()) / bboxes[j].area() > 0.25)
			{
				if (bboxes[i].area() < bboxes[j].area())
				{
					bboxes.erase(bboxes.begin() + i);
					i--;
					break;
				}
				else
				{
					bboxes.erase(bboxes.begin() + j);
					j--;
				}
			}
}

void MserFilter::clusterFilter(void) //到此步删bbox就要开始谨慎了
{
	// 将bbox分两个方向聚类，聚类的结果为MserFilter成员
	cluster(mserWidthHuge*2.5, ROW_MODE);
	cluster(mserHeightHuge*2, COL_MODE);
	
	int rowMaxLen = rowClusterFilter();
	int colMaxLen = colClusterFilter();

	if (rowMaxLen == 0 || colMaxLen == 0)
	{
		rowClusters.clear();
		colClusters.clear();
	}
	else if (colMaxLen / rowMaxLen >= 2 || (colMaxLen >= 6 && rowMaxLen <= 4))	// col聚类得到的结果更好，保留col类
		rowClusters.clear();
	else if (rowMaxLen / colMaxLen >= 2 || (rowMaxLen >= 6 && colMaxLen <= 4))
		colClusters.clear();
	else	// hard
	{
		cout << rowMaxLen << ' ' << colMaxLen << endl;
	}
}

// 行聚类或列聚类，取决于mode，mode会传给isClose()函数判定是否属于一类(行方向或者列方向)
void MserFilter::cluster(int disThres, int mode)
{
	// idx中存的是每个bbox所属的类，初始时每个bbox都属于自己一类
	vector<int> idx(bboxes.size());
	for (int i = 0; i < idx.size(); i++)
		idx[i] = i;

	// 开始聚类，此for循环结果是一个和bboxes等长的数组，数组中元素表示对应的bbox所属的类
	for (int i = 0; i < bboxes.size(); i++)
		for (int j = i + 1; j < bboxes.size(); j++)
			if (isClose(bboxes[i], bboxes[j], disThres, mode) == true)
			{
				if (idx[j] == idx[i])	//已在同一类中
					continue;
				else if (idx[j] == j)	//无所属类
					idx[j] = idx[i];
				else                    //close但不属于一个类，合并这两个类
				{
					int srcClass = idx[j]>idx[i] ? idx[j] : idx[i];
					int desClass = idx[i] == srcClass ? idx[j] : idx[i];
					for (int k = 0; k < idx.size(); k++)
						if (idx[k] == srcClass)
							idx[k] = desClass;
				}
			}

	// 将聚类结果用顺序表来保存，一个类对应一个顺序表，更方便之后的操作
	// 此处避免使用多层for循环
	vector<vector<Rect>> bboxClusters(idx.size());
	for (int i = 0; i < idx.size(); i++)
		bboxClusters[idx[i]].push_back(bboxes[i]);
	for (int i = 0; bboxClusters.begin() + i != bboxClusters.end(); i++)
	{
		if (bboxClusters[i].size() == 0)
		{
			bboxClusters.erase(bboxClusters.begin()+i);
			i--;
		}
	}

	if (mode == ROW_MODE)	//将聚类结果存储
		rowClusters = bboxClusters;
	else
		colClusters = bboxClusters;
}

int MserFilter::rowClusterFilter(void)
{
	//先尽量删除结果中会影响mainRow判定的干扰类(如比较靠下的类）
	for (int i = 0; rowClusters.begin() + i != rowClusters.end(); i++)
	{
		//对于摄像机位置过高导致图片有一半在摄像机外面时，不做此种处理
		if (!(deeplabBbox.y + deeplabBbox.height > 0.95*srcImg.rows && deeplabBbox.y > srcImg.rows*0.5))
			if ((rowClusters[i][0].y - deeplabBbox.y)*1.0 / deeplabBbox.height > 0.4)
			{
				rowClusters.erase(rowClusters.begin() + i);
				i--;
			}
	}
	// 查找最长的行
	int mainRow = -1;
	int maxLength = 0;
	for (int i = 0; i < rowClusters.size(); i++)	//查找mainRow
		if (rowClusters[i].size() > maxLength)
		{
			mainRow = i;
			maxLength = rowClusters[i].size();
		}

	if (mainRow == -1)
		return 0;

	// 根据mainRow计算
	int charHeight = 0, charWidth = 0;
	for (int i = 0; i < rowClusters[mainRow].size(); i++)
	{
		charHeight += rowClusters[mainRow][i].height;
		charWidth += rowClusters[mainRow][i].width;
	}
	charHeight = charHeight*1.0 / rowClusters[mainRow].size();
	charWidth = charWidth*1.0 / rowClusters[mainRow].size();

	// 删除离mainRow过远的类(x方向+y方向)
	for (int i = 0; i < rowClusters.size(); i++)	//先简单排列一下
		sort(rowClusters[i].begin(), rowClusters[i].end(), sortByX);
	for (int i = 0; i < rowClusters.size(); i++)
	{
		if (rowClusters[i][0].y <= rowClusters[mainRow][0].y 
			  && rowClusters[mainRow][0].y - rowClusters[i][0].y > 2*charHeight)	//在mainRow太上方
			rowClusters[i].clear();
		if (rowClusters[i][0].y > rowClusters[mainRow][0].y 
			  && rowClusters[i][0].y - rowClusters[mainRow][0].y > 3 * charHeight)  //在mainRow太下方
			rowClusters[i].clear();
		if(rowClusters[i][rowClusters[i].size()-1].x < rowClusters[mainRow][0].x
			  && rowClusters[mainRow][0].x - rowClusters[i][rowClusters[i].size() - 1].x > 8 * charWidth)	//太左边
			rowClusters[i].clear();
		if(rowClusters[mainRow][rowClusters[mainRow].size() - 1].x < rowClusters[i][0].x
			  && rowClusters[i][0].x - rowClusters[mainRow][rowClusters[mainRow].size() - 1].x  > 8 * charWidth)	//太右边
			rowClusters[i].clear();
	}

	return maxLength;
}

int MserFilter::colClusterFilter(void)
{
	//删除干扰类：很混杂的过宽的聚类(聚类后总高度远小于各个bbox的高度之和)
	for (int i = 0; colClusters.begin() + i != colClusters.end(); i++)
	{
		Rect r = colClusters[i][0];
		int totalHeight = colClusters[i][0].height;
		for (int j = 1; j < colClusters[i].size(); j++)
		{
			totalHeight += colClusters[i][j].height;
			r = r | colClusters[i][j];
		}
		if (r.height * 1.1 < totalHeight)
		{
			colClusters.erase(colClusters.begin() + i);
			i--;
		}
	}

	int mainCol = -1;
	int maxLength = 0;
	for (int i = 0; i < colClusters.size(); i++)	//把每个聚类中矩形按照从上到下的顺序排列
		sort(colClusters[i].begin(), colClusters[i].end(), sortByY);

	for (int i = 0; i < colClusters.size(); i++)	//查找mainCol，且mainCol聚类必须有处于集装箱上半部的元素
		if (colClusters[i].size() > maxLength)
			if ((deeplabBbox.y + deeplabBbox.height>0.95*srcImg.rows && deeplabBbox.y > srcImg.rows*0.5) //对于相机位置过高的特殊情况则无此种要求
				|| (colClusters[i][0].y-deeplabBbox.y < 0.4*deeplabBbox.height))
			{
				mainCol = i;
				maxLength = colClusters[i].size();
			}

	if (mainCol == -1)
		return 0;

	//离这一列很远的列要删除
	int charWidth = 0, charHeight = 0;
	for (int i = 0; i < colClusters[mainCol].size(); i++)
	{
		charHeight += colClusters[mainCol][i].height;
		charWidth += colClusters[mainCol][i].width;
	}
	charWidth = charWidth / colClusters[mainCol].size();
	charHeight = charHeight / colClusters[mainCol].size();

	for (int i = 0; i < colClusters.size(); i++)
		if (abs(colClusters[i][0].x - colClusters[mainCol][0].x) > 8 * charWidth)	//在mainCol太左边或者太右边
			colClusters[i].clear();
	return maxLength;
}

void MserFilter::patch(void)
{

}

MserFilter::~MserFilter()
{

}

// 判断两者是否可以被聚为一类
bool MserFilter::isClose(Rect r1, Rect r2, int disThres, int mode)
{
	int r1CenterX = r1.x + r1.width / 2, r1CenterY = r1.y + r1.height / 2;
	int r2CenterX = r2.x + r2.width / 2, r2CenterY = r2.y + r2.height / 2;
	if (mode == COL_MODE)		//需要把一列聚为一类
	{
		double xCover = min(abs(r1.x - r2.x - r2.width), abs(r2.x - r1.x - r1.width));//两个rect之间x方向重合的长度
		if(max(abs(r1.x - r2.x - r2.width), abs(r2.x - r1.x - r1.width)) < r1.width + r2.width)
			if (xCover / r1.width > 0.60 || xCover / r2.width > 0.60) // 两个Rect是否在一列上
				if (abs(r1CenterY - r2CenterY) < disThres)		//两者distance满足条件
					return true;
	}
	if (mode == ROW_MODE)		// 把一行聚为一类
	{
		double yCover = min(abs(r1.y - r2.y - r2.height), abs(r2.y - r1.y - r1.height));//两个rect之间y方向重合的长度
		if (max(abs(r1.y - r2.y - r2.height), abs(r2.y - r1.y - r1.height)) < r1.height + r2.height)
			if (yCover / r1.height > 0.8 || yCover / r2.height > 0.8) // 两个Rect是否在一行上，此处应该更严格一点
				if (abs(r1CenterX - r2CenterX) < disThres) //两者distance
					return true;
	}	
	return false;
}
