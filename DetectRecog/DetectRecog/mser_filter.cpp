#include "mser_filter.h"
using namespace cv;

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
	rectangle(mserImg, deeplabBbox, 255, 1, 8, 0);
	imwrite(outputPath, mserImg);
}

// 在全黑图上画bbox
void MserFilter::drawBboxes(string outputPath, vector<Rect> bboxes)
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
			int color = (240 / clusterNum) * clusterCnt;
			for (int j = 0; j < clusters[i].size(); j++)
			{
				//color = 250;
				rectangle(mserImg, clusters[i][j], color, CV_FILLED, 8, 0);
			}
		}
	}
	rectangle(mserImg, Rect(0, 0, int(mserWidthHuge), int(mserHeightSmall)), 255, 1, 8, 0);
	rectangle(mserImg, Rect(0, 0, int(mserWidthHuge), int(mserHeightHuge)), 255, 1, 8, 0);
	rectangle(mserImg, deeplabBbox, 255, 1, 8, 0);
	imwrite(outputPath, mserImg);
}

void MserFilter::drawResult(string outputPath, vector<Rect> result)
{
	Mat resImg = srcImg.clone();
	for (int i = 0; i < result.size(); i++)
		rectangle(resImg, result[i], Scalar(0, 255, 0), 1, 8, 0);
	rectangle(resImg, deeplabBbox, Scalar(0, 255, 0), 1, 8, 0);
	imwrite(outputPath, resImg);
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
	cluster(mserWidthHuge*2.5, bboxes, rowClusters, ROW_MODE);
	cluster(mserHeightHuge*2, bboxes, colClusters, COL_MODE);
	
	int mainRowSize = clusterProcess(ROW_MODE);
	int mainColsize = clusterProcess(COL_MODE);

	if (mainRowSize==0 && mainColsize==0)	//都为0
	{
		rowClusters.clear();
		colClusters.clear();
	}
	else if(mainRowSize==0 && mainColsize != 0)	//有一者为0
		rowClusters.clear();
	else if (mainColsize == 0 && mainRowSize != 0)
		colClusters.clear();
	else	//均不为0
	{
		if((mainColsize >= 6 && mainRowSize <= 4) || mainColsize >= 11 || mainColsize / mainRowSize >= 2)	// col聚类得到的结果更好，保留col类
			rowClusters.clear();
		else if ((mainRowSize >= 6 && mainColsize <= 4) || mainRowSize >= 11 || mainRowSize / mainColsize >= 2)
			colClusters.clear();
		else	// hard，判断谁更在右边一点
		{
			if (rowResult[0].x + 0.5*rowResult[0].width >= colResult[0].x)
				colClusters.clear();
			else
				rowClusters.clear();
		}
	}
}

// 行聚类或列聚类，取决于mode，mode会传给isClose()函数判定是否属于一类(行方向或者列方向)
void MserFilter::cluster(int disThres, vector<Rect> bboxes, vector<vector<Rect>>& clusters,int mode)
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
	clusters.clear();
	clusters.resize(idx.size());
	for (int i = 0; i < idx.size(); i++)
		clusters[idx[i]].push_back(bboxes[i]);
	for (int i = 0; clusters.begin() + i != clusters.end(); i++)
	{
		if (clusters[i].size() <= 0)
		{
			clusters.erase(clusters.begin()+i);
			i--;
		}
	}
}

//返回其最长的rowCluster的size
int MserFilter::clusterProcess(int mode)
{
	vector<Rect> mainCluster;
	if(mode == ROW_MODE)	//找到最长的类
		findMainRowCluster(mainCluster);
	else
		findMainColCluster(mainCluster);

	if (mainCluster.size() < 4)
		return 0;

	if (mode == ROW_MODE)	//根据最长的类做进一步的filter和patch
	{
		delByMainRow(mainCluster);
		buildRowResult();
	}
	else
	{
		delByMainCol(mainCluster);
		buildColResult();
	}
	return mainCluster.size();
}

void MserFilter::findMainRowCluster(vector<Rect>& mainRowCluster)
{
	//先删除比较靠下的类,避免干扰
	vector<vector<Rect>> clusterTmp(rowClusters);
	for (int i = 0; clusterTmp.begin() + i != clusterTmp.end(); i++)
		//对于摄像机位置过高导致图片有一半在镜头外面时，不做此种处理
		if (!(deeplabBbox.y + deeplabBbox.height > 0.95*srcImg.rows && deeplabBbox.y > srcImg.rows*0.5))
			if ((clusterTmp[i][0].y - deeplabBbox.y)*1.0 / deeplabBbox.height > 0.4)
			{
				clusterTmp.erase(clusterTmp.begin() + i);
				i--;
			}
	// 从长到短排序，默认最长的那个作为mainRow
	int mainRow = 0;
	if (clusterTmp.size() == 0)
		return;
	sort(clusterTmp.begin(), clusterTmp.end(), sortByLen);
	if (clusterTmp.size() > 1)
		if (clusterTmp[0].size() >= 6 && clusterTmp[1].size() >= 6)	//前两个都很长，则选择右边的那个
		{
			sort(clusterTmp[0].begin(), clusterTmp[0].end(), sortByX);
			sort(clusterTmp[1].begin(), clusterTmp[1].end(), sortByX);
			if (clusterTmp[1][0].x > clusterTmp[0][0].x)
				mainRow = 1;
		}
	mainRowCluster.assign(clusterTmp[mainRow].begin(), clusterTmp[mainRow].end());
}

void MserFilter::delByMainRow(vector<Rect> mainRowCluster)
{
	vector<Rect> clusterBboxes(rowClusters.size());	//计算每个聚类的bbox
	vector<int> avgWidth(rowClusters.size());		//计算平均size
	vector<int> avgHeight(rowClusters.size());
	for (int i = 0; i < rowClusters.size(); i++)
	{
		getClusterBbox(rowClusters[i], clusterBboxes[i]);
		getClusterAvgHW(rowClusters[i], avgHeight[i], avgWidth[i]);
	}
		
	//计算mainRowCluster的相关参数
	Rect mainBbox;
	int mainAvgHeight, mainAvgWidth;
	getClusterBbox(mainRowCluster, mainBbox);
	getClusterAvgHW(mainRowCluster, mainAvgHeight, mainAvgWidth);
	int mainXCenter = mainBbox.x + 0.5*mainBbox.width;
	int mainYCenter = mainBbox.y + 0.5*mainBbox.height;

	for (int i = 0; i < rowClusters.size(); i++)	// 根据mainRow删除
	{
		int xCenter = clusterBboxes[i].x + 0.5*clusterBboxes[i].width;
		int yCenter = clusterBboxes[i].y + 0.5*clusterBboxes[i].height; 
		// 此处即是只保留mainRow正上，正下，正左，正右区域的类(不包括斜着的区域)，xCenter左边条件更松一点是因为有时候mainRow左边会漏检几位
		if((xCenter+0.25*clusterBboxes[i].width< mainBbox.x || xCenter > mainBbox.x + mainBbox.width)
			&&(yCenter < mainBbox.y || yCenter > mainBbox.y + mainBbox.height))
		{
			rowClusters[i].clear();
			continue;
		}
		// 删除离mainRow的y方向过远的类
		if ((mainRowCluster.size() <= 8 && abs(yCenter - mainYCenter) >= 2 * mainBbox.height)	//排布是三行的情况
			|| (mainRowCluster.size()>8 && (yCenter-mainYCenter>=2*mainBbox.height||yCenter-mainYCenter<=-0.75*mainBbox.height)))	//排布是两行的情况，此时上方的检测会放的更严
		{
			rowClusters[i].clear();
			continue;
		}
		if (clusterBboxes[i].x - mainBbox.x - mainBbox.width >= 8*mainAvgWidth
			|| mainBbox.x-clusterBboxes[i].x- clusterBboxes[i].width>=8* mainAvgWidth)	// 删除离mainRow的x方向过远的类
		{
			rowClusters[i].clear();
			continue;
		}
	}

	//将删除后且最终过滤完成的结果重新整理一下
	vector<Rect> rowFilteredBboxes;
	for (int i = 0; i < rowClusters.size(); i++)
		for (int j = 0; j < rowClusters[i].size(); j++)
			rowFilteredBboxes.push_back(rowClusters[i][j]);
	cluster(1000, rowFilteredBboxes, rowClusters, ROW_LOOSE_MODE);	//重新聚类，一行作为一类，此时聚类条件很松，聚类结果还在rowCluster中
}

void MserFilter::buildRowResult(void)
{
	sort(rowClusters.begin(), rowClusters.end(), sortByLen);

	for (int i = 0; i < rowClusters.size() && i<3; i++)	//按长度排序后，为前三个类找大bbox
	{
		sort(rowClusters[i].begin(), rowClusters[i].end(), sortByX);
		int minYTop = 10000, maxYBottom = 0;
		for (int j = 0; j < rowClusters[i].size(); j++)
		{
			minYTop = min(minYTop, rowClusters[i][j].y);
			maxYBottom = max(maxYBottom, rowClusters[i][j].y + rowClusters[i][j].height);
		}
		int endIdx = rowClusters[i].size() - 1;
		rowResult.push_back(Rect(rowClusters[i][0].x, minYTop, rowClusters[i][endIdx].x + rowClusters[i][endIdx].width - rowClusters[i][0].x, maxYBottom - minYTop));
	}

	//最长的箱号左边起码应该是在最左边
	int farLeft = rowResult[0].x;
	for (int i = 1; i < rowResult.size(); i++)
		farLeft = min(rowResult[i].x, farLeft);
	rowResult[0].width = rowResult[0].x - farLeft + rowResult[0].width;
	rowResult[0].x = farLeft;

	//三行的情况上面的公司号应该和mainRow左对齐，下面的箱型若是其在mainRow的左半部分，且mainRow是只有箱号的，也应该左对齐，且此时箱号和箱型也应该右对齐
	for (int i = 1; i < rowResult.size() && rowResult.size()==3; i++)
	{
		if (rowResult[i].y < rowResult[0].y)	//即公司号的情况
		{
			rowResult[i].width = rowResult[i].x + rowResult[i].width - rowResult[0].x;
			rowResult[i].x = rowResult[0].x;
		}
		if(rowResult[i].y > rowResult[0].y && rowResult[i].x+rowResult[i].width*0.5 < rowResult[0].x+rowResult[0].width*0.5 && rowClusters[0].size() <= 8)
		{
			rowResult[i].width = rowResult[i].x + rowResult[i].width - rowResult[0].x;
			rowResult[i].x = rowResult[0].x;
			int upIdx = (i == 1) ? 2 : 1;
			int maxRight = max(rowResult[i].x + rowResult[i].width, rowResult[upIdx].x + rowResult[upIdx].width);
			rowResult[i].width = maxRight - rowResult[i].x;
			rowResult[upIdx].width = maxRight - rowResult[upIdx].x;
		}
	}

	// 若是三行，则箱号至少要有公司号的二倍长
	for (int i = 1; i < rowResult.size() && rowResult.size() == 3; i++)
		if (rowResult[i].y < rowResult[0].y)	//即公司号的情况
			rowResult[0].width = max(rowResult[0].width, rowResult[i].width*2);

	//左右再延长一点
	for (int i = 0; i < rowResult.size(); i++)
	{
		rowResult[i].x = max(deeplabBbox.x, rowResult[i].x - 5);
		rowResult[i].width = min(rowResult[i].width + 10, deeplabBbox.x + deeplabBbox.width - rowResult[i].x);
	}
}

void MserFilter::findMainColCluster(vector<Rect>& mainColCluster)
{
	// 合并竖类聚类中本不应该出现的并排的bbox
	for (int i = 0; i < colClusters.size(); i++)	
	{
		sort(colClusters[i].begin(), colClusters[i].end(), sortByY);	//从上到下排列一下
		for (int j = 1; colClusters[i].begin() + j != colClusters[i].end(); j++)
			if ((colClusters[i][j - 1].y + colClusters[i][j - 1].height - colClusters[i][j].y)*1.0 / colClusters[i][j - 1].height > 0.7)	//竖直方向重合度大于0.7
			{
				colClusters[i][j - 1] = colClusters[i][j - 1] | colClusters[i][j];
				colClusters[i].erase(colClusters[i].begin() + j);
				j--;
			}
	}
	if (colClusters.size() == 0)
		return ;

	int mainCol = 0;	//查找最长列
	sort(colClusters.begin(), colClusters.end(), sortByLen);	//从长到短排列
	//如果集装箱整个位置都出现在画面中，则要求mainCol必需有元素处于集装箱上半部，否则则直接选取最长的那个
	if (!(deeplabBbox.y + deeplabBbox.height>0.99*srcImg.rows && deeplabBbox.y > srcImg.rows*0.5))	
		for (int i = 0; i < colClusters.size(); i++)	//mainCol聚类是有处于集装箱上半部的元素的最长的聚类
			if (colClusters[i][0].y - deeplabBbox.y < 0.4*deeplabBbox.height)
			{
				mainCol = i;
				break;
			}
	mainColCluster.assign(colClusters[mainCol].begin(), colClusters[mainCol].end());
}

void MserFilter::delByMainCol(vector<Rect> mainColCluster)
{
	int mainAvgWidth = 0, mainAvgHeight = 0;	// 获取mainCluster的bbox相关参数，根据这些参数来删除一些类
	Rect mainBbox;
	getClusterAvgHW(mainColCluster, mainAvgHeight, mainAvgWidth);
	getClusterBbox(mainColCluster, mainBbox);
	int mainXCenter = mainBbox.x + mainBbox.width*0.5;

	for (int i = 0; colClusters.begin()+i != colClusters.end(); i++)	//删除一些类
	{
		Rect rtmp;
		getClusterBbox(colClusters[i], rtmp);
		int xCenter = rtmp.x + rtmp.width*0.5, yCenter = rtmp.y + rtmp.height*0.5;
		bool deleteFlag = false;
		if ((xCenter<mainBbox.x || xCenter>mainBbox.x + mainBbox.width) //x, y方向均没有重合的类，删掉
			&& (yCenter<mainBbox.y || yCenter>mainBbox.y + mainBbox.height))
			deleteFlag = true;
		if (abs(xCenter - mainXCenter) >= 8 * mainAvgWidth)	// 删除水平方向离mainCol过远的类
			deleteFlag = true;
		if(deleteFlag == true)
		{
			colClusters.erase(colClusters.begin() + i);
			i--;
		}
	}

	//将删除后的结果重新整理一下，再进行竖直方向的聚类
	vector<Rect> colFilteredBboxes;
	for (int i = 0; i < colClusters.size(); i++)
		for (int j = 0; j < colClusters[i].size(); j++)
			colFilteredBboxes.push_back(colClusters[i][j]);
	cluster(2000, colFilteredBboxes, colClusters, COL_MODE);	//重新聚类，此处阈值设的很大，只要在一条竖线上即可认为是一类
}

void MserFilter::buildColResult(void)
{
	sort(colClusters.begin(), colClusters.end(), sortByLen);
	// 进行类中的进一步聚类，以进行colCluster[0](也即mainCol)的修正，目标是聚为三类(4+6+1)
	vector<vector<Rect>> mainCol;
	int mainAvgHeight = 0, mainAvgWidth = 0;
	getClusterAvgHW(colClusters[0], mainAvgHeight, mainAvgWidth);
	cluster(1.25*mainAvgHeight, colClusters[0], mainCol, COL_MODE);
	/*for (int i = 0; mainCol.begin() + i + 1 != mainCol.end(); i++)	//第一阶段，将mainCol[0]的size校正到4
	{
		if (mainCol[0].size() >= 4)
			break;
		if(mainCol[0] + mainCol[1])
	}*/

	//进行非mainCol的处理，要把非mainCol合并成一类
	for (int i = 1; i < colClusters.size(); i++)
	{
		if (i == 1 || colClusters[i].size() >= 4)
		{
			vector<vector<Rect>> tmpCluster;
			int avgHeight = 0, avgWidth = 0;
			getClusterAvgHW(colClusters[0], avgHeight, avgWidth);
			cluster(1.25*mainAvgHeight, colClusters[i], tmpCluster, COL_MODE);
			sort(tmpCluster.begin(), tmpCluster.end(), clusterSortByY);

			int cnt = tmpCluster.size();
			while(tmpCluster.size() > 1)	//
			{
				cnt--;
				if (cnt < 0)
					break;
				if (tmpCluster[0].size() == 4)
				{
					colClusters[i].assign(tmpCluster[0].begin(), tmpCluster[0].end());
					break;
				}
				if (tmpCluster[1].size() > 4)
				{
					colClusters[i].clear();
					break;
				}
				if (tmpCluster[1].size() == 4)
				{
					colClusters[i].assign(tmpCluster[1].begin(), tmpCluster[1].end());
					break;
				}
				if (tmpCluster[0].size() + tmpCluster[1].size() >= 4)
				{
					if (tmpCluster[0].size() < tmpCluster[1].size())
						tmpCluster[0].assign(tmpCluster[1].begin(), tmpCluster[1].end());
					tmpCluster.erase(tmpCluster.begin() + 1);
				}
			}
		}
	}

	for (int i = 0; i < colClusters.size() && i<2; i++)
		for (int j = 0; j < colClusters[i].size(); j++)
		colResult.push_back(colClusters[i][j]);
}

MserFilter::~MserFilter()
{

}

// 判断两者是否可以被聚为一类
bool MserFilter::isClose(Rect r1, Rect r2, int disThres, int mode)
{
	int r1CenterX = r1.x + r1.width / 2, r1CenterY = r1.y + r1.height / 2;
	int r2CenterX = r2.x + r2.width / 2, r2CenterY = r2.y + r2.height / 2;
	if (mode == COL_MODE || mode == COL_LOOSE_MODE)		//需要把一列聚为一类
	{
		double xCoverRatio = 0.65;
		if (mode == ROW_LOOSE_MODE) //宽松行聚类，放宽了条件
			xCoverRatio = 0.2;
		double xCover = min(abs(r1.x - r2.x - r2.width), abs(r2.x - r1.x - r1.width));//两个rect之间x方向重合的长度
		if(max(abs(r1.x - r2.x - r2.width), abs(r2.x - r1.x - r1.width)) < r1.width + r2.width)
			if (xCover / r1.width > xCoverRatio || xCover / r2.width > xCoverRatio) // 两个Rect是否在一列上
				if (abs(r1CenterY - r2CenterY) < disThres)		//两者distance满足条件
					return true;
	}
	if (mode == ROW_MODE || mode == ROW_LOOSE_MODE)		// 把一行聚为一类
	{
		double yCoverRatio = 0.8;
		if (mode == ROW_LOOSE_MODE) //宽松行聚类，放宽了条件
			yCoverRatio = 0.6;
		double yCover = min(abs(r1.y - r2.y - r2.height), abs(r2.y - r1.y - r1.height));//两个rect之间y方向重合的长度
		if (max(abs(r1.y - r2.y - r2.height), abs(r2.y - r1.y - r1.height)) < r1.height + r2.height)
			if (yCover / r1.height > yCoverRatio || yCover / r2.height > yCoverRatio) // 两个Rect是否在一行上，此处应该更严格一点
				if (abs(r1CenterX - r2CenterX) < disThres) //两者distance
					return true;
	}	
	return false;
}

int MserFilter::judgeSide(void)
{
	filter();
	if (rowClusters.size() != 0)
		return ROW_MODE;
	else if (colClusters.size() != 0)
		return COL_MODE;
	else
		return WRONG;

}
