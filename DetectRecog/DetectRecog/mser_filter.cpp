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

	//�趨��ֵ
	double heightRatio = (double(deeplabBbox.height)) / srcImg.rows;
	//������һ����ͷ�����λ����ʱ����߽���һ��У��
	if (deeplabBbox.y+deeplabBbox.height > 0.99*srcImg.rows && deeplabBbox.y > srcImg.rows*0.5)
		heightRatio += 0.2;
	//����ͷλ�ù��͵�У��
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
	//�趨��ֵ
	double heightRatio = (double(deeplabBbox.height)) / srcImg.rows;
	//������һ����ͷ�����λ����ʱ����߽���һ��У��
	if (deeplabBbox.y + deeplabBbox.height > 0.99*srcImg.rows && deeplabBbox.y > srcImg.rows*0.5)
		heightRatio += 0.2;
	//����ͷλ�ù��͵�У��
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

// ��ȫ��ͼ�ϻ�bbox
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

// ��һ��ȫ��ͼ�л��������
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
	else if (colMaxLen > rowMaxLen)	// col����õ��Ľ�����ã�����col��
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
	for (int i = 0; bboxes.begin() + i != bboxes.end(); i++)	//����deeplab bbox��ɾ��
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
		// size���Բ������ֵĲ�Ҫ
		if (bboxes[i].height < bboxes[i].width || bboxes[i].height <= mserHeightSmall
			|| bboxes[i].height >= mserHeightHuge || bboxes[i].width >= mserWidthHuge
			|| bboxes[i].height > 10 * bboxes[i].width)
		{
			bboxes.erase(bboxes.begin() + i);
			i--;
		}
	}

	// ɾ���ص�����
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

void MserFilter::clusterFilter(void) //���˲�ɾbbox��Ҫ��ʼ������
{
	// ��bbox������������࣬����Ľ��ΪMserFilter��Ա
	cluster(mserWidthHuge*2.5, ROW_MODE);
	cluster(mserHeightHuge*2, COL_MODE);
	
	int rowMaxLen = rowClusterFilter();
	int colMaxLen = colClusterFilter();

	if (rowMaxLen == 0 || colMaxLen == 0)
	{
		rowClusters.clear();
		colClusters.clear();
	}
	else if (colMaxLen / rowMaxLen >= 2 || (colMaxLen >= 6 && rowMaxLen <= 4))	// col����õ��Ľ�����ã�����col��
		rowClusters.clear();
	else if (rowMaxLen / colMaxLen >= 2 || (rowMaxLen >= 6 && colMaxLen <= 4))
		colClusters.clear();
	else	// hard
	{
		cout << rowMaxLen << ' ' << colMaxLen << endl;
	}
}

// �о�����о��࣬ȡ����mode��mode�ᴫ��isClose()�����ж��Ƿ�����һ��(�з�������з���)
void MserFilter::cluster(int disThres, int mode)
{
	// idx�д����ÿ��bbox�������࣬��ʼʱÿ��bbox�������Լ�һ��
	vector<int> idx(bboxes.size());
	for (int i = 0; i < idx.size(); i++)
		idx[i] = i;

	// ��ʼ���࣬��forѭ�������һ����bboxes�ȳ������飬������Ԫ�ر�ʾ��Ӧ��bbox��������
	for (int i = 0; i < bboxes.size(); i++)
		for (int j = i + 1; j < bboxes.size(); j++)
			if (isClose(bboxes[i], bboxes[j], disThres, mode) == true)
			{
				if (idx[j] == idx[i])	//����ͬһ����
					continue;
				else if (idx[j] == j)	//��������
					idx[j] = idx[i];
				else                    //close��������һ���࣬�ϲ���������
				{
					int srcClass = idx[j]>idx[i] ? idx[j] : idx[i];
					int desClass = idx[i] == srcClass ? idx[j] : idx[i];
					for (int k = 0; k < idx.size(); k++)
						if (idx[k] == srcClass)
							idx[k] = desClass;
				}
			}

	// ����������˳��������棬һ�����Ӧһ��˳���������֮��Ĳ���
	// �˴�����ʹ�ö��forѭ��
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

	if (mode == ROW_MODE)	//���������洢
		rowClusters = bboxClusters;
	else
		colClusters = bboxClusters;
}

int MserFilter::rowClusterFilter(void)
{
	//�Ⱦ���ɾ������л�Ӱ��mainRow�ж��ĸ�����(��ȽϿ��µ��ࣩ
	for (int i = 0; rowClusters.begin() + i != rowClusters.end(); i++)
	{
		//���������λ�ù��ߵ���ͼƬ��һ�������������ʱ���������ִ���
		if (!(deeplabBbox.y + deeplabBbox.height > 0.95*srcImg.rows && deeplabBbox.y > srcImg.rows*0.5))
			if ((rowClusters[i][0].y - deeplabBbox.y)*1.0 / deeplabBbox.height > 0.4)
			{
				rowClusters.erase(rowClusters.begin() + i);
				i--;
			}
	}
	// ���������
	int mainRow = -1;
	int maxLength = 0;
	for (int i = 0; i < rowClusters.size(); i++)	//����mainRow
		if (rowClusters[i].size() > maxLength)
		{
			mainRow = i;
			maxLength = rowClusters[i].size();
		}

	if (mainRow == -1)
		return 0;

	// ����mainRow����
	int charHeight = 0, charWidth = 0;
	for (int i = 0; i < rowClusters[mainRow].size(); i++)
	{
		charHeight += rowClusters[mainRow][i].height;
		charWidth += rowClusters[mainRow][i].width;
	}
	charHeight = charHeight*1.0 / rowClusters[mainRow].size();
	charWidth = charWidth*1.0 / rowClusters[mainRow].size();

	// ɾ����mainRow��Զ����(x����+y����)
	for (int i = 0; i < rowClusters.size(); i++)	//�ȼ�����һ��
		sort(rowClusters[i].begin(), rowClusters[i].end(), sortByX);
	for (int i = 0; i < rowClusters.size(); i++)
	{
		if (rowClusters[i][0].y <= rowClusters[mainRow][0].y 
			  && rowClusters[mainRow][0].y - rowClusters[i][0].y > 2*charHeight)	//��mainRow̫�Ϸ�
			rowClusters[i].clear();
		if (rowClusters[i][0].y > rowClusters[mainRow][0].y 
			  && rowClusters[i][0].y - rowClusters[mainRow][0].y > 3 * charHeight)  //��mainRow̫�·�
			rowClusters[i].clear();
		if(rowClusters[i][rowClusters[i].size()-1].x < rowClusters[mainRow][0].x
			  && rowClusters[mainRow][0].x - rowClusters[i][rowClusters[i].size() - 1].x > 8 * charWidth)	//̫���
			rowClusters[i].clear();
		if(rowClusters[mainRow][rowClusters[mainRow].size() - 1].x < rowClusters[i][0].x
			  && rowClusters[i][0].x - rowClusters[mainRow][rowClusters[mainRow].size() - 1].x  > 8 * charWidth)	//̫�ұ�
			rowClusters[i].clear();
	}

	return maxLength;
}

int MserFilter::colClusterFilter(void)
{
	//ɾ�������ࣺ�ܻ��ӵĹ���ľ���(������ܸ߶�ԶС�ڸ���bbox�ĸ߶�֮��)
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
	for (int i = 0; i < colClusters.size(); i++)	//��ÿ�������о��ΰ��մ��ϵ��µ�˳������
		sort(colClusters[i].begin(), colClusters[i].end(), sortByY);

	for (int i = 0; i < colClusters.size(); i++)	//����mainCol����mainCol��������д��ڼ�װ���ϰ벿��Ԫ��
		if (colClusters[i].size() > maxLength)
			if ((deeplabBbox.y + deeplabBbox.height>0.95*srcImg.rows && deeplabBbox.y > srcImg.rows*0.5) //�������λ�ù��ߵ�����������޴���Ҫ��
				|| (colClusters[i][0].y-deeplabBbox.y < 0.4*deeplabBbox.height))
			{
				mainCol = i;
				maxLength = colClusters[i].size();
			}

	if (mainCol == -1)
		return 0;

	//����һ�к�Զ����Ҫɾ��
	int charWidth = 0, charHeight = 0;
	for (int i = 0; i < colClusters[mainCol].size(); i++)
	{
		charHeight += colClusters[mainCol][i].height;
		charWidth += colClusters[mainCol][i].width;
	}
	charWidth = charWidth / colClusters[mainCol].size();
	charHeight = charHeight / colClusters[mainCol].size();

	for (int i = 0; i < colClusters.size(); i++)
		if (abs(colClusters[i][0].x - colClusters[mainCol][0].x) > 8 * charWidth)	//��mainCol̫��߻���̫�ұ�
			colClusters[i].clear();
	return maxLength;
}

void MserFilter::patch(void)
{

}

MserFilter::~MserFilter()
{

}

// �ж������Ƿ���Ա���Ϊһ��
bool MserFilter::isClose(Rect r1, Rect r2, int disThres, int mode)
{
	int r1CenterX = r1.x + r1.width / 2, r1CenterY = r1.y + r1.height / 2;
	int r2CenterX = r2.x + r2.width / 2, r2CenterY = r2.y + r2.height / 2;
	if (mode == COL_MODE)		//��Ҫ��һ�о�Ϊһ��
	{
		double xCover = min(abs(r1.x - r2.x - r2.width), abs(r2.x - r1.x - r1.width));//����rect֮��x�����غϵĳ���
		if(max(abs(r1.x - r2.x - r2.width), abs(r2.x - r1.x - r1.width)) < r1.width + r2.width)
			if (xCover / r1.width > 0.60 || xCover / r2.width > 0.60) // ����Rect�Ƿ���һ����
				if (abs(r1CenterY - r2CenterY) < disThres)		//����distance��������
					return true;
	}
	if (mode == ROW_MODE)		// ��һ�о�Ϊһ��
	{
		double yCover = min(abs(r1.y - r2.y - r2.height), abs(r2.y - r1.y - r1.height));//����rect֮��y�����غϵĳ���
		if (max(abs(r1.y - r2.y - r2.height), abs(r2.y - r1.y - r1.height)) < r1.height + r2.height)
			if (yCover / r1.height > 0.8 || yCover / r2.height > 0.8) // ����Rect�Ƿ���һ���ϣ��˴�Ӧ�ø��ϸ�һ��
				if (abs(r1CenterX - r2CenterX) < disThres) //����distance
					return true;
	}	
	return false;
}
