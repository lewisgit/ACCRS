#include"GetCandidate.h"
using namespace std;
using namespace cv;

std::vector<Candidate> GetCandidate::run(cv::Mat& Image)
{
	oriBgrImage_8UC3 = Image;
	//Image.copyTo(gray_source_image);
	cv::cvtColor(oriBgrImage_8UC3,gray_source_image,CV_BGR2GRAY);
	//const String filepath = "./resource/CharacterClassifier";
	CharacterClassifier=cv::Algorithm::load<RTrees>("D:\\Lewis\\projects\\container_ocr\\ctpn_vs2015_py36\\windows\\classification\\textDetection/resource/CharacterClassifier");
	featureExtract();
	swtprocess();
	Filter();
	return candidateStore;
}

void GetCandidate::featureExtract()
{
	strVectorStore.clear();
	vector<cv::Rect> rects;
	//mser->detectRegions(this->gray_source_image, this->strVectorStore,rects);
	mser(this->gray_source_image, this->strVectorStore);
	Mat img = this->gray_source_image;
	//int imgrows = this->gray_source_image;
	//int imgcols = this->gray_source_image.cols;
	Mat showimg;
	this->oriBgrImage_8UC3.copyTo(showimg);
	Mat demoimg(img.rows, img.cols, CV_8UC1);
	for (auto points : this->strVectorStore) {
		int minx = 9999, maxx = 0;
		int miny = 9999, maxy = 0;
		for (auto point : points) {
			if (point.y < miny)
				miny = point.y;
			if (point.y > maxy)
				maxy = point.y;
			if (point.x < minx)
				minx = point.x;
			if (point.x > maxx)
				maxx = point.x;
			demoimg.at<char>(point.y, point.x) = 255;
		}
		Rect rect = Rect(cv::Point2i(minx, miny), cv::Point2i(maxx, maxy));
		rectangle(showimg, rect, cv::Scalar(213, 188, 38),2);
	}
	cv::imshow("mser2", showimg);
	cv::imshow("mser", demoimg);
	cv::waitKey(0);
}

void GetCandidate::swtprocess()
{
	swt.initialize(this->gray_source_image,this->strVectorStore); 
}

void GetCandidate::ExtractCCfeatures()
{
	ccStore.clear();
	int Num = strVectorStore.size();
	Candidate cd;
	for(int i= 0; i < Num ;i++)
	{
		cd.pointsNum = strVectorStore[i].size();
		cd.max_i = strVectorStore[i][0].y;
		cd.min_i = strVectorStore[i][0].y;
		cd.max_j = strVectorStore[i][0].x; 
		cd.min_j = strVectorStore[i][0].x; 
		float temp = 0;
		float tempr = 0;
		float tempg = 0;
		float tempb = 0;
		for(int j = 1;j < cd.pointsNum;j++)
		{
			//get min i, min j, max i and max j
			if(strVectorStore[i][j].y > cd.max_i)
				cd.max_i = strVectorStore[i][j].y;
			else if(strVectorStore[i][j].y < cd.min_i)
				cd.min_i = strVectorStore[i][j].y;

			if(strVectorStore[i][j].x > cd.max_j)
				cd.max_j = strVectorStore[i][j].x;
			else if(strVectorStore[i][j].x < cd.min_j)
				cd.min_j = strVectorStore[i][j].x;

			//temp is the sum of all pixel values
			temp  +=  this->gray_source_image.at<unsigned char>(strVectorStore[i][j].y, strVectorStore[i][j].x);
			//tempr is the sum of green values of all pixels
			tempr +=  this->oriBgrImage_8UC3.at<cv::Vec3b>(strVectorStore[i][j].y, strVectorStore[i][j].x)[0];
			tempg +=  this->oriBgrImage_8UC3.at<cv::Vec3b>(strVectorStore[i][j].y, strVectorStore[i][j].x)[1];
			tempb +=  this->oriBgrImage_8UC3.at<cv::Vec3b>(strVectorStore[i][j].y, strVectorStore[i][j].x)[2];
		}
		cd.middle_i = (cd.max_i + cd.min_i)/2;
		cd.middle_j = (cd.max_j + cd.min_j)/2;
		cd.avergecolor = temp / cd.pointsNum;
		cd.averarer = tempr / cd.pointsNum;
		cd.averareg = tempg / cd.pointsNum;
		cd.averareb = tempb / cd.pointsNum;
		cd.index = i;
		cd.alive = true;
		cd.high = cd.max_i - cd.min_i;
		cd.width = cd.max_j - cd.min_j;
		swt.getStrokeWidth(strVectorStore[i],cd);
	    float var = 0;
		for(int j = 0; j< cd.pointsNum;j++)
			var += (gray_source_image.at<unsigned char>(strVectorStore[i][j].y, strVectorStore[i][j].x) - cd.avergecolor)*(gray_source_image.at<unsigned char>(strVectorStore[i][j].y, strVectorStore[i][j].x) - cd.avergecolor);
		cd.colorVariance = var/cd.pointsNum;
		cd.highWidthRatio = double(cd.max_i - cd.min_i)/(cd.max_j - cd.min_j);
		float area = (cd.max_i - cd.min_i)*(cd.max_j - cd.min_j);
		cd.grayRatio = float(cd.pointsNum)/(area + 1);
		ccStore.push_back(cd);
	}
}

void GetCandidate::Filter()
{
	ExtractCCfeatures();
	candidateStore.clear();
	for (int i = 0; i < ccStore.size(); i++)
	{
		Candidate cd = ccStore[i];
		cv::Mat rtVector = cv::Mat(1, 14, CV_32FC1);
		rtVector.at<float>(0, 0) = cd.pointsNum;
		rtVector.at<float>(0, 1) = cd.high;
		rtVector.at<float>(0, 2) = cd.width;
		rtVector.at<float>(0, 3) = cd.high / this->oriBgrImage_8UC3.rows;
		rtVector.at<float>(0, 4) = cd.width / this->oriBgrImage_8UC3.cols;
		rtVector.at<float>(0, 5) = cd.high / (cd.width + 1);
		rtVector.at<float>(0, 6) = cd.strokeVariance;
		rtVector.at<float>(0, 7) = cd.strokeWidh;
		rtVector.at<float>(0, 8) = cd.strokeWidthRatio;
		rtVector.at<float>(0, 9) = 0;
		rtVector.at<float>(0, 10) = cd.avergecolor;
		rtVector.at<float>(0, 11) = cd.colorVariance;
		rtVector.at<float>(0, 12) = abs(cd.middle_i - cd.high / 2);
		rtVector.at<float>(0, 13) = abs(cd.middle_j - cd.width / 2);
		try {
			Mat result;
			CharacterClassifier->getVotes(rtVector,result,0);
			const int* result_now = result.ptr<int>(1);
			
			printf("%d\n", result_now[0]);
			printf("%d\n", result_now[1]);
			if (result_now[1] > 14)
				candidateStore.push_back(cd);
		}
		catch (Exception e) {
			printf("ohohoho\n");
		}
	}

	//show the filtered result
	Mat show_img = Mat(gray_source_image.rows, gray_source_image.cols, CV_8UC1);
	Mat show_img2;
	oriBgrImage_8UC3.copyTo(show_img2);
	vector<Rect> filtered_rects;
	for (auto cd : candidateStore) {
		Rect rect = Rect(cd.min_j, cd.min_i, cd.width, cd.high);
		filtered_rects.push_back(rect);
		rectangle(show_img2, rect, cv::Scalar(213, 188, 38), 2);
		for (auto point : strVectorStore[cd.index]) {
			show_img.at<unsigned char>(point.y, point.x) = 255;
		}
	}
	cv::imshow("filtered mser", show_img);
	cv::imshow("filtered rects", show_img2);
	cv::waitKey();
}

