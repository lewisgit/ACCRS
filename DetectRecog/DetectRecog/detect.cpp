#define DLL_GEN

//#include <caffe/caffe.hpp>
#include "getMSER.hpp"
#if 1
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <cmath>
#include <string>
#include <utility>
#include <vector>
#include<boost/property_tree/json_parser.hpp>
#include<boost/property_tree/ptree.hpp>
#include<boost/foreach.hpp>
#include "detect.h"

//#include <Python.h>
//#include <numpy/arrayobject.h>

#include "recallratio.h"
#include "pythonCall.h"
#ifdef _MSC_VER
namespace caffe {
	extern void PythonInitEmbeddedCaffeModule();
	
}
using caffe::PythonInitEmbeddedCaffeModule;
#endif
//#if defined(WITH_PYTHON_LAYER) && defined(_MSC_VER)

//#endif

#define SHOW_MSER


float detectRatio1=0;
float detectRatio2=0;
int containerBoxNum=0;
float detectRatio11 = 0;
float detectRatio22 = 0;
int containerBoxNum2 = 0;

float detectRatio111 = 0;
float detectRatio222 = 0;
int containerBoxNum3 = 0;

float para = 30.0;

float part1 = 4.5 / 16;
float part2 = 6.8 / 16;
float part3 = 13.2 / 16;
float part4 = 14.5 / 16;

float scale;
float scaley1, scaley2, scaley3;
using namespace boost::property_tree;

#if 1
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;
using cv::Rect;
using cv::Mat;
using std::min;
using std::max;

//class CTPN_Detector {
//private:
//	PyObject* ctpn_object;
//	PyObject* ctpn_init;
//	PyObject* ctpn_detect;
//public:
//	CTPN_Detector(string model_file, string trained_file, string py_filename) {
//		Py_Initialize();
//		ctpn_object = PyImport_Import(PyUnicode_FromString(py_filename.c_str()));
//		ctpn_init = PyObject_GetAttrString(ctpn_object, "init");
//		ctpn_detect = PyObject_GetAttrString(ctpn_object, "detect");
//
//		PyObject* ArgArray = PyTuple_New(2);
//		PyObject* modelfile_str = PyUnicode_FromString(model_file.c_str());
//		PyObject* trainedfile_str = PyUnicode_FromString(trained_file.c_str());
//		PyTuple_SetItem(ArgArray, 0, modelfile_str);
//		PyTuple_SetItem(ArgArray, 1, trainedfile_str);
//		PyObject* ctpnret = PyObject_CallObject(ctpn_init, ArgArray);
//	}
//
//	int detect(string& img_path, vector<Rect>& rects) {
//		PyObject* ArgArray = PyTuple_New(1);
//		PyObject* imgpath_str = PyUnicode_FromString(img_path.c_str());
//		PyTuple_SetItem(ArgArray, 0, imgpath_str);
//
//		PyObject* ctpnret = PyObject_CallObject(ctpn_detect, ArgArray);
//
//
//		if (PyList_Check(ctpnret)) {
//			int SizeOfList = PyList_Size(ctpnret);
//			for (int i = 0; i < SizeOfList; i++) {
//				PyArrayObject *ListItem = (PyArrayObject *)PyList_GetItem(ctpnret, i);
//				int Rows = ListItem->dimensions[0], columns = ListItem->dimensions[1];
//				for (int j = 0; j < Rows; j++) {
//					int x1 = *(float *)(ListItem->data + j * ListItem->strides[0] + 0 * ListItem->strides[1]);
//					int y1 = *(float *)(ListItem->data + j * ListItem->strides[0] + 1 * ListItem->strides[1]);
//					int x2 = *(float *)(ListItem->data + j * ListItem->strides[0] + 2 * ListItem->strides[1]);
//					int y2 = *(float *)(ListItem->data + j * ListItem->strides[0] + 3 * ListItem->strides[1]);
//					rects.push_back(Rect(x1, y1, x2 - x1, y2 - y1));
//				}
//				Py_DECREF(ListItem);
//			}
//		}
//		else {
//			return 0;
//		}
//		return 1;
//	}
//};
CTPN_Detector* ctpn_detector;









/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Classifier {
public:
	Classifier(const string& model_file,
		const string& trained_file,
		const string& mean_file,
		const string& label_file);

	std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

private:
	void SetMean(const string& mean_file);

	std::vector<float> Predict(const cv::Mat& img);

	void WrapInputLayer(std::vector<cv::Mat>* input_channels);

	void Preprocess(const cv::Mat& img,
		std::vector<cv::Mat>* input_channels);

private:
	shared_ptr<Net<float> > net_;
	cv::Size input_geometry_;
	int num_channels_;
	cv::Mat mean_;
	std::vector<string> labels_;
};

Classifier *classifier;

Classifier::Classifier(const string& model_file,
	const string& trained_file,
	const string& mean_file,
	const string& label_file) {
#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	Caffe::set_mode(Caffe::GPU);
#endif

	/* Load the network. */
	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(trained_file);

	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3 || num_channels_ == 1)
		<< "Input layer should have 1 or 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

	/* Load the binaryproto mean file. */
	SetMean(mean_file);

	/* Load labels. */
	std::ifstream labels(label_file.c_str());
	CHECK(labels) << "Unable to open labels file " << label_file;
	string line;
	while (std::getline(labels, line))
		labels_.push_back(string(line));

	Blob<float>* output_layer = net_->output_blobs()[0];
	CHECK_EQ(labels_.size(), output_layer->channels())
		<< "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
	const std::pair<float, int>& rhs) {
	return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
	std::vector<std::pair<float, int> > pairs;
	for (size_t i = 0; i < v.size(); ++i)
		pairs.push_back(std::make_pair(v[i], static_cast<int>(i)));
	std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

	std::vector<int> result;
	for (int i = 0; i < N; ++i)
		result.push_back(pairs[i].second);
	return result;
}

/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {
	std::vector<float> output = Predict(img);

	N = std::min<int>(labels_.size(), N);
	std::vector<int> maxN = Argmax(output, N);
	std::vector<Prediction> predictions;
	for (int i = 0; i < N; ++i) {
		int idx = maxN[i];
		predictions.push_back(std::make_pair(labels_[idx], output[idx]));
	}

	return predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
	BlobProto blob_proto;
	ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

	/* Convert from BlobProto to Blob<float> */
	Blob<float> mean_blob;
	mean_blob.FromProto(blob_proto);
	CHECK_EQ(mean_blob.channels(), num_channels_)
		<< "Number of channels of mean file doesn't match input layer.";

	/* The format of the mean file is planar 32-bit float BGR or grayscale. */
	std::vector<cv::Mat> channels;
	float* data = mean_blob.mutable_cpu_data();
	for (int i = 0; i < num_channels_; ++i) {
		/* Extract an individual channel. */
		cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
		channels.push_back(channel);
		data += mean_blob.height() * mean_blob.width();
	}

	/* Merge the separate channels into a single image. */
	cv::Mat mean;
	cv::merge(channels, mean);

	/* Compute the global mean pixel value and create a mean image
	* filled with this value. */
	cv::Scalar channel_mean = cv::mean(mean);
	mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
	Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(1, num_channels_,
		input_geometry_.height, input_geometry_.width);
	/* Forward dimension change to all layers. */
	net_->Reshape();

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);

	Preprocess(img, &input_channels);

	net_->Forward();

	/* Copy the output layer to a std::vector */
	Blob<float>* output_layer = net_->output_blobs()[0];
	const float* begin = output_layer->cpu_data();
	const float* end = begin + output_layer->channels();
	return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
* (one per channel). This way we save one memcpy operation and we
* don't need to rely on cudaMemcpy2D. The last preprocessing
* operation will write the separate channels directly to the input
* layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
	Blob<float>* input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

void Classifier::Preprocess(const cv::Mat& img,
	std::vector<cv::Mat>* input_channels) {
	/* Convert the input image to the input image format of the network. */
	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

	cv::Mat sample_resized;
	if (sample.size() != input_geometry_)
		cv::resize(sample, sample_resized, input_geometry_);
	else
		sample_resized = sample;

	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);

	cv::Mat sample_normalized;
	cv::subtract(sample_float, mean_, sample_normalized);

	/* This operation will write the separate BGR planes directly to the
	* input layer of the network because it is wrapped by the cv::Mat
	* objects in input_channels. */
	cv::split(sample_normalized, *input_channels);

	CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
		== net_->input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";
}


bool rectsCompare(const vector<Rect>& lhs, const vector<Rect>& rhs) {
	return lhs.size() > rhs.size();
}
bool rectXcompare(const Rect& lhs, const Rect& rhs) {
	return lhs.x < rhs.x;
}


//Now here is my code

#define IMAGE_MAX 255;

//string OUT_PATH = "E:\\OneDrive\\Lewis\\projects\\BoxNumberDetect\\demo\\numberRegion";





//string model_file = "E:\\OneDrive\\Lewis\\projects\\BoxNumberDetect\\demo\\model\\deploy.prototxt";
//string trained_file = "E:\\OneDrive\\Lewis\\projects\\BoxNumberDetect\\demo\\model\\exVersion_iter_30000.caffemodel";
//string mean_file = "E:\\OneDrive\\Lewis\\projects\\BoxNumberDetect\\demo\\model\\testMean.binaryproto";
//string label_file = "E:\\OneDrive\\Lewis\\projects\\BoxNumberDetect\\demo\\model\\labels.txt";
//
//
//string jsonfile = "D:\\caffe-master\\boxes\\boxes.json";



//MSER parameters
int thresDelta = 2;
int minArea = 50;
int maxArea = 200;


//Filter parameters
int minWidth = 10;
int minHeight = 20;

float scaleX = 0.1;
float scaleY = 0.1;

float maxRatio = 6;
float minRatio = 1;

float scaleRect;
float scalethres;
float fillThres;
CvScalar blue(255, 255, 255);
CvScalar red(255, 255, 255);
CvScalar green(255, 255, 255);

int maxY = 50;
int maxX = 10;

double sigma = 6, threshold = 5, amount = 1;


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


//classify character image
Prediction classify_char(Mat& img, Classifier& classifier, int type) {
	std::vector<Prediction> predictions = classifier.Classify(img);
#if 0
	/* Print the top N predictions. */
	for (size_t i = 0; i < predictions.size(); ++i) {
		Prediction p = predictions[i];
		std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
			<< p.first << "\"" << std::endl;
	}
#endif
	if (type == 1) {
		for (int i = 0; i<predictions.size(); i++) {
			char ch = predictions[i].first.at(0);
			if (ch >= 'A' && ch <= 'z')
				return predictions[i];
		}
	}
	if (type == 2) {
		for (int i = 0; i<predictions.size(); i++) {
			char ch = predictions[i].first.at(0);
			if (ch >= '0' && ch <= '9')
				return predictions[i];
		}
	}
	return predictions[0];
}

int minElemPos(vector<Prediction> predictions) {
	float minv = 100;
	int pos = 0;
	for (int i = 0; i < predictions.size(); i++) {
		float v = predictions[i].second;
		if (v < minv) {
			minv = v;
			pos = i;
		}
	}
	return pos;
}

string removeMoreChar(vector<Prediction> predictions, int maxn, bool isback) {
	string result = "";
	if (isback)
		result += "0|";
	else
		result += "1|";
	int count = 0;
	if (predictions.size() <= maxn) {
		for (auto pred : predictions) {
			result += pred.first;
			if (count == 10)
				result += "|";
			count++;
		}
		for (int i = predictions.size(); i < maxn; i++) {
			result += "?";
			count++;
		}
	}
	else {
		while (predictions.size() > maxn) {
			predictions.erase(predictions.begin() + minElemPos(predictions));
		}
		for (int i = 0; i < maxn; i++) {
			result += predictions[i].first;
			//result += "?";
			if (count == 10)
				result += "|";
			count++;
		}
	}
	return result;

}
string recogFront(Mat& frontimg, vector<vector<Rect>>& front_c, Classifier& classifier) {

	int frontn = front_c.size();

	vector<Prediction> recogFront;

	for (int i = 0; i < front_c[0].size(); i++) {
		Rect rect = front_c[0][i];
		//if(i<=3)
		//	recogFront.push_back(classify_char(frontimg(rect), classifier,1));
		//if (i >3)
		recogFront.push_back(classify_char(frontimg(rect), classifier, 0));
	}

	return removeMoreChar(recogFront, 11, false);

}
string recogBack(Mat& backimg, vector<vector<Rect>>& back_c, Classifier& classifier) {

	

	int backn = back_c.size();

	vector<Prediction> recogBack;

	if (backn == 2) {
		for (int i = 0; i < backn; i++) {
			for (int j = 0; j < back_c[i].size(); j++) {
				Rect rect = back_c[i][j];
				//if(backn ==3 && i==0)
				//recogBack.push_back(classify_char(backimg(rect), classifier,1));
				//if (backn == 3 && i == 1)
				//	recogBack.push_back(classify_char(backimg(rect), classifier, 2));
				if (backn == 2 && i == 0 && j <= 3)
					recogBack.push_back(classify_char(backimg(rect), classifier, 1));
				else if (backn == 2 && i == 0 && j > 3)
					recogBack.push_back(classify_char(backimg(rect), classifier, 2));
				else
					recogBack.push_back(classify_char(backimg(rect), classifier, 0));
			}
		}
	}
	else if (backn == 3) {
		for (int i = 0; i < back_c[1].size(); i++) {
			Rect rect = back_c[1][i];
			recogBack.push_back(classify_char(backimg(rect), classifier, 1));
		}
		for (int i = 0; i < back_c[0].size(); i++) {
			Rect rect = back_c[0][i];
			recogBack.push_back(classify_char(backimg(rect), classifier, 2));
		}
		for (int i = 0; i < back_c[2].size(); i++) {
			Rect rect = back_c[2][i];
			recogBack.push_back(classify_char(backimg(rect), classifier, 0));
		}
	}

	return removeMoreChar(recogBack, 15, true);
#if 0
	cv::imshow("img", backimg);
	cv::waitKey(0);
#endif
}
vector<string> recog(Mat& backimg, vector<vector<Rect>>& back_c, Mat& frontimg, vector<vector<Rect>>& front_c, Classifier& classifier) {
	int backn = back_c.size();
	int frontn = front_c.size();
	vector<string> result;
	vector<Prediction> recogBack;
	vector<Prediction> recogFront;
	if (backn == 2) {
		for (int i = 0; i < backn; i++) {
			for (int j = 0; j < back_c[i].size(); j++) {
				Rect rect = back_c[i][j];
				//if(backn ==3 && i==0)
				//recogBack.push_back(classify_char(backimg(rect), classifier,1));
				//if (backn == 3 && i == 1)
				//	recogBack.push_back(classify_char(backimg(rect), classifier, 2));
				if (backn == 2 && i == 0 && j <= 3)
					recogBack.push_back(classify_char(backimg(rect), classifier, 1));
				else if (backn == 2 && i == 0 && j > 3)
					recogBack.push_back(classify_char(backimg(rect), classifier, 2));
				else
					recogBack.push_back(classify_char(backimg(rect), classifier, 0));
			}
		}
	}
	else {
		for (int i = 0; i < back_c[1].size(); i++) {
			Rect rect = back_c[1][i];
			recogBack.push_back(classify_char(backimg(rect), classifier, 1));
		}
		for (int i = 0; i < back_c[0].size(); i++) {
			Rect rect = back_c[0][i];
			recogBack.push_back(classify_char(backimg(rect), classifier, 2));
		}
		for (int i = 0; i < back_c[2].size(); i++) {
			Rect rect = back_c[2][i];
			recogBack.push_back(classify_char(backimg(rect), classifier, 0));
		}
	}
	for (int i = 0; i < front_c[0].size(); i++) {
		Rect rect = front_c[0][i];
		//if(i<=3)
		//	recogFront.push_back(classify_char(frontimg(rect), classifier,1));
		//if (i >3)
		recogFront.push_back(classify_char(frontimg(rect), classifier, 0));
	}
	result.push_back(removeMoreChar(recogBack, 15, true));
	result.push_back(removeMoreChar(recogFront, 11, false));
#if 0
	cv::imshow("img", backimg);
	cv::waitKey(0);
#endif
	return result;

}

vector<vector<Rect>> findFrontNumbers(vector<Rect> rects, int xthres, int ythres, int heightthres);
vector<vector<Rect>> findBackNumbers(vector<Rect> rects, int xthres, int ythres, int heightthres);
vector<Rect> removeRedun(vector<Rect>& rects);
vector<vector<Rect>> findRightRegions(vector<vector<Rect>> clusters);
Rect mymergeRects(vector<Rect> rects);
void scaleSingleRect(Rect& rect, float scalex, float scaley, int maxx, int maxy);
vector<Rect> removeOverlap(vector<Rect> rects);
vector<Rect> combineCTPNandCV(vector<vector<Rect>>& clusters, vector<Rect> ctpn_boxes);



vector<vector<Rect>> fillHoles(vector<vector<Rect>> clusters, int thres) {
	vector<vector<Rect>> result;
	if (clusters.size() == 2) {
		int count = 0;
		vector<Rect> cluster1;
		for (int i = 0; i < clusters[0].size() - 1; i++) {
			cluster1.push_back(clusters[0][i]);
			int dist = abs(clusters[0][i + 1].x - clusters[0][i].x - clusters[0][i].width);
			if (dist > thres && count != 3 && count != 9) {
				int width = (clusters[0][i + 1].width>clusters[0][i].width) ? clusters[0][i + 1].width : clusters[0][i].width;
				float missrate = 0.85*dist / width;
				int missnum = floor(missrate);
				if (missrate - missnum > 0.5) {
					missnum++;
				}
				float newwidth = 0.85*dist / missnum;
				float start = clusters[0][i].x + clusters[0][i].width;
				int y = clusters[0][i].y;
				int height = clusters[0][i].height;
				for (int i = 0; i < missnum; i++) {
					count++;

					Rect rect(start + newwidth*0.2 + newwidth*i, y, newwidth, height);
					cluster1.push_back(rect);
					if (count == 3 || count == 9)
						break;
				}

			}
			count++;
		}
		cluster1.push_back(clusters[0][clusters[0].size() - 1]);
		result.push_back(cluster1);
		result.push_back(clusters[1]);
		return result;
	}
	else {
		return clusters;
	}
}




//judge which image is the front sidett
bool judgeFrontBack(Mat gray1, Mat gray2) {



	vector<Rect> boxes1 = getMSER(gray1, thresDelta, 20, maxArea);
	vector<Rect> boxes2 = getMSER(gray2, thresDelta, 20, maxArea);
	boxes1 = removeRedun(boxes1);
	boxes2 = removeRedun(boxes2);


	vector<Rect> candidates1;
	//filter out wrong regions
	for (auto rect : boxes1) {
		float ratio = 1.0 * rect.height / rect.width;
		if (rect.x > gray1.cols*0.1 && rect.x<gray1.cols*0.9 && rect.y>gray1.rows*0.1) {
			candidates1.push_back(rect);
			//rectangle(img, rect, blue);
		}
	}
	vector<Rect> candidates2;
	for (auto rect : boxes2) {
		float ratio = 1.0 * rect.height / rect.width;
		if (rect.x > gray2.cols*0.1 && rect.x<gray2.cols*0.9  && rect.y>gray2.rows*0.1) {
			candidates2.push_back(rect);
			//rectangle(img, rect, blue);
		}
	}
#if 0
	for (auto rect : candidates1)
		rectangle(gray1, rect, red);
	for (auto rect : candidates2)
		rectangle(gray2, rect, red);
#endif
#if 0
	cv::imshow("ad", gray1);
	cv::imshow("ads", gray2);
	cvWaitKey(0);
#endif 
	if (candidates1.size()>candidates2.size()) {

		return true;
	}
	else {

		return false;
	}

}

float overlapRatio(Rect& rect1, Rect& rect2) {
	if (isOverlap(rect1, rect2)) {
		float width = min(rect1.x + rect1.width, rect2.x + rect2.width) - max(rect1.x, rect2.x);
		float height= min(rect1.y + rect1.height, rect2.y + rect2.height) - max(rect1.y, rect2.y);
		
		float overlap_area = width*height;
		float base_area = rect2.width*rect2.height;

		return overlap_area / base_area;
	}
	else {
		return 0.0;
	}
}
bool geometryFilter(Rect rect) {
	float hwRatio = 1.0*rect.height / rect.width;
	if (hwRatio<maxRatio && hwRatio>minRatio && rect.height > minHeight) 
		return true;
	else return false;
}

void MSERFilter(vector<Rect>& mser_rects, vector<Rect>& ctpn_rects, vector<vector<Rect>>& filtered_clusters, float overlap_thres) {
	for (auto ctpn_rect : ctpn_rects) {
		vector<Rect> rects;
		for (auto mser_rect : mser_rects) {
			if (overlapRatio(ctpn_rect,mser_rect) > overlap_thres && geometryFilter(mser_rect)) {
				rects.push_back(mser_rect);
			}
		}
		filtered_clusters.push_back(rects);
	}
}



vector<vector<Rect>> findBackNumberRegion(string img_path, Mat img, Mat greyImg, Classifier& classifier) {

	//ctpn detection
	vector<Rect> ctpn_rects;
	ctpn_detector->detect(img_path, ctpn_rects);
	if (!ctpn_detector->detect(img_path, ctpn_rects)) {
		std::cout << "CTPN detection failed." << std::endl;
	}

	//find MSER
	vector<Rect> posbox = getMSER(greyImg, thresDelta, minArea, maxArea);

	//filter rects by ctpn detection
	vector<vector<Rect>> filtered_clusters;
	MSERFilter(posbox,ctpn_rects,filtered_clusters,0.5);
	
	//for (int i = 0; i < filtered_clusters.size(); i++)
		//filtered_clusters[i] = removeRedun(filtered_clusters[i]);
	string imgsavepath = "D:\\OneDrive\\Lewis\\GraduationAffairs\\imgs\\";
#if 1
	Mat showimg;
	img.copyTo(showimg);
	for (auto cluster : filtered_clusters) {
		CvScalar mycolor=red;

		for (auto rect : cluster) {
			rectangle(showimg, rect, mycolor);
		}
	}
	//for (auto rect : ctpn_rects) {
		//rect.x = rect.x - 10;
		//rect.height = rect.height*1.1;
	//	rectangle(showimg, rect, blue);
//	}
	cv::imwrite(imgsavepath + "pic1.jpg", showimg);
	cv::imshow("MSER filter phase 1", showimg);
	cv::waitKey(0);
	//for (auto rect : ctpn_rects) {
	//	Mat strimg;
	//	cv::cvtColor(img(rect), strimg, cv::COLOR_RGB2GRAY);
	//	Mat strimg_thres;
	//	cv::threshold(strimg, strimg_thres, 10, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
	//	cv::imshow("detected strings", strimg_thres);
	//	cv::waitKey(0);
	//}

	//cv::waitKey(0);
	
#endif
	

	//Mat showimg;


	img.copyTo(showimg);
	for (auto rect : ctpn_rects) {
		//rect.x = rect.x - 10;
		//rect.height = rect.height*1.1;
		rectangle(showimg, rect, blue);
	}
	cv::imwrite(imgsavepath + "pic2.jpg", showimg);
	cv::imshow("pic", showimg);
	cv::waitKey(0);

	img.copyTo(showimg);
	for (auto rect : posbox)
		rectangle(showimg, rect, red);
	cv::imwrite(imgsavepath + "pic3.jpg", showimg);
	cv::imshow("pic1", showimg);
	cv::waitKey(0);

	img.copyTo(showimg);

	//remove unecessary boxes
	vector<Rect> candidates;
	if (img.rows > img.cols) {
		for (auto rect : posbox) {
			float ratio = 1.0 * rect.height / rect.width;
			if (ratio<maxRatio&& ratio>minRatio && rect.height > minHeight ) {
				candidates.push_back(rect);

				rectangle(showimg, rect, red);
			}
		}
	}
	else {
		for (auto rect : posbox) {
			float ratio = 1.0 * rect.height / rect.width;
			if (ratio<maxRatio&& ratio>minRatio && rect.height > minHeight) {
				candidates.push_back(rect);

				rectangle(showimg, rect, red);
			}
		}
	}
	cv::imwrite(imgsavepath + "pic4.jpg", showimg);
	cv::imshow("pic2", showimg);
	cv::waitKey(0);


	//candidates = removeRedun(candidates);

	//find characters
	vector<vector<Rect>> clusters = findBackNumbers(candidates, 50, 10, 20);


	img.copyTo(showimg);
	for (int i = 0; i < clusters.size(); i++){
		int nn = clusters.size();
		int color2 = rand() * 255 / RAND_MAX;
		CvScalar mycolor((255-i)*255/nn, color2, color2);
		for (auto rect : clusters[i]) {
			rectangle(showimg, rect, mycolor);
		}
		//clusters[i] = removeOverlap(clusters[i]);
	}
	cv::imwrite(imgsavepath + "pic5.jpg", showimg);
	cv::imshow("pic3", showimg);
	cv::waitKey(0);

	vector<vector<Rect>> clusters2 = findRightRegions(clusters);

	for (int i = 0; i < clusters2.size(); i++) {
		clusters2[i] = removeOverlap(clusters2[i]);
	}


	//std::cout << "cluster size: " << clusters2.size() << std::endl;
#if 0
	vector<vector<Rect>> clusters3;
	if (clusters2.size() == 2) {
		vector<Rect> cluster1;
		Rect rect = mymergeRects(clusters2[0]);
		//rectangle(greyImg, rect, 2);
		float halfwidth = rect.width*part1 / 8;
		float start1 = rect.x;
		//float scale = 0.3;
		//float scaley = 0.8;
		for (int i = 0; i < 4; i++) {
			Rect srect(start1 + (i * 2 - scale)*halfwidth, rect.y, 2 * halfwidth*(1 + scale), rect.height*scaley1);
			cluster1.push_back(srect);
			//rectangle(greyImg, srect, 1);
		}

		//scaley = 0.8;
		float start2 = rect.width*part2 + rect.x;
		halfwidth = rect.width*(part3 - part2) / 12;
		for (int i = 0; i < 6; i++) {
			Rect srect(start2 + (i * 2 - scale) * halfwidth, rect.y, 2 * halfwidth*(1 + scale), rect.height*scaley2);
			//rectangle(greyImg, srect, 1);
			cluster1.push_back(srect);
			//cluster1.push_back(srect);
		}
		halfwidth = rect.width*(1 - part4) / 2;
		Rect srect(rect.x + rect.width*part4 - scale*halfwidth, rect.y, rect.width*(1 - part4)*(1 + scale), rect.height*scaley3);
		//rectangle(greyImg, srect, 1);
		cluster1.push_back(srect);

		float scale2 = 1.1;
		vector<Rect> cluster2;
		Rect rect2 = mymergeRects(clusters2[1]);
		halfwidth = scale2*rect2.width / 8;
		for (int i = 0; i < 4; i++) {
			Rect srect(rect2.x + (i * 2 - scale) * halfwidth, rect2.y, 2 * halfwidth*(1 + scale), rect2.height);
			//rectangle(greyImg, srect, 1);
			cluster2.push_back(srect);
		}
		clusters3.push_back(cluster1);
		clusters3.push_back(cluster2);

	}
	//cv::imshow("ratio", img);
	//cv::imshow("asd", greyImg);
	//cv::waitKey(0);
#endif
	//read boxes from json file
	//ptree root, backbox;

	//vector<Rect> ctpn_boxes;
	//read_json<ptree>(jsonfile, root);
	//backbox = root.get_child("back");
	//for (ptree::iterator it = backbox.begin(); it != backbox.end(); ++it) {
	//	
	//	string key = it->first;
	//	int top = it->second.get<int>("top");
	//	int bottom = it->second.get<int>("bottom");
	//	int left = it->second.get<int>("left");
	//	int right = it->second.get<int>("right");

	//	Rect rect(left,top,right-left,bottom-top);
	//	rect.x = rect.x + img.cols*0.5;
	//	ctpn_boxes.push_back(rect);
	//	//std::cout << "top :" << top << " bottom: " << bottom << " left: " << left << " right: " << right << std::endl;
	//}
	//vector<Rect> finalboxes = combineCTPNandCV(clusters2,ctpn_boxes);

	//if (finalboxes.size() == 2) {
	//	
	//}


#if 0
	//merge the regions
	vector<Rect> finalcan;
	std::cout << clusters2.size() << std::endl;
	for (int i = 0; i<clusters2.size(); i++)
		for (int j = 0; j < clusters2[i].size(); j++) {
			finalcan.push_back(clusters2[i][j]);
			rectangle(greyImg, clusters2[i][j], red);
			//classify_char(img(clusters2[i][j]),classifier);
		}

	Rect mrects = mymergeRects(finalcan);

	scaleSingleRect(mrects, 0.15, 0.15, img.cols, img.rows);
	//imwrite(OUT_PATH + "/" + "back.jpg", greyImg(mrects));



	//rectangle(greyImg, mrects, red, 2);
	//imshow("filter", img);
	//imshow("candidate", greyImg);
	//cvWaitKey(0);
#endif

	for (auto cluster : clusters2)
		for (auto rect : cluster) {
			rectangle(greyImg, rect, red);
		}

	clusters2 = fillHoles(clusters2, fillThres);

	for (int i = 0; i < clusters2.size(); i++)
		for (int j = 0; j < clusters2[i].size(); j++) {
			float ratio = 1.0*clusters2[i][j].height / clusters2[i][j].width;
			if (ratio<scalethres)
				scaleSingleRect(clusters2[i][j], scaleRect, scaleRect, img.cols, img.rows);
			else
				scaleSingleRect(clusters2[i][j], scaleRect, 0, img.cols, img.rows);
		}

#if 0
	imshow("filter", greyImg);
	//imshow("candidate", greyImg);
	cvWaitKey(0);
#endif
	return clusters2;

}

vector<vector<Rect>> findBackNumberRegion2(string img_path, Mat img, Mat greyImg, Classifier& classifier,string labelpath) {

	//ctpn detection
	vector<Rect> ctpn_rects;
	ctpn_detector->detect(img_path, ctpn_rects);
	if (!ctpn_detector->detect(img_path, ctpn_rects)) {
		std::cout << "CTPN detection failed." << std::endl;
	}

	ctpn_rects = removeRedun(ctpn_rects);

	//find MSER
	vector<Rect> posbox = getMSER(greyImg, thresDelta, minArea, maxArea);

	//filter rects by ctpn detection
	vector<vector<Rect>> filtered_clusters;
	MSERFilter(posbox, ctpn_rects, filtered_clusters, 0.5);

	for (int i = 0; i < filtered_clusters.size(); i++)
		filtered_clusters[i] = removeRedun(filtered_clusters[i]);


		
	std::sort(filtered_clusters.begin(), filtered_clusters.end(), rectsCompare);
	for (int i = 0; i < filtered_clusters.size(); i++) {
		std::sort(filtered_clusters[i].begin(), filtered_clusters[i].end(), rectXcompare);
		for (int j = 0; j < filtered_clusters[i].size(); j++) {
			scaleSingleRect(filtered_clusters[i][j], 0.2, 0.1, img.cols, img.rows);
		}
	}

#ifdef SHOW_MSER
	Mat showimg;
	img.copyTo(showimg);
	for (auto cluster : filtered_clusters) {
		std::cout << "size: " << cluster.size() << std::endl;
		for (auto rect : cluster) {
			rectangle(showimg, rect, red);
		}
	}
	for (int k = 0; k < ctpn_rects.size(); k++) {

		rectangle(showimg, ctpn_rects[k], blue);
		string text = std::to_string(k);
		cv::putText(showimg, text, cv::Point(ctpn_rects[k].x, ctpn_rects[k].y), cv::FONT_HERSHEY_PLAIN,2,red,2);
	}
	cv::imshow("MSER filter phase 1", showimg);
	cv::waitKey(0);

	vector<Rect> eval_rects;
	vector<vector<Rect>> eval_rects2;
	int numofeval;
	std::cin >> numofeval;
	for (int k = 0; k < numofeval; k++) {
		int ind;
		std::cin >> ind;
		eval_rects.push_back(ctpn_rects[ind]);
		eval_rects2.push_back(filtered_clusters[k]);
	}
	vector<Rect> refined_rects;
	for (auto rects : eval_rects2) {
		int minx = 10000;
		int miny = 10000;
		int maxx = 0;
		int maxy = 0;
		for (auto rect : rects) {
			if (rect.x < minx)
				minx = rect.x;
			if (rect.x + rect.width > maxx)
				maxx = rect.x + rect.width;
			if (rect.y < miny)
				miny = rect.y;
			if (rect.y + rect.height > maxy)
				maxy = rect.y + rect.height;
		}
		Rect rect(minx,miny,maxx-minx,maxy-miny);
		rectangle(showimg, rect, green);
		refined_rects.push_back(rect);
	}
	calRatio1(eval_rects, labelpath, &detectRatio1, &detectRatio2, &containerBoxNum);
	calRatio2(eval_rects2, labelpath, &detectRatio11, &detectRatio22, &containerBoxNum2);

	calRatio1(refined_rects, labelpath, &detectRatio111, &detectRatio222, &containerBoxNum3);
	/*for (auto rect : ctpn_rects) {
	Mat strimg;
	cv::cvtColor(img(rect), strimg, cv::COLOR_RGB2GRAY);
	Mat strimg_thres;
	cv::threshold(strimg, strimg_thres, 10, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
	cv::imshow("detected strings", strimg_thres);
	cv::waitKey(0);
	}*/

	//cv::waitKey(0);

#endif
	vector<vector<Rect>> result;
	if (filtered_clusters[0].size() > 7) {
		result.push_back(filtered_clusters[0]);
		result.push_back(filtered_clusters[1]);
	}
	else {
		result.push_back(filtered_clusters[0]);
		result.push_back(filtered_clusters[1]);
		result.push_back(filtered_clusters[2]);
	}
	return result;
}


vector<vector<Rect>> splitMerge(vector<vector<Rect>> clusters, int thres) {
	vector<vector<Rect>> result;
	vector<Rect> cluster;
	int n = clusters[0].size();
	for (auto rect : clusters[0]) {
		if (rect.height > thres) {
			float height = rect.height;
			Rect rect1(rect.x, rect.y, rect.width, height / 2.0);
			Rect rect2(rect.x, rect.y + height / 2.0, rect.width, height / 2.0);
			cluster.push_back(rect1);
			cluster.push_back(rect2);
		}
		else {
			cluster.push_back(rect);
		}

	}
	result.push_back(cluster);
	return result;
}
vector<Rect> rearrange(vector<Rect> rects, int thres) {
	int n = rects.size();
	int secPos = 0;
	vector<Rect> cluster1;
	for (int i = 0; i < n - 1; i++) {
		cluster1.push_back(rects[i]);
		if (rects[i + 1].y - rects[i].y - rects[i].height > thres) {
			secPos = i + 1;

			break;
		}

	}
	vector<Rect> result;
	if (cluster1.size() == 0)
		return rects;
	Rect merge1 = mymergeRects(cluster1);
	float height = merge1.height / 4.0;
	for (int i = 0; i < 4; i++) {
		Rect rect(merge1.x, merge1.y + i*height, merge1.width, height);
		result.push_back(rect);
	}
	vector<Rect> cluster2;
	for (int i = secPos; i < n - 1; i++) {
		cluster2.push_back(rects[i]);
		if (rects[i + 1].y - rects[i].y - rects[i].height > thres) {
			//secPos = i + 1;

			break;
		}
	}
	if (cluster2.size() > 0) {
		Rect merge2 = mymergeRects(cluster2);
		height = merge2.height / 6.0;
		for (int i = 0; i < 6; i++) {
			Rect rect(merge2.x, merge2.y + i*height, merge2.width, height);
			result.push_back(rect);

		}
		result.push_back(rects[n - 1]);
	}
	else {
		vector<Rect> retVal;
		return retVal;
	}
	return result;
}

vector<vector<Rect>> findFrontNumberRegion(Mat img, Mat greyImg, Classifier& classifier) {

	//mser

	vector<Rect> posbox = getMSER(greyImg, thresDelta, minArea, maxArea);



	vector<Rect> candidates;
	//filter out wrong regions

	if (img.rows > img.cols) {
		for (auto rect : posbox) {
			float ratio = 1.0 * rect.height / rect.width;
			if (ratio<maxRatio&& ratio>minRatio && rect.height > minHeight && rect.x > img.cols*0.5) {
				candidates.push_back(rect);
				//rectangle(img, rect, blue);
			}
		}
	}
	else {
		for (auto rect : posbox) {
			float ratio = 1.0 * rect.height / rect.width;
			if (ratio<maxRatio&& ratio>minRatio && rect.height > minHeight) {
				candidates.push_back(rect);
				//rectangle(img, rect, blue);
			}
		}
	}

	candidates = removeRedun(candidates);

	vector<vector<Rect>> clusters = findFrontNumbers(candidates, 10, 40, 100);

	float minheight = 1000;
	for (auto rect : clusters[0]) {
		if (rect.height < minheight)
			minheight = rect.height;
	}
	vector<vector<Rect>> clusters2;
	if (clusters[0].size() > 11) {

		clusters2.push_back(rearrange(clusters[0], 15));
	}
	else {
#if 1

		clusters2 = splitMerge(clusters, minheight*1.5);

		for (int j = 0; j < clusters2[0].size(); j++) {
			float ratio = 1.0*clusters2[0][j].height / clusters2[0][j].width;
			if (ratio < scalethres)
				scaleSingleRect(clusters2[0][j], scaleRect*1.1, scaleRect, img.cols, img.rows);
			else
				scaleSingleRect(clusters2[0][j], scaleRect*1.1, 0, img.cols, img.rows);
		}
#endif 
	}

#if 0
	for (int i = 0; i < clusters2[0].size(); i++) {
		rectangle(greyImg, clusters2[0][i], red);
	}
	for (int i = 0; i < clusters[0].size(); i++) {
		rectangle(img, clusters[0][i], red);
	}

	Rect mrects = mymergeRects(clusters[0]);
	scaleSingleRect(mrects, 0.15, img.cols, img.rows);
	imwrite(OUT_PATH + "/" + "front.jpg", greyImg(mrects));

	rectangle(greyImg, mrects, red, 2);
	imshow("candidate", greyImg);
	imshow("result", img);
	cvWaitKey(0);
	imshow("result", greyImg);
	cvWaitKey(0);
#endif

	return clusters2;
}






void saveBackFrontImg(Mat image1, Mat image2) {
	imwrite("judge/back.png", image1);
	imwrite("judge/front.png", image2);
}

bool box_detect(char* output1, char* output2, Classifier& classifier, string input1, string input2) {


	//load images and convert to gray scale
	Mat image1 = cv::imread(input1);
	Mat image2 = cv::imread(input2);
	Mat gray1, gray2;
	cvtColor(image1, gray1, CV_BGR2GRAY);
	cvtColor(image2, gray2, CV_BGR2GRAY);

	vector<string> result;
	if (judgeFrontBack(gray1, gray2)) {
		//	saveBackFrontImg(image1, image2);
		result = recog(image1, findBackNumberRegion(input1,image1, gray1, classifier), image2, findFrontNumberRegion(image2, gray2, classifier), classifier);
		//std::cout << std::endl << "back detect: " << result[0] << std::endl;
		//std::cout << std::endl << "front detect: " << result[1] << std::endl;
		strcpy_s(output1, result[0].length() + 1, result[0].c_str());
		strcpy_s(output2, result[1].length() + 1, result[1].c_str());
		return true;
		//detstr.push_back();
	}
	else {
		//	saveBackFrontImg(image2, image1);
		result = recog(image2, findBackNumberRegion(input2,image2, gray2, classifier),
			image1, findFrontNumberRegion(image1, gray1, classifier), classifier);
		//std::cout << std::endl << "back detect: " << result[0] << std::endl;
		//std::cout << std::endl << "front detect: " << result[1] << std::endl;
		strcpy_s(output1, result[0].length() + 1, result[0].c_str());
		strcpy_s(output2, result[1].length() + 1, result[1].c_str());
		return false;
	}
}

void sortCluster(vector<vector<Rect>>& clusters) {
	for (int i = clusters.size() - 1; i >= 0; i--) {
		for (int j = 0; j<i; j++) {
			if (clusters[j].size()<clusters[j + 1].size()) {
				vector<Rect> tmp = clusters[j];
				clusters[j] = clusters[j + 1];
				clusters[j + 1] = tmp;
			}
		}
	}
}
bool sortByY(const Rect& lhs, const Rect& rhs) {
	if (lhs.y<rhs.y)
		return true;
	else return false;
}
bool sortByX(const Rect& lhs, const Rect& rhs) {
	if (lhs.x<rhs.x)
		return true;
	else return false;
}

bool rectXCompare(const Rect& lhs, const Rect& rhs) {
	if (lhs.x<rhs.x)
		return true;
	else return false;
}






bool isInside(vector<int> ai, vector<int> bi, vector<int> ci, vector<int> di, vector<int> aj, vector<int> bj, vector<int> cj, vector<int> dj) {
	int thres = 1;
	if (ai[0] - thres<aj[0] && ai[1] - thres<aj[1])
		if (bi[0] + thres>bj[0] && bi[1] - thres<bj[1])
			if (ci[0] - thres<cj[0] && ci[1] + thres>cj[1])
				if (di[0] + thres>dj[0] && di[1] + thres>dj[1])
					return true;
	return false;
}
int isContained(Rect recti, Rect rectj) {
	vector<int> ai, bi, ci, di;
	vector<int> aj, bj, cj, dj;
	ai.push_back(recti.x);            ai.push_back(recti.y);
	bi.push_back(recti.x + recti.width); bi.push_back(recti.y);
	ci.push_back(recti.x);            ci.push_back(recti.y + recti.height);
	di.push_back(recti.x + recti.width); di.push_back(recti.y + recti.height);

	aj.push_back(rectj.x);            aj.push_back(rectj.y);
	bj.push_back(rectj.x + rectj.width); bj.push_back(rectj.y);
	cj.push_back(rectj.x);            cj.push_back(rectj.y + rectj.height);
	dj.push_back(rectj.x + rectj.width); dj.push_back(rectj.y + rectj.height);

	if (isInside(ai, bi, ci, di, aj, bj, cj, dj))
		return 1;
	if (isInside(aj, bj, cj, dj, ai, bi, ci, di))
		return 2;
	return 0;

}
vector<Rect> removeRedun(vector<Rect>& rects) {
	vector<int> indices;
	vector<Rect> filtered;
	int thres = 200;
	for (int i = 0; i<rects.size(); i++)
		indices.push_back(0);
	for (int i = 0; i<rects.size(); i++) {
		for (int j = i + 1; j<rects.size(); j++) {
			if (abs(rects[i].x - rects[j].x)<thres && abs(rects[i].y - rects[j].y)<thres) {
				int ans = isContained(rects[i], rects[j]);
				if (ans == 1)
					indices[j] = 1;
				if (ans == 2)
					indices[i] = 1;
			}
		}
	}
	for (int i = 0; i<rects.size(); i++) {
		if (indices[i] == 0)
			filtered.push_back(rects[i]);
	}
	return filtered;
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
vector<vector<Rect>> findBackNumbers(vector<Rect> rects, int xthres, int ythres, int heightthres) {
	vector<vector<Rect>> clusters;
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
	sortCluster(clusters);
	//sort(clusters.begin(),clusters.end(),sortByClusterSize);
	return clusters;
}


vector<vector<Rect>> findFrontNumbers(vector<Rect> rects, int xthres, int ythres, int heightthres) {
	vector<vector<Rect>> clusters;
	vector<float> rectsx, rectsy, rectsheight;
	sort(rects.begin(), rects.end(), sortByY);
	for (auto rect : rects) {
		int index = isClusterExist(rect, clusters, rectsx, rectsy, rectsheight, xthres, ythres, heightthres);
		if (index == -1) {
			vector<Rect> newcluster;
			newcluster.push_back(rect);
			clusters.push_back(newcluster);
			float newx = rect.x*1.0 + rect.width / 2.0, newy = rect.y + rect.height;
			rectsx.push_back(newx);
			rectsy.push_back(newy);
			rectsheight.push_back(rect.height);
		}
		else {

			rectsy[index] = rect.y + rect.height;
			rectsx[index] = (clusters[index].size()*rectsx[index] + rect.x + rect.width / 2.0) / (clusters[index].size() + 1);
			rectsheight[index] = (clusters[index].size()*rectsheight[index] + rect.height) / (clusters[index].size() + 1);
			clusters[index].push_back(rect);
		}
	}
	sortCluster(clusters);
	//sort(clusters.begin(),clusters.end(),sortByClusterSize);
	return clusters;
}
bool isTwoRow(vector<vector<Rect>> clusters) {
	int y1 = clusters[0][0].y;
	for (int i = 1; i < clusters.size(); i++) {
		if (clusters[i][0].y < y1 && clusters[i].size() >= 4) {
			return false;
		}
	}
	return true;
}
vector<vector<Rect>> findRightRegions(vector<vector<Rect>> clusters) {
	vector<vector<Rect>> result;
	result.push_back(clusters[0]);

	int n = clusters[0].size();
	Rect prim = clusters[0][0];
	int thres = 100;
	if (isTwoRow(clusters)) {
		int index1;
		int dist = 10000;
		for (int i = 1; i<clusters.size(); i++) {
			Rect rect = clusters[i][0];
			int tmp = abs(rect.x - prim.x) + abs(rect.y - prim.y);
			//std::cout << clusters[i].size() << std::endl;
			if (tmp<dist && rect.y>prim.y && rect.x + 20>prim.x && clusters[i].size()>2) {
				dist = tmp;
				index1 = i;
			}
		}
		if (index1<clusters.size())
			result.push_back(clusters[index1]);
	}
	else {
		int index1, index2;
		int dist1 = 10000, dist2 = 10000;
		for (int i = 1; i<clusters.size(); i++) {
			Rect rect = clusters[i][0];
			int tmp = abs(rect.x - prim.x) + abs(rect.y - prim.y);
			if (tmp<dist1 && rect.y<prim.y && rect.x + 20>prim.x && clusters[i].size()>2) {
				dist1 = tmp;
				index1 = i;
			}
			if (tmp<dist2 && rect.y>prim.y && rect.x + 20>prim.x && clusters[i].size()>2) {
				dist2 = tmp;
				index2 = i;
			}
		}
		if (index1<clusters.size())
			result.push_back(clusters[index1]);
		if (index2<clusters.size())
			result.push_back(clusters[index2]);

	}
	return result;
}

//bool isOverlap(Rect rect1, Rect rect2) {
//	int x1 = rect1.x, y1 = rect1.x + rect1.width;
//	int x2 = rect2.x, y2 = rect2.x + rect2.width;
//	if (x1>x2 && x1<y2)
//		return true;
//	if (y1>x2 && y1<y2)
//		return true;
//	return false;
//}
vector<Rect> removeOverlap(vector<Rect> rects) {
	int tag[20];
	for (int i = 0; i<rects.size(); i++)
		tag[i] = 1;
	for (int i = 0; i<rects.size(); i++) {
		for (int j = i + 1; j<rects.size(); j++) {
			Rect one = rects[i], two = rects[j];
			if (isOverlap(one, two)) {
				float arearatio = 1.0*one.width*one.height / (two.width*two.height);
				if (one.height>two.height && arearatio>0.3 && arearatio<3.3)
					tag[i] = 0;
				else if (one.height<two.height && arearatio>0.3 && arearatio<3.3)
					tag[j] = 0;
				else if (one.width*one.height < (two.width*two.height))
					tag[i] = 0;
				else tag[j] = 0;
			}
		}
	}
	vector<Rect> result;
	for (int i = 0; i<rects.size(); i++) {
		if (tag[i])
			result.push_back(rects[i]);
	}
	return result;
}




Rect mymergeRects(vector<Rect> rects) {

	Rect out = rects[0];
	for (int i = 1; i<rects.size(); i++) {
		out = out | rects[i];
	}
	return out;
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


bool isCorrectBox(vector<vector<Rect>>& clusters, Rect rect) {
	for (vector<Rect> cluster : clusters)
		for (Rect rect1 : cluster)
			if (isOverlap(rect1, rect))
				return true;
	return false;

}

vector<Rect> combineCTPNandCV(vector<vector<Rect>>& clusters, vector<Rect> ctpn_boxes) {
	vector<Rect> result;
	for (Rect rect : ctpn_boxes) {
		if (isCorrectBox(clusters, rect))
			result.push_back(rect);
	}
	return result;

}

void init(char* model_file, char* trained_file, char* mean_file, char* label_file) {

	scale = 0.6;
	scaley1 = 0.8;
	scaley2 = 0.8;
	scaley3 = 0.8;
	maxArea = 300;
	minRatio = 1.2;
	scaleRect = 0.2;
	scalethres = 4;
	fillThres = 10;

	classifier = new Classifier(model_file, trained_file, mean_file, label_file);
}

void uninit() {
	free(classifier);
}

void __frontDetect(char* output, char* input) {
	Mat image = cv::imread(input);
	Mat gray;
	cvtColor(image, gray, CV_BGR2GRAY);
	string result = recogFront(image, findFrontNumberRegion(image, gray, *classifier), *classifier);
	strcpy_s(output, result.length() + 1, result.c_str());
}

void __backDetect(char* output, char* input) {
	Mat image = cv::imread(input);
	Mat gray;
	cvtColor(image, gray, CV_BGR2GRAY);
	string result = recogBack(image, findBackNumberRegion(input,image, gray, *classifier), *classifier);
	strcpy_s(output, result.length() + 1, result.c_str());
}


#if 0
bool detect(char* output1, char* output2, char* input1, char* input2) {
	//Classifier classifier(model_file, trained_file, mean_file, label_file);
	//string input1 = "E:\\OneDrive\\Lewis\\projects\\BoxNumberDetect\\demo\\select\\first30.png";
	//string input2 = "E:\\OneDrive\\Lewis\\projects\\BoxNumberDetect\\demo\\select\\second11.png";

	if (output1 == NULL) output1 = new char[1024];
	if (output2 == NULL) output2 = new char[1024];


	scale = 0.6;
	scaley1 = 0.8;
	scaley2 = 0.8;
	scaley3 = 0.8;
	maxArea = 300;
	minRatio = 1.2;
	scaleRect = 0.2;
	scalethres = 4;
	fillThres = 10;

	return box_detect(output1, output2, *classifier, input1, input2);


#else

#include "other.h"

void initCTPN() {
	string model_path = "E:\\container_ocr\\CTPN\\models\\";
	string model_file = model_path + "deploy.prototxt";
	string trained_file = model_path + "ctpn_trained_model.caffemodel";
	string img_path = "E:\\container_ocr\\CTPN\\demo_images\\img_1.jpg";
	CaffeModel caffeModel(model_file, trained_file);

	Mat img = cv::imread(img_path);
	caffeModel.forward2(img);
}

void testpy()
{
	Py_Initialize(); // 初始化python虚拟机
	int val;
	char *retStr = "";
	
	//PyRun_SimpleString("import sys");
	//PyRun_SimpleString("sys.path.append('E:\\container_ocr\\ctpn_vs2015_py36\\Build\\x64\\Release')");
	//PyRun_SimpleString("a=1");
	//PyObject* name = PyBytes_FromString("mytest");

	string file_name;


	//PyTuple_SetItem();

	string im_path = "E:\\container_ocr\\detected_imgs\\122_0002_170929_091901_06_rear_1_1.jpg";
	Mat img = cv::imread(im_path);
	npy_intp dims[1] = { img.rows*img.cols*img.channels() };
	npy_intp dim_dim[1] = { 3 };
	int dim_data[3] = { img.rows, img.cols, img.channels() };
	//PyObject* PyArray=PyArray_SimpleNewFromData(1,dims,NPY_INT,(int *)img.data);
	//PyObject* dim_info = PyArray_SimpleNewFromData(1, dim_dim, NPY_INT, dim_data);
	PyObject* ArgArray = PyTuple_New(1);
	PyObject* py_impath = PyUnicode_FromString(im_path.c_str());
	/*if (ArgArray == nullptr)
	{
		PyErr_Print();  
		system("pause");
	}*/
	PyTuple_SetItem(ArgArray, 0, py_impath);
	//PyTuple_SetItem(ArgArray, 1, dim_info);
	//Mat img;
	//img.data
	PyObject* ctpndemo = PyImport_Import(PyUnicode_FromString("demo"));
	if (ctpndemo == nullptr)
	{
		PyErr_Print();
		system("pause");
	}
	PyObject* ctpnforward = PyObject_GetAttrString(ctpndemo, "detect1");
	if(ctpnforward == nullptr)
	{
		PyErr_Print();
		system("pause");
	}
	PyObject* ctpnret = PyObject_CallObject(ctpnforward, ArgArray);
	if (ctpnret == nullptr)
	{
		PyErr_Print();
		system("pause");
	}
	vector<vector<cv::Point>> rects_point;
	vector<Rect> rects;
	if (PyList_Check(ctpnret)) {

		int SizeOfList = PyList_Size(ctpnret);

		for (int i = 0; i < SizeOfList; i++) {

			PyArrayObject *ListItem = (PyArrayObject *)PyList_GetItem(ctpnret, i);//读取List中的PyArrayObject对象，这里需要进行强制转换。

			int Rows = ListItem->dimensions[0], columns = ListItem->dimensions[1];
			std::cout << "The " << i << "th Array is:" << std::endl;
			for (int j = 0; j < Rows; j++) {

				vector<cv::Point> points;
				
				int x1 = *(float *)(ListItem->data + j * ListItem->strides[0] + 0 * ListItem->strides[1]);
				int y1 = *(float *)(ListItem->data + j * ListItem->strides[0] + 1 * ListItem->strides[1]);
				int x2 = *(float *)(ListItem->data + j * ListItem->strides[0] + 2 * ListItem->strides[1]);
				int y2 = *(float *)(ListItem->data + j * ListItem->strides[0] + 3 * ListItem->strides[1]);

				points.push_back(cv::Point(x1, y1));
				points.push_back(cv::Point(x2, y2));

				rects.push_back(Rect(x1,y1,x2-x1,y2-y1));
				//for (int k = 0; k < columns; k++) {

					//std::cout << *(float *)(ListItem->data + j * ListItem->strides[0] + k * ListItem->strides[1]) << " ";//访问数据，Index_m 和 Index_n 分别是数组元素的坐标，乘上相应维度的步长，即可以访问数组元素
				//}
				//std::cout << std::endl;
			}

			Py_DECREF(ListItem);
		}
	}
	else {

		std::cout << "Not a List" << std::endl;
	}

	Py_Finalize();
	for (auto rect : rects) {
		rectangle(img,rect,red);
	}
	cv::imshow("ctpn", img);
	cv::waitKey(0);
	system("pause");
}

void drawHist(Mat img) {
	double maxVal;
	cv::minMaxLoc(img, 0, &maxVal, 0, 0);
	std::cout << "max value is: " << maxVal << std::endl;

	Mat histimg(100, (int)maxVal, CV_8UC1, cv::Scalar(0));
	Mat newimg(img.rows, img.cols, CV_8UC1, cv::Scalar(0));
	int maxhist = 0;
	int* hist = new int[(int)maxVal]{0};
	for(int i=0;i<img.rows;i++)
		for (int j = 0; j < img.cols; j++) {
			int index = img.at<uchar>(i, j);
			newimg.at < uchar >(i,j)= (int)index*255.0/179;
			hist[index] += 1;
			if (hist[index] > maxhist) {
				maxhist = hist[index];
			}
		}
	for (int i = 0; i < (int)maxVal; i++) {
		int num = hist[i];
		int height = 100.0*num / maxhist;
		for (int j = 0; j < height; j++) {
			histimg.at<uchar>(j, i) = 180;
		}
	}
	cv::imshow("hist", histimg);
	cv::imshow("newimg", newimg);
	cv::waitKey(0);
}

vector<string> imgnames;
//vector<string> imglabels;
vector<string> imgpaths;
vector<string> labelpaths;
void collectImgsLabels(string path,string labelpath) {
	iterDir(path, imgnames);
	for (auto imgname : imgnames) {
		imgpaths.push_back(path + "\\" + imgname);
		imgname.replace(imgname.size() - 3, 3, "xml");
		labelpaths.push_back(labelpath + "\\" + imgname);
	}
}


#include "opencvTest.hpp"
int main1(int argc, char** argv) {
	//testpy();
	//PythonInitEmbeddedCaffeModule();
	string ctpnmodel_file = "d:/Lewis/projects/container_ocr/CTPN/models/deploy.prototxt";
	string ctpntrainded_file = "d:/Lewis/projects/container_ocr/CTPN/models/ctpn_trained_model.caffemodel";
	ctpn_detector=new CTPN_Detector(ctpnmodel_file, ctpntrainded_file,"demo");
	
	

	//vector<Rect> ctpn_rects;
	//ctpn_detector->detect(input1, ctpn_rects);

#ifdef SHOW_CTPN //show ctpn detection result
	Mat input_img = cv::imread(input1);
	for (auto rect : ctpn_rects) {
		cv::rectangle(input_img, rect, red);
	}
	cv::imshow("ctpn_detection", input_img);
	cv::waitKey(0);
#endif
	//initCTPN();
	//::google::InitGoogleLogging(argv[0]);
	char* model_file = "D:/OneDrive/Lewis/projects/BoxNumberDetect/demo/model/deploy.prototxt";
	char* trained_file = "D:\\OneDrive\\Lewis\\projects\\BoxNumberDetect\\demo\\model\\exVersion_iter_30000.caffemodel";
	char* mean_file = "D:\\OneDrive\\Lewis\\projects\\BoxNumberDetect\\demo\\model\\testMean.binaryproto";
	char* label_file = "D:\\OneDrive\\Lewis\\projects\\BoxNumberDetect\\demo\\model\\labels.txt";
	//char* input1 = "E:\\OneDrive\\Lewis\\projects\\BoxNumberDetect\\demo\\select\\first2.png";
	//char* input2 = "E:\\OneDrive\\Lewis\\projects\\BoxNumberDetect\\demo\\select\\second11.png";

	init(model_file, trained_file, mean_file, label_file);
	//Classifier classifier(model_file, trained_file, mean_file, label_file);
	//string input1 = "E:\\OneDrive\\Lewis\\projects\\BoxNumberDetect\\demo\\select\\first30.png";
	//string input2 = "E:\\OneDrive\\Lewis\\projects\\BoxNumberDetect\\demo\\select\\second11.png";

	//std::cin >> para;

	scale = 0.6;
	scaley1 = 0.8;
	scaley2 = 0.8;
	scaley3 = 0.8;
	maxArea = 300;
	minRatio = 1.2;
	scaleRect = 0.2;
	scalethres = 4;
	fillThres = 10;

	/*scale = atof(argv[3]);
	scaley1 = atof(argv[4]);
	scaley2 = atof(argv[5]);
	scaley3 = atof(argv[6]);
	maxArea = atoi(argv[7]);
	minRatio = atof(argv[8]);
	scaleRect = atof(argv[9]);
	scalethres = atof(argv[10]);
	fillThres = atof(argv[11]);*/
	std::cout << std::endl << scale << " " << scaley1 << " " << scaley2 << " " << scaley3 << std::endl;
	//vector<string> output = box_detect(classifier, input1, input2);

	
#if 1
	//string testimgPath = "D:\\OneDrive\\Lewis\\GraduationAffairs\\imgs\\first6.png";
	string testimgPath = "D:\\Lewis\\projects\\container_ocr\\testset\\122_0030_170929_105249_06_rear_1_1.jpg";

	hueHist(testimgPath);

	Mat testimg = cv::imread(testimgPath);
	Mat testgreyimg;
	cvtColor(testimg, testgreyimg, CV_BGR2GRAY);
	findBackNumberRegion(testimgPath, testimg, testgreyimg, *classifier);
	system("pause");
#endif

	vector<string> arg1;
	vector<string> arg2;
	

	arg1.push_back("122_0056_170929_125147_06_rear_1_1.jpg");
	arg2.push_back("122_0056_170929_125147_05_front_1.jpg");

	arg1.push_back("122_0002_170929_091901_06_rear_1_1.jpg");
	arg2.push_back("122_0057_170929_125757_05_front_1.jpg");

	arg1.push_back("122_0105_170929_171915_06_rear_1_1.jpg");
	arg2.push_back("122_0027_170929_104647_05_front_1.jpg");

	string imgdir="D:\\Lewis\\projects\\container_ocr\\testset";
	string labeldir = "D:\\Lewis\\projects\\container_ocr\\boxlabel";
	collectImgsLabels(imgdir,labeldir);

	for (int k = 0; k < imgpaths.size();k++) {
		//string input2 = "E:\\OneDrive\\Lewis\\projects\\BoxNumberDetect\\imgs\\126_0062_170214_165717_Rear_03.jpg";
		//string input1 = "E:\\OneDrive\\Lewis\\projects\\BoxNumberDetect\\imgs\\0075\\126_0075_170221_152440_Rear.jpg";
		//string input1 = "D:\\container_ocr\\detected_imgs\\" + arg1[k];
		//string input2 = "D:\\container_ocr\\detected_imgs\\" + arg2[k];
		string input1 = imgpaths[k];
		Mat image1 = cv::imread(input1);
		Mat gray1;

		

		cvtColor(image1, gray1, CV_BGR2GRAY);

		Mat imghsv;
		vector<Mat> hsvChannels;
		cvtColor(image1, imghsv, CV_BGR2HLS);
		split(imghsv, hsvChannels);

		//drawHist(hsvChannels[0]);
		/*Mat testImg(image1.cols, image1.rows, CV_8UC1, cv::Scalar(0));
		for (int i = 0; i < hsvChannels[0].rows; i++)
			for (int j = 0; j < hsvChannels[0].cols; j++) {
				if (hsvChannels[0].at<float>(j, i) < 100) {
					testImg.at<uchar>(j, i) = 255;
				}
				else {
					testImg.at<uchar>(j, i) = 0;
				}
			}*/
		//cv::imshow("test", testImg);
		//cv::waitKey(0);
		/*Mat h_hist;
		float range[] = { 0,180 };
		const float* histrange = { range };
		float histsize = 30;
		int channels[] = { 0 };
		cv::calcHist(&imghsv, 1, channels, Mat(), h_hist, 1, &histsize, histrange, true, false);
		*/
		// Quantize the hue to 30 levels
		// and the saturation to 32 levels
		//int hbins = 30, sbins = 32;
		//int histSize[] = { hbins };
		//// hue varies from 0 to 179, see cvtColor
		//float hranges[] = { 0, 180 };
		//// saturation varies from 0 (black-gray-white) to
		//// 255 (pure spectrum color)
		//float sranges[] = { 0, 256 };
		//const float* ranges[] = { hranges };
		//Mat hist;
		//// we compute the histogram from the 0-th and 1-st channels
		//int channels[] = { 0 };

		//calcHist(&imghsv, 1, channels, Mat(), // do not use mask
		//	hist, 1, histSize, ranges,
		//	true, // the histogram is uniform
		//	false);

		//int hist_w = 260; int hist_h = 400;
		//int bin_w = cvRound((double)hist_w / histSize[0]);
		//Mat histImage(hist_w, hist_h, CV_8UC3, cv::Scalar(0, 0, 0));
		//normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, Mat());
		//for (int i = 1; i < histSize[0]; i++)
		//	line(histImage, cv::Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
		//		cv::Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
				//cv::Scalar(255, 0, 0), 2, 8, 0);
		//cv::namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE);
		//imshow("calcHist Demo", histImage);
		//cv::imshow("h", hsvChannels[0]);

		//cv::imshow("s", hsvChannels[1]);
		//cv::imshow("v", hsvChannels[2]);
		//cv::waitKey(0);
		//Mat image2 = cv::imread(input2);
		Mat gray2;
		//cvtColor(image2, gray2, CV_BGR2GRAY);

		//cv::imshow("image1", image1);
	//	cv::imshow("image2", image2);
	//	cv::waitKey(0);
		std::cout << std::endl << "Detection Result!" << std::endl;
		std::cout << recogBack(image1, findBackNumberRegion2(input1, image1, gray1, *classifier,labelpaths[k]), *classifier) << std::endl;

		
	//	std::cout << recogFront(image2, findFrontNumberRegion(image2, gray2, *classifier), *classifier) << std::endl;
	}
	std::cout << detectRatio1 << ' ' << detectRatio2 << ' ' << containerBoxNum << ' ' << detectRatio1 / containerBoxNum << ' ' << detectRatio2 / containerBoxNum << std::endl;
	std::cout << detectRatio11 << ' ' << detectRatio22 << ' ' << containerBoxNum2 << ' ' << detectRatio11 / containerBoxNum2 << ' ' << detectRatio22 / containerBoxNum2 << std::endl;
	std::cout << detectRatio111 << ' ' << detectRatio222 << ' ' << containerBoxNum3 << ' ' << detectRatio111 / containerBoxNum3 << ' ' << detectRatio222 / containerBoxNum3 << std::endl;
	system("pause");
#endif
	return 0;
}
#else
int main(int argc, char** argv) {
	LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif // USE_OPENCV