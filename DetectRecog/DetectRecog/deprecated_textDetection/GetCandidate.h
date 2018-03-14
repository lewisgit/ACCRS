#ifndef _GET_CANDIDATE_
#define _GET_CANDIDATE_
#include "headline.h"
#include <vector>
#include "Candidate.h"
#include "swt.h"
#include "mser.h"
using namespace std;
using cv::ml::RTrees;
class GetCandidate{
public:
	std::vector<Candidate> run(cv::Mat& Image);
	void featureExtract();
	void swtprocess();
	void ExtractCCfeatures();
	void Filter();

private:
	//Ptr<cv::MSER> mser=cv::MSER::create(5,20,14400);
	MYMSER mser=MYMSER(5,20,14400);
	Swt swt;
	
	Ptr<RTrees> CharacterClassifier=RTrees::create();
	
	cv::Mat oriBgrImage_8UC3;
	cv::Mat gray_source_image;
	vector< vector<cv::Point2i> > strVectorStore;
	vector<Candidate> ccStore;
	vector<Candidate> candidateStore;
};
#endif
