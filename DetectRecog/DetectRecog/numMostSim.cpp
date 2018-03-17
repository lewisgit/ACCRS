#include <string>
#include <map>
#include <fstream>
#include <list>
#include <regex>  
#include <iostream>
#include <cstdlib>
#include <vector>
#include <cstring>
using namespace std;

string CONNUMDATA = "D:/ACCRS-master/conNumData.txt";

static bool readMatchData(list<string> * p_matchData);
extern void lcss(const string str1, const string str2, string& str);

static bool readMatchData(list<string> * p_matchData)
{
	ifstream ifstrm = ifstream(CONNUMDATA);
	if (!ifstrm.is_open())
	{
		cerr << "open " + CONNUMDATA + " failed!" << endl;
		return false;
	}
	while (true)
	{
		string temp;
		ifstrm >> temp;
		if (temp.empty())
			break;
		p_matchData->push_back(temp);
	}
	ifstrm.close();
	return true;
}

bool conNuMostSimMatch(const string &iStr, string &oStr)
{
	list<string> matchData;
	bool isOk = readMatchData(&matchData);
	if (!isOk)
	{
		oStr = iStr;
		return false;
		//exit(-1);
	}


	int maxResultLength = 0;
	string maxResult = "";
	for (auto dataIterator = matchData.begin(); dataIterator != matchData.end(); ++dataIterator)
	{
		string temp = *dataIterator;
		string matchResult;
		lcss(temp, iStr, matchResult);
		int resultLength = matchResult.length();
		if (resultLength == 4)
		{
			oStr = temp;
			return true;
		}
		else
		{
			if (resultLength > maxResultLength)
			{
				maxResultLength = resultLength;
				maxResult = temp;
			}
		}
	}
	oStr = maxResult;
	if (maxResultLength > 2)
	{ 
		return true;
	}
	else
	{
		return false;
	}

}