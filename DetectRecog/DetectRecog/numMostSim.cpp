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
		exit(-1);
	}


	/*以下两个for循环将4个字符全部匹配和只有第3位不匹配这两种情况单独拿出来处理*/
	for (auto dataIterator = matchData.begin(); dataIterator != matchData.end(); ++dataIterator)
	{
		string temp = *dataIterator;
		/*匹配输入字符串iStr和数据库中的字符串temp*/
		string matchResult;
		lcss(temp, iStr, matchResult);
		if (matchResult.length() == 4)
		{
			oStr = temp;
			return true;
		}
	}

	for (auto dataIterator = matchData.begin(); dataIterator != matchData.end(); ++dataIterator)
	{
		string temp = *dataIterator;
		/*匹配输入字符串iStr和数据库中的字符串temp*/
		string matchResult;
		lcss(temp, iStr, matchResult);
		if (matchResult.length() == 3)
		{
			oStr = temp;
			return true;
		}
	}

	for (auto dataIterator = matchData.begin(); dataIterator != matchData.end(); ++dataIterator)
	{
		string temp = *dataIterator;
		/*匹配输入字符串iStr和数据库中的字符串temp*/
		string matchResult;
		lcss(temp, iStr, matchResult);
		if (matchResult.length() == 2)
		{
			oStr = temp;
			return false;
		}
	}

	for (auto dataIterator = matchData.begin(); dataIterator != matchData.end(); ++dataIterator)
	{
		string temp = *dataIterator;
		/*匹配输入字符串iStr和数据库中的字符串temp*/
		string matchResult;
		lcss(temp, iStr, matchResult);
		if (matchResult.length() == 1)
		{
			oStr = temp;
			return false;
		}
	}
	oStr = "";
	return false;
}