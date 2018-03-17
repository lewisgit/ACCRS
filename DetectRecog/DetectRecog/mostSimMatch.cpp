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

/*存放匹配字符串的文本文件名*/
string MATCHDATA = "D:/ACCRS-master/matchData.txt";
string SPECIALCASEDATA = "D:/ACCRS-master/specialCase.ini";

extern void lcss(const string str1, const string str2, string& str);
bool mostSimMatch(const string &iStr, string &oStr);
static bool readMatchData(list<string> * p_matchData);
static bool readSpecCaseData(list<string> * p_matchData, map<string, string> * p_specCaseData);
static bool specialCase(const string ,string &, list<string>&, map<string, string>&);
vector<string> split(const string& str, const string& delim);

static bool readMatchData(list<string> * p_matchData)
{
	ifstream ifstrm = ifstream(MATCHDATA);
	if (!ifstrm.is_open())
	{
		cerr << "open " + MATCHDATA + " failed!" << endl;
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

vector<string> split(const string& str, const string& delim) 
{
	vector<string> res;
	if ("" == str) return res;
	//先将要切割的字符串从string类型转换为char*类型  
	char * strs = new char[str.length() + 1]; //不要忘了  
	strcpy_s(strs,str.length()+1 ,str.c_str());

	char * d = new char[delim.length() + 1];
	strcpy_s(d, delim.length()+1, delim.c_str());

	char *buf;
	char *p = strtok_s(strs, d, &buf);
	while (p) {
		string s = p; //分割得到的字符串转换为string类型  
		res.push_back(s); //存入结果数组  
		p = strtok_s(NULL, d, &buf);
	}

	return res;
}

static bool readSpecCaseData(list<string> * p_matchData, map<string, string> * p_specCaseData)
{
	ifstream ifstrm = ifstream(SPECIALCASEDATA);
	string temp;
	vector<string> splitResult;
	if (!ifstrm.is_open())
	{
		cerr << "open " + SPECIALCASEDATA + " failed!" << endl;
		return false;
	}
	while (true)
	{
		ifstrm >> temp;
		if (temp.empty())
			break;
		splitResult = split(temp, ":");
		p_matchData->push_back(splitResult.at(0));
		p_specCaseData->insert(pair<string, string>(splitResult.at(0), splitResult.at(1)));
		temp.clear();
	}
	ifstrm.close();
	return true;
}

bool mostSimMatch(const string &iStr, string &oStr)
{
	list<string> matchData;
	list<string> specialCaseIndex;
	map<string, string> specialCaseData;
	bool isOk = readMatchData(&matchData);
	if (!isOk)
	{
		exit(-1);
	}

	isOk = readSpecCaseData(&specialCaseIndex, &specialCaseData);
	if (!isOk)
	{
		exit(-1);
	}

	/*以下两个for循环将4个字符全部匹配和只有第3位不匹配这两种情况单独拿出来处理*/
	int maxResultLength = 0;
	string maxResult = "";
	for (auto dataIterator = matchData.begin(); dataIterator != matchData.end(); ++dataIterator)
	{
		string temp = *dataIterator;
		/*匹配输入字符串iStr和数据库中的字符串temp*/
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

	if (maxResultLength == 3)
	{
		for (auto dataIterator = matchData.begin(); dataIterator != matchData.end(); ++dataIterator)
		{
			string temp = *dataIterator;
			std::regex reg(temp.replace(2, 1, "."));
			std::smatch matchResult;
			if (std::regex_search((string::const_iterator)iStr.begin(), (string::const_iterator)iStr.end(), matchResult, reg))
			{
				char thridChar = matchResult.str().at(2);
				if (thridChar == '6')
				{
					oStr = matchResult.str().replace(2, 1, "g");
				}
				else
				{
					oStr = matchResult.str().replace(2, 1, "u");
				}
				return true;
			}
		}
		oStr = maxResult;
		return false;
	}
	if (maxResultLength == 2)
	{
		bool isRecog = specialCase(iStr, oStr, specialCaseIndex, specialCaseData);
		if (isRecog)
		{
			return true;
		}
	}
	
	oStr = maxResult;
	return false;
}



static bool specialCase(const string iStr, string &oStr, list<string>& index, map<string, string>& mapData)
{
	for (auto dataIterator = index.begin(); dataIterator != index.end(); ++dataIterator)
	{
		string temp = *dataIterator;
		if (iStr.find(temp) != string::npos)
		{
			oStr = mapData[temp];
			return true;
		}
	}
	return false;
}

