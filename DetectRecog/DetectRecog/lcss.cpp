#include <iostream>
#include <vector>
#include <cstdlib>
#include <algorithm>    //max()函数
#include <string>

using namespace std;

void lcss(const string str1, const string str2, string& str) 
{
	int size1 = str1.length();
	int size2 = str2.length();
	const char* s1 = str1.data() - 1;//从1开始数，方便后面的代码编写(将字符串指针str1指向了 首元素的前面。)，这样第0行/列，LCS（即c[0][j],c[i][0]) 为0。
	const char* s2 = str2.data() - 1;
	vector<vector<int>> chess(size1 + 1, vector<int>(size2 + 1));
	//因为是从1起数。而第0行/列是外加的（因为字符串与空串的LCS永远为0）;但是总长需要+1
	//chess棋盘：二维数组，行数为size1，列数为size2
	int i, j;
	//for (i = 0; i < size1; i++) {//第0列初始化为0（其实自动初始化的，可以不写）
	//  chess[i][0] = 0;
	//}
	//for (j = 0; j < size2; j++) {//第0行初始化为0
	//  chess[0][j] = 0;
	//}

	for (i = 1; i < size1; i++) { // 一定要从1起！！！
		for (j = 1; j < size2; j++) {
			if (s1[i] == s2[j]) {
				//若两串当前元素相同，则其当前LCS等于左上角LCS（chess[i - 1][j - 1] ）+ 1（此处加的1，即为当前元素(i,j).）
				chess[i][j] = chess[i - 1][j - 1] + 1;
			}
			else {  // if (s1[i] != s2[j])
				//若两串当前元素不同，则其当前LCS一定等于左边、上边的LCS的最大值。（而到了（i,j）处，不是公共元素。跳过。）
				chess[i][j] = max(chess[i - 1][j], chess[i][j - 1]);
			}
		}
	}
	//到此，排表（棋盘）结束。

	i = size1;//将i、j下标落到末尾元素上。
	j = size2;
	while (i != 0 && j != 0)
	{
		if (s1[i] == s2[j]) {   //将相同的子元素压栈。然后指针前移，直到i、j指向0终止(因为任何字符串 与0 求公共子序列，都是0)
			str.push_back(s1[i]);
			i--;
			j--;
		}
		else { //若二者不相等，而最长公共子序列一定是由LCS（chess[i][j-1] or chess[i-1][j]）的较大者得来，故将较大者的指针前移，接着遍历。
			if (chess[i][j - 1] > chess[i - 1][j]) {
				j--;        //将当前列前移到j-1列
			}
			else { // if(chess[i][j - 1] <= chess[i - 1][j])
				i--;
			}
		}
	}
	//求LCS（之一）
	reverse(str.begin(), str.end());

	//求LCS的长度
	/*
	int maxLen = 0;
	for (i = 0; i < size1; i++){
		for (j = 0; j < size2; j++){
			maxLen = max(maxLen, chess[i][j]);
		}
	}
	return  maxLen;
	*/

}