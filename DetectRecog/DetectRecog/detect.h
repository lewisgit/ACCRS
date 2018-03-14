#pragma once
#include<string>
#include<vector>
using std::vector;
using std::string;

#ifdef DLL_GEN
#define DLL_API _declspec(dllexport) 
#else
#define DLL_API _declspec(dllimport) 
#endif
/*input1: 第一个输入图片路径
input2: 第二个输入图片路径
output1: 第一张图识别结果
output2: 第二张图识别结果
mode: =0 自动判断正反面
=1 设定，第一张图为正面，第二张图为反面
*/

DLL_API bool detect(char* output1, char* output2, const char* input1,const  char* input2, int mode);
