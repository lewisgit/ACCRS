#include "tcpClient.h"
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

//#include <string>
using namespace rapidjson;
//using std::string;
WSADATA wsa;
//SOCKET client_ctpn;
//SOCKET client_east;
//SOCKET client_crnn;
//SOCKET client_deeplab;
//SOCKET client_alex;
static const char* kTypeNames[] =
{ "Null", "False", "True", "Object", "Array", "String", "Number" };

int  initSocket() {
	if (WSAStartup(MAKEWORD(1, 1), &wsa) != 0)
		return -1;
	
	return 0;
}

int east_request(string imgpath, std::vector<east_bndbox>& boxes) {
	SOCKADDR_IN addrServer;
	//SOCKET client_alex = socket(AF_INET, SOCK_STREAM, 0);
	//SOCKET client_ctpn = socket(AF_INET, SOCK_STREAM, 0);
	SOCKET client_east = socket(AF_INET, SOCK_STREAM, 0);
	//SOCKET client_crnn = socket(AF_INET, SOCK_STREAM, 0);
	//SOCKET client_deeplab = socket(AF_INET, SOCK_STREAM, 0);
	addrServer.sin_addr.S_un.S_addr = inet_addr("127.0.0.1");
	addrServer.sin_family = AF_INET;
	addrServer.sin_port = htons(6001);
	int ret = connect(client_east, (SOCKADDR* )&addrServer, sizeof(SOCKADDR));
	send(client_east, imgpath.c_str(), imgpath.length() + 1, 0);
	const int buflen = 4000;
	char recvBuf[buflen];
	for (int i = 0; i < buflen; i++) {
		recvBuf[i] = '0';
		//printf("%c ", recvBuf[i]);
	}
	recv(client_east, recvBuf, buflen, 0);
	
	Document json;
	json.Parse(recvBuf);
	printf("\n%s\n", recvBuf);
	
	static const char* kTypeNames[] =
	{ "Null", "False", "True", "Object", "Array", "String", "Number" };
	/*for (Value::ConstMemberIterator itr = json.MemberBegin();
		itr != json.MemberEnd(); ++itr)
	{
		printf("Type of member %s is %s\n",
			itr->name.GetString(), kTypeNames[itr->value.GetType()]);
	}*/
	
	for (int j = 0; j < json["boxnum"].GetInt(); j++) {
		//char *charind = itoa(j,);
		string charind=std::to_string(j);
		const Value& array1 = json[charind.c_str()];
		assert(array1.IsArray());
		//printf("%d\n", array1.Size());
		//for (Value::ConstValueIterator itr = array1.Begin(); itr != array1.End(); ++itr)
		//	printf("%f ", itr->GetFloat());
		//for (SizeType i = 0; i < array1.Size(); i++)
		//	printf("%f ", array1[i].GetFloat());

		east_bndbox box;
		box.x0 = array1[0].GetFloat();
		box.y0 = array1[1].GetFloat();
		box.x1 = array1[2].GetFloat();
		box.y1 = array1[3].GetFloat();
		box.x2 = array1[4].GetFloat();
		box.y2 = array1[5].GetFloat();
		box.x3 = array1[6].GetFloat();
		box.y3 = array1[7].GetFloat();

		boxes.push_back(box);

		//printf("\n");
	}
	//printf("%d\n", json[1].GetInt());
	//printf("%d\n", json["2"].GetInt());
	//printf("%d\n", json["3"].GetInt());
	//printf("%d\n", json["4"].GetInt());
	closesocket(client_east);
	//WSACleanup();
	return 0;
}



int ctpn_request(string imgpath,std::vector<cv::Rect>& rects, bool isRotate) {
	SOCKADDR_IN addrServer;
	//SOCKET client_alex = socket(AF_INET, SOCK_STREAM, 0);
	SOCKET client_ctpn = socket(AF_INET, SOCK_STREAM, 0);
	//SOCKET client_east = socket(AF_INET, SOCK_STREAM, 0);
	//SOCKET client_crnn = socket(AF_INET, SOCK_STREAM, 0);
	//SOCKET client_deeplab = socket(AF_INET, SOCK_STREAM, 0);
	addrServer.sin_addr.S_un.S_addr = inet_addr("127.0.0.1");
	addrServer.sin_family = AF_INET;
	addrServer.sin_port = htons(6002);
	int ret = connect(client_ctpn, (SOCKADDR*)&addrServer, sizeof(SOCKADDR));
	send(client_ctpn, imgpath.c_str(), imgpath.length() + 1, 0);
	const int buflen = 4000;
	char recvBuf[buflen];
	for (int i = 0; i < buflen; i++) {
		recvBuf[i] = '0';
		//printf("%c ", recvBuf[i]);
	}
	recv(client_ctpn, recvBuf, buflen, 0);

	Document json;
	json.Parse(recvBuf);
	printf("\n%s\n", recvBuf);

	
	//for (Value::ConstMemberIterator itr = json.MemberBegin();
	//	itr != json.MemberEnd(); ++itr)
	//{
	//	printf("Type of member %s is %s\n",
	//		itr->name.GetString(), kTypeNames[itr->value.GetType()]);
	//}
	//
	if (isRotate) {
		for (int j = 0; j < json["boxnum"].GetInt(); j++) {
			//char *charind = itoa(j,);
			string charind = std::to_string(j);
			const Value& array1 = json[charind.c_str()];
			assert(array1.IsArray());
			printf("%d\n", array1.Size());

			int y0 = array1[0].GetFloat();
			int x0 = array1[1].GetFloat();
			int y1 = array1[2].GetFloat();
			int x1 = array1[3].GetFloat();

			cv::Rect rect(x0, y0, x1 - x0, y1 - y0);
			rects.push_back(rect);
			//for (Value::ConstValueIterator itr = array1.Begin(); itr != array1.End(); ++itr)
			printf("\n");
		}
	}
	else {
		for (int j = 0; j < json["boxnum"].GetInt(); j++) {
			//char *charind = itoa(j,);
			string charind = std::to_string(j);
			const Value& array1 = json[charind.c_str()];
			assert(array1.IsArray());
			printf("%d\n", array1.Size());

			int x0 = array1[0].GetFloat();
			int y0 = array1[1].GetFloat();
			int x1 = array1[2].GetFloat();
			int y1 = array1[3].GetFloat();

			cv::Rect rect(x0, y0, x1 - x0, y1 - y0);
			rects.push_back(rect);
			//for (Value::ConstValueIterator itr = array1.Begin(); itr != array1.End(); ++itr)
			printf("\n");
		}
	}
	
	//printf("%d\n", json[1].GetInt());
	//printf("%d\n", json["2"].GetInt());
	//printf("%d\n", json["3"].GetInt());
	//printf("%d\n", json["4"].GetInt());
	closesocket(client_ctpn);
	//WSACleanup();
	return 0;
}


int crnn_request(string imgpath, std::vector<string>& strs) {
	SOCKADDR_IN addrServer;
	//SOCKET client_alex = socket(AF_INET, SOCK_STREAM, 0);
	//SOCKET client_ctpn = socket(AF_INET, SOCK_STREAM, 0);
	//SOCKET client_east = socket(AF_INET, SOCK_STREAM, 0);
	SOCKET client_crnn = socket(AF_INET, SOCK_STREAM, 0);
	//SOCKET client_deeplab = socket(AF_INET, SOCK_STREAM, 0);
	addrServer.sin_addr.S_un.S_addr = inet_addr("127.0.0.1");
	addrServer.sin_family = AF_INET;
	addrServer.sin_port = htons(6003);
	int ret = connect(client_crnn, (SOCKADDR*)&addrServer, sizeof(SOCKADDR));
	send(client_crnn, imgpath.c_str(), imgpath.length(), 0);
	const int buflen = 4000;
	char recvBuf[buflen];
	for (int i = 0; i < buflen; i++) {
		recvBuf[i] = '0';
		//printf("%c ", recvBuf[i]);
	}
	recv(client_crnn, recvBuf, buflen, 0);

	Document json;
	json.Parse(recvBuf);
	printf("\n%s\n", recvBuf);

	static const char* kTypeNames[] =
	{ "Null", "False", "True", "Object", "Array", "String", "Number" };
	/*for (Value::ConstMemberIterator itr = json.MemberBegin();
		itr != json.MemberEnd(); ++itr)
	{
		printf("Type of member %s is %s\n",
			itr->name.GetString(), kTypeNames[itr->value.GetType()]);
	}*/

	for (int j = 0; j < json["strnum"].GetInt(); j++) {
		//char *charind = itoa(j,);
		string charind = std::to_string(j);
		const Value& array1 = json[charind.c_str()];
		//std::cout << array1.GetString() << std::endl;
		strs.push_back(array1.GetString());

		printf("\n");
	}
	//printf("%d\n", json[1].GetInt());
	//printf("%d\n", json["2"].GetInt());
	//printf("%d\n", json["3"].GetInt());
	//printf("%d\n", json["4"].GetInt());
	closesocket(client_crnn);
	//WSACleanup();
	return 0;
}

int deeplab_request(string imgpath, cv::Rect& rect) {
	SOCKADDR_IN addrServer;
	//SOCKET client_alex = socket(AF_INET, SOCK_STREAM, 0);
	//SOCKET client_ctpn = socket(AF_INET, SOCK_STREAM, 0);
	//SOCKET client_east = socket(AF_INET, SOCK_STREAM, 0);
	//SOCKET client_crnn = socket(AF_INET, SOCK_STREAM, 0);
	SOCKET client_deeplab = socket(AF_INET, SOCK_STREAM, 0);
	addrServer.sin_addr.S_un.S_addr = inet_addr("127.0.0.1");
	addrServer.sin_family = AF_INET;
	addrServer.sin_port = htons(6004);
	int ret = connect(client_deeplab, (SOCKADDR*)&addrServer, sizeof(SOCKADDR));
	send(client_deeplab, imgpath.c_str(), imgpath.length(), 0);
	const int buflen = 4000;
	char recvBuf[buflen];
	for (int i = 0; i < buflen; i++) {
		recvBuf[i] = '0';
		//printf("%c ", recvBuf[i]);
	}
	recv(client_deeplab, recvBuf, buflen, 0);

	Document json;
	json.Parse(recvBuf);
	printf("\n%s\n", recvBuf);

	static const char* kTypeNames[] =
	{ "Null", "False", "True", "Object", "Array", "String", "Number" };
	/*for (Value::ConstMemberIterator itr = json.MemberBegin();
		itr != json.MemberEnd(); ++itr)
	{
		printf("Type of member %s is %s\n",
			itr->name.GetString(), kTypeNames[itr->value.GetType()]);
	}*/


	int x0 = json["x0"].GetInt();
	int x1 = json["x1"].GetInt();
	int y0 = json["y0"].GetInt();
	int y1 = json["y1"].GetInt();
	rect = cv::Rect(x0,y0,x1-x0,y1-y0);

	//for (int j = 0; j < json["strnum"].GetInt(); j++) {
	//	//char *charind = itoa(j,);
	//	string charind = std::to_string(j);
	//	const Value& array1 = json[charind.c_str()];
	//	std::cout << array1.GetString() << std::endl;
	//	strs.push_back(array1.GetString());

	//	printf("\n");
	//}
	//printf("%d\n", json[1].GetInt());
	//printf("%d\n", json["2"].GetInt());
	//printf("%d\n", json["3"].GetInt());
	//printf("%d\n", json["4"].GetInt());
	closesocket(client_deeplab);
	//WSACleanup();
	return 0;
}

int alex_request(string imgpath, string& recog_result) {
	SOCKADDR_IN addrServer;
	SOCKET client_alex = socket(AF_INET, SOCK_STREAM, 0);
	//SOCKET client_ctpn = socket(AF_INET, SOCK_STREAM, 0);
	//SOCKET client_east = socket(AF_INET, SOCK_STREAM, 0);
	//SOCKET client_crnn = socket(AF_INET, SOCK_STREAM, 0);
	//SOCKET client_deeplab = socket(AF_INET, SOCK_STREAM, 0);
	addrServer.sin_addr.S_un.S_addr = inet_addr("127.0.0.1");
	addrServer.sin_family = AF_INET;
	addrServer.sin_port = htons(6005);
	int ret = connect(client_alex, (SOCKADDR*)&addrServer, sizeof(SOCKADDR));
	send(client_alex, imgpath.c_str(), imgpath.length(), 0);
	const int buflen = 4000;
	char recvBuf[buflen];
	for (int i = 0; i < buflen; i++) {
		recvBuf[i] = '0';
		//printf("%c ", recvBuf[i]);
	}
	recv(client_alex, recvBuf, buflen, 0);

	Document json;
	json.Parse(recvBuf);
	printf("\n%s\n", recvBuf);

	static const char* kTypeNames[] =
	{ "Null", "False", "True", "Object", "Array", "String", "Number" };
	/*for (Value::ConstMemberIterator itr = json.MemberBegin();
		itr != json.MemberEnd(); ++itr)
	{
		printf("Type of member %s is %s\n",
			itr->name.GetString(), kTypeNames[itr->value.GetType()]);
	}*/


	recog_result = json["result"].GetString();
	//int x1 = json["x1"].GetInt();
	//int y0 = json["y0"].GetInt();
	//int y1 = json["y1"].GetInt();
	//rect = cv::Rect(x0, y0, x1 - x0, y1 - y0);

	//for (int j = 0; j < json["strnum"].GetInt(); j++) {
	//	//char *charind = itoa(j,);
	//	string charind = std::to_string(j);
	//	const Value& array1 = json[charind.c_str()];
	//	std::cout << array1.GetString() << std::endl;
	//	strs.push_back(array1.GetString());

	//	printf("\n");
	//}
	//printf("%d\n", json[1].GetInt());
	//printf("%d\n", json["2"].GetInt());
	//printf("%d\n", json["3"].GetInt());
	//printf("%d\n", json["4"].GetInt());
	closesocket(client_alex);
	//WSACleanup();
	return 0;
}
//
//int main(int argc, char* argv[])
//{
//	try
//	{
//		// the user should specify the server - the 2nd argument
//		if (argc != 2)
//		{
//			std::cerr << "Usage: client " << std::endl;
//			return 1;
//		}
//
//		// Any program that uses asio need to have at least one io_service object
//		boost::asio::io_service io_service;
//
//		// Convert the server name that was specified as a parameter to the application, to a TCP endpoint. 
//		// To do this, we use an ip::tcp::resolver object.
//		tcp::resolver resolver(io_service);
//
//		// A resolver takes a query object and turns it into a list of endpoints. 
//		// We construct a query using the name of the server, specified in argv[1], 
//		// and the name of the service, in this case "daytime".
//		tcp::resolver::query query(argv[1], "daytime");
//
//		// The list of endpoints is returned using an iterator of type ip::tcp::resolver::iterator. 
//		// A default constructed ip::tcp::resolver::iterator object can be used as an end iterator.
//		tcp::resolver::iterator endpoint_iterator = resolver.resolve(query);
//
//		// Now we create and connect the socket.
//		// The list of endpoints obtained above may contain both IPv4 and IPv6 endpoints, 
//		// so we need to try each of them until we find one that works. 
//		// This keeps the client program independent of a specific IP version. 
//		// The boost::asio::connect() function does this for us automatically.
//		tcp::socket socket(io_service);
//		boost::asio::connect(socket, endpoint_iterator);
//
//		// The connection is open. All we need to do now is read the response from the daytime service.
//		for (;;)
//		{
//			// We use a boost::array to hold the received data. 
//			boost::array<char, 100> buf;
//			boost::system::error_code error;
//
//			// The boost::asio::buffer() function automatically determines 
//			// the size of the array to help prevent buffer overruns.
//			size_t len = socket.read_some(boost::asio::buffer(buf), error);
//
//			// When the server closes the connection, 
//			// the ip::tcp::socket::read_some() function will exit with the boost::asio::error::eof error, 
//			// which is how we know to exit the loop.
//			if (error == boost::asio::error::eof)
//				break; // Connection closed cleanly by peer.
//			else if (error)
//				throw boost::system::system_error(error); // Some other error.
//
//			std::cout.write(buf.data(), len);
//		}
//	}
//	// handle any exceptions that may have been thrown.
//	catch (std::exception& e)
//	{
//		std::cerr << e.what() << std::endl;
//	}
//
//	return 0;
//}