#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

void filter_embossing();                                  //���������͸�
void emb_change(int pos, void* userdata);                 //Ʈ����

void gaussian();                                          //����þ����͸�
void gau_change(int pos, void* userdata);                 //Ʈ����

void mean();                                              //��հ����͸�
void mean_change(int pos, void* userdata);                //Ʈ����

void unsharp_mask();                                      //���������ũ���͸�
void unsharp_change(int pos, void* userdata);             //Ʈ����

void filter_bilateral();                                  //�����߰�,��������͸�
void bilateral_change(int pos, void* userdata);           //Ʈ����

void filter_median();                                     //�����߰�,�̵�����͸�
void median_change(int pos, void* userdata);              //Ʈ����

void sobel_edge();                                        //����ũ��ݿ�������
void sobel_change(int pos, void* userdata);               //Ʈ����

void canny_edge();                                        //ĳ�Ͽ�������
void canny_change(int pos, void* userdata);               //Ʈ����

void hough_lines();                                       //������ȯ��������
void lines_change(int pos, void* userdata);               //Ʈ����

void hough_circles();                                     //������ȯ�����
void circles_change(int pos, void* userdata);             //Ʈ����

void adaptive();                                          //����������ȭ
void adaptive_change(int pos, void* userdata);            //Ʈ����

void erode_dilate();                                      //���������� ħ�İ���â
void open_close();                                        //���������� ����� ����

void labeling_stats();                                    //�������� ���̺�

void contours_hier();                                     //�ܰ�������
void hier_change(int pos, void* userdata);                //Ʈ����

int main(void)                    //���ι�
{
	//filter_embossing();           //���������͸�
	//gaussian();                   //����þ����͸�
	//mean();                       //��հ����͸�
	//unsharp_mask();               //��������͸�
	
	//filter_bilateral();           //�����߰�,��������͸�
	//filter_median();              //�����߰�,�̵�����͸�
	
	//sobel_edge();                 //����ũ��� ��������
	//canny_edge();                 //ĳ�Ͽ�������
	//hough_lines();                //������ȯ��������
	//hough_circles();              //������ȯ�����

	//adaptive();                   //����������ȭ
	//erode_dilate();               //���������� ħ�İ� ��â
	//open_close();                 //���������� ����� ����
	
	//labeling_stats();             //���̺�
	contours_hier();              //�ܰ�������

	return 0;
}


void filter_embossing()                                                      //���������͸�
{
	Mat src = imread("my.bmp", IMREAD_GRAYSCALE);                            //���� �� ��� �ҷ�����
	namedWindow("dst");                                                      //Ʈ���� ������ â �̸�
	createTrackbar("level", "dst", 0, 256, emb_change, (void*)&src);         //Ʈ���� �����Լ�, Ʈ���ٰ��� 256����

	waitKey();
}
void emb_change(int pos, void* userdata)                                     //���������͸� Ʈ����
{
	Mat src = *(Mat*)userdata;                                               //void*Ÿ�� ���� userdata�� 
	                                                                         //Mat*Ÿ������ ����ȯ�� �� src������ ����
	float data[] = { -1, -1, 0, -1, 0, 1, 0, 1, 1 };                         //3x3�ױ��� ������ ���� ����ũ ���
	Mat emboss(3, 3, CV_32FC1, data);                                        //�� ����� emboss����

	Mat dst;                                                                 //����̸�����
	filter2D(src, dst, -1, emboss, Point(-1, -1), pos);                      //���������� ����, ���͸� ��� ���� ��ȭ�ϴ�pos���� ����

	imshow("dst", dst);                                                      //���
}


void gaussian()                                                              //����þ����͸�
{
	Mat src = imread("my.bmp", IMREAD_GRAYSCALE);                            //���� �� ��� �ҷ�����
	namedWindow("dst");                                                      //Ʈ���� ������â �̸�
	createTrackbar("level", "dst", 0, 2, gau_change, (void*)&src);           //Ʈ���� �����Լ�,Ʈ���� 5���� ��ȯ

	waitKey();
}
void gau_change(int pos, void* userdata)                                      //Ʈ����
{
	Mat src = *(Mat*)userdata;                                                //void*Ÿ�� ���� userdata�� 
	                                                                          //Mat*Ÿ������ ����ȯ�� �� src������ ����
	if (pos % 2 == 0)pos = pos + (pos + 1);                                          //pos�� ¦���϶� ���� (0�϶� 1, 2�϶� 5�� ��ȯ)
	else pos = pos + 2;                                                         //Ȧ���϶� ������ �°��ϱ����� 2����(1�϶� 3���� ��ȯ)
	Mat dst;                                                                  //dst���� ����
	GaussianBlur(src, dst, Size(), (double)pos);                              //src���� ����þ����͸� ������ dst�� ����

	imshow("dst", dst);                                                       //dst ����
}

void mean()                                                                   //��հ����͸�
{
	Mat src = imread("my.bmp", IMREAD_GRAYSCALE);                             //���� �� ��� �ҷ�����
	namedWindow("dst");                                                       //Ʈ���� ������â �̸�
	createTrackbar("level", "dst", 0, 7, mean_change, (void*)&src);           //Ʈ���� �����Լ�, Ʈ���� 7���� ��ȯ

	waitKey();
}
void mean_change(int pos, void* userdata)
{
	Mat src = *(Mat*)userdata;                                                //void*Ÿ�� ���� userdata�� 
	                                                                          //Mat*Ÿ������ ����ȯ�� �� src������ ����
	Mat dst;                                                                  //dst���� ����
	blur(src, dst, Size(pos, pos));                                           //posXposũ���� ��հ����� ����ũ �̿��Ͽ� ���� ����

	imshow("dst", dst);                                                       //dst����
}

void unsharp_mask()                                                           //��������͸�
{
	Mat src = imread("my.bmp", IMREAD_GRAYSCALE);                             //���� �� ��� �ҷ�����
	namedWindow("dst");                                                       //Ʈ���� ������â �̸�
	createTrackbar("level", "dst", 0, 5, unsharp_change, (void*)&src);        //Ʈ���� �����Լ�, Ʈ���� 5���� ��ȯ

	waitKey();
	destroyAllWindows();
}
void unsharp_change(int pos, void* userdata)                                  //Ʈ����
{
	Mat src = *(Mat*)userdata;                                                //void*Ÿ�� ���� userdata�� 
	                                                                          //Mat*Ÿ������ ����ȯ�� �� src������ ����

	Mat blurred;                                                              //blurred���� ����
	GaussianBlur(src, blurred, Size(), pos);                                  //����þ� ���� �̿��� ���� ������ blurred�� ����

	float alpha = 1.f;                                                        //����� ����ũ ���͸� ����
	Mat dst = (1 + alpha) * src - alpha * blurred;

	imshow("dst", dst);
}


void filter_bilateral()                                                       //��������͸�
{
	Mat src = imread("my.bmp", IMREAD_GRAYSCALE);                             //���� �� ������� �ҷ�����
	
	Mat noise(src.size(), CV_32SC1);                                          //���� �߰�
	randn(noise, 0, 10);                                                      //ǥ�������� 10�� ����
	Mat src1;
	add(src, noise, src1, Mat(), CV_8U);                                      //������ ���Ѱ� scr1�� ����
	namedWindow("dst");                                                       //Ʈ���� ������â �̸�
	createTrackbar("level", "dst", 0, 5, bilateral_change, (void*)&src1);     //Ʈ���� ���� �Լ�, Ʈ���ٰ� 5����

	waitKey();

	destroyAllWindows();
}
void bilateral_change(int pos, void* userdata)                                 //Ʈ����
{
	Mat src1 = *(Mat*)userdata;                                                //void*Ÿ�� ���� userdata�� 
	                                                                           //Mat*Ÿ������ ����ȯ�� �� src1������ ����
	Mat dst;             
	bilateralFilter(src1, dst, -1, pos+5, pos);                                //�������� ǥ������ pos+5,��ǥ���� ǥ������ pos�� ����� ���͸�
	imshow("dst", dst);                                                        //���â���
}

void filter_median()                                                           //�̵�� ���͸�
{
	Mat src = imread("my.bmp", IMREAD_GRAYSCALE);                              //���� �� ��� �ҷ�����

	int num = (int)(src.total() * 0.1);                                        //src���󿡼� 10%�ش��ϴ� �ȼ� ���� 0�Ǵ� 255�� ����(�ұ�&���� ����)
	for (int i = 0; i < num; i++) {
		int x = rand() % src.cols;
		int y = rand() % src.rows;
		src.at<uchar>(y, x) = (i % 2) * 255;
	}
	
	namedWindow("dst");                                                        //Ʈ���� ������ â �̸�
	createTrackbar("level", "dst", 0, 2, median_change, (void*)&src);          //Ʈ���� ���� �Լ�, Ʈ���ٰ� 2����
	waitKey();
	destroyAllWindows();
}
void median_change(int pos, void* userdata)                                    //Ʈ����
{
	Mat src = *(Mat*)userdata;                                                 //void*Ÿ�� ���� userdata�� 
	                                                                           //Mat*Ÿ������ ����ȯ�� �� src1������ ����
	if (pos%2==0)pos = pos + (pos+1);                                          //pos�� ¦���϶� ���� (0�϶� 1, 2�϶� 5�� ��ȯ)
	else pos = pos +2;                                                         //Ȧ���϶� ������ �°��ϱ����� 2����(1�϶� 3���� ��ȯ)
	
	Mat dst;
	medianBlur(src, dst, pos);                                                 //ũ�Ⱑ pos�� �̵�� ���� ����
	imshow("dst", dst);                                                        //��� ����
}

void sobel_edge()                                                              //����ũ��� ��������
{
	Mat src = imread("my.bmp", IMREAD_GRAYSCALE);                              //���� �� ��� �ҷ�����

	Mat dx, dy;                                                                //dx,dy���� ����
	Sobel(src, dx, CV_32FC1, 1, 0);                                            //x�� �������� 1�� ��̺� ���Ͽ� dx��Ŀ� ����
	Sobel(src, dy, CV_32FC1, 0, 1);                                            //y�� �������� 1�� ��̺� ���Ͽ� dy��Ŀ� ����

	Mat fmag, mag;
	magnitude(dx, dy, fmag);                                                   //dx,dy��ķκ��� �׷����Ʈ ũ�� ����Ͽ� fmag�� ����
	fmag.convertTo(mag, CV_8UC1);                                              //�Ǽ��� ��� fmag�� �׷��̽������������� ��ȯ�Ͽ� mag�� ����
	namedWindow("edge");                                                       //Ʈ���� ������â �̸�
	createTrackbar("level", "edge", 0, 5, sobel_change, (void*)&mag);          //Ʈ���� ���� �Լ�, Ʈ���� �� 5����
	
	imshow("src", src);
	waitKey();
	destroyAllWindows();
}
void sobel_change(int pos, void* userdata)
{
	Mat mag = *(Mat*)userdata;
	Mat edge = mag > pos*30;                                                    //���� �Ǻ��� ���� �׷����Ʈ ũ�� �Ӱ谪��
	                                                                            //pos*30���� �����Ͽ� ���� �Ǻ�
	                                                                            //��� edge�� ���Ұ��� mag��� ���� ���� pos*30���� ũ�� 255,
																				//������0���� ����
	imshow("edge", edge);                                                       //��� ��� 
}




void canny_edge()                                                               //ĳ�Ͽ�������
{
	Mat src = imread("my.bmp", IMREAD_GRAYSCALE);								//���� �� ��� �ҷ�����

	namedWindow("dst");                                                         //Ʈ���� ������â �̸�
	createTrackbar("level", "dst", 0, 5, canny_change, (void*)&src);            //Ʈ���� ���� �Լ�, Ʈ���ٰ� 5����
	
	imshow("src", src);                                                         //���� ����

	waitKey();
	destroyAllWindows();
}
void canny_change(int pos, void* userdata)                                      //Ʈ����
{
	Mat src = *(Mat*)userdata;                                                  //void*Ÿ�� ���� userdata�� 
	                                                                            //Mat*Ÿ������ ����ȯ�� �� src1������ ����
	Mat dst;
	Canny(src, dst, 50, pos * 30);                                              //���� �Ӱ谪�� 50, ���� �Ӱ谪�� pos*30���� �����Ͽ� ĳ�� ���� ����

	imshow("dst", dst);                                                         //��� ���
}

void hough_lines()                                                              //������ȯ��������
{
	Mat src = imread("my.bmp", IMREAD_GRAYSCALE);                               //���� �� ��� �ҷ�����

	Mat edge;                                          
	Canny(src, edge, 50, 150);													//ĳ�Ͽ�������⸦ �̿��Ͽ� ���� ���� ���� edge�� ����
	namedWindow("dst");															//Ʈ���� ������â �̸�
	createTrackbar("level", "dst", 0, 4, lines_change, (void*)&edge);           //Ʈ���� ���� �Լ�, Ʈ���� �� 4����

	imshow("src", src);
	waitKey(0);
	destroyAllWindows();
}
void lines_change(int pos, void* userdata)										//Ʈ����
{
	Mat edge = *(Mat*)userdata;													//void*Ÿ�� ���� userdata�� 
	                                                                            //Mat*Ÿ������ ����ȯ�� �� src1������ ����
	vector<Vec2f> lines;
	HoughLines(edge, lines, 1, CV_PI / 180, pos * 50);							//rho(�ȼ�����),theta(���� ����)�� ���� line�� ����

	Mat dst;
	cvtColor(edge, dst, COLOR_GRAY2BGR);                                        //edge�� BGR 3ä�� �÷� �������� ��ȯ�Ͽ� dst�� ����

	for (size_t i = 0; i < lines.size(); i++) {                                 //line ������ŭ for�� �ݺ�
		float rho = lines[i][0], theta = lines[i][1];                           
		float cos_t = cos(theta), sin_t = sin(theta);
		float x0 = rho * cos_t, y0 = rho * sin_t;
		float alpha = 1000;

		Point pt1(cvRound(x0 - alpha * sin_t), cvRound(y0 + alpha * cos_t));    //pt1��x0���� �ָ� �������ִ� �������� ����ǥ ����
		Point pt2(cvRound(x0 + alpha * sin_t), cvRound(y0 - alpha * cos_t));    //pt2��y0���� �ָ� �������ִ� �������� ����ǥ ����
		line(dst, pt1, pt2, Scalar(0, 0, 255), 2, LINE_AA);                     //����� ������ �β��� 2�� ������ �Ǽ����� �׸�
	}

	
	imshow("dst", dst);                                                         //��� ���
}




void hough_circles()     //������ȯ�
{
	Mat src = imread("my.bmp", IMREAD_GRAYSCALE);                               //���� �� ��� �ҷ�����

	Mat blurred;                                                                
	blur(src, blurred, Size(3, 3));                                             //��������, blurred�� ����
	namedWindow("dst");                                                         //Ʈ���� ������ â �̸�
	createTrackbar("level", "dst", 0, 5, circles_change, (void*)&blurred);      //Ʈ���� ���� �Լ�, Ʈ���� �� 5 ����

	waitKey(0);
	destroyAllWindows();
}
void circles_change(int pos, void* userdata)                                    //Ʈ����
{
	Mat blurred = *(Mat*)userdata;											    //void*Ÿ�� ���� userdata�� 
	                                                                            //Mat*Ÿ������ ����ȯ�� �� blurred������ ����
	Mat src = imread("my.bmp", IMREAD_GRAYSCALE);                               //Ʈ�����Լ� �ȿ��� ����ϱ����� �ҷ���
	vector<Vec3f> circles;                                                      
	HoughCircles(blurred, circles, HOUGH_GRADIENT, 1, pos*10, 150, 30);         //������,�������� ũ�� �����ϰ�,���߽ɰŸ�pos*10���� ������ ����x
																				//ĳ�Ͽ������� �����Ӱ谪 150,���� �Ӱ谪 30���� ����
																				//������ circles�� ����
	Mat dst;
	cvtColor(src, dst, COLOR_GRAY2BGR);                                         //�Է¿����� 3ä�� �÷��������� ��ȯ

	for (Vec3f c : circles) {                                                   //����� ���� ���������� �׸�
		Point center(cvRound(c[0]), cvRound(c[1]));
		int radius = cvRound(c[2]);
		circle(dst, center, radius, Scalar(0, 0, 255), 2, LINE_AA);
	}


	imshow("dst", dst);
}


void adaptive()                                                                  //����������ȭ
{
	Mat src = imread("my.bmp", IMREAD_GRAYSCALE);								 //���� �� ��� �ҷ�����

	imshow("src", src);

	namedWindow("dst");                                                          //Ʈ���� ������â �̸�
	createTrackbar("Block Size", "dst", 0, 200, adaptive_change, (void*)&src);   //Ʈ���� �����Լ�, Ʈ���ٰ� 200����
	setTrackbarPos("Block Size", "dst", 11);                                     //Ʈ���� �ʱ� ��ġ 11�� ����

	waitKey(0);
	
}

void adaptive_change(int pos, void* userdata)                                    //Ʈ����
{
	Mat src = *(Mat*)userdata;                                                   //void*Ÿ�� ���� userdata�� 
	                                                                             //Mat*Ÿ������ ����ȯ�� �� src������ ����

	int bsize = pos;
	if (bsize % 2 == 0) bsize--;                                                 //bsize���� ¦���̸� 1���� Ȧ����
	if (bsize < 3) bsize = 3;                                                    //bsize���� 3���� ������ 3���� ����

	Mat dst;
	adaptiveThreshold(src, dst, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY,  //Ʈ���ٿ��� ������ ���ũ�⸦ �̿��Ͽ� ������ ����ȭ ����
		bsize, 2);                                                               //����þ� ���� ��� ��� ��� ��տ��� 5 �� ���� �԰谪���� ���

	imshow("dst", dst);                                                          //��� ���
}




void erode_dilate()																 //�������� ħ��,��â
{
	Mat src = imread("my.bmp", IMREAD_GRAYSCALE);                                //���� �� ��� �ҷ�����

	Mat bin;
	threshold(src, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);                    //�Է¿����� �����˰������� �ڵ� ����ȭ ���� bin�� ����

	Mat dst1, dst2;
	erode(bin, dst1, Mat());                                                     //bin���� 3x3 ������ ���� ��Ҹ� �̿��Ͽ� ħ�� ���� ����
	dilate(bin, dst2, Mat());                                                    //bin���� 3x3 ������ ���� ��Ҹ� �̿��Ͽ� ��â ���� ����

	imshow("src", src);                                                          //����
	imshow("bin", bin);															 //����ȭ����
	imshow("erode", dst1);                                                       //ħ�ļ����� ����
	imshow("dilate", dst2);														 //��â ������ ����

	waitKey();
	destroyAllWindows();
}

void open_close()                                                                //�������� ����,�ݱ�
{
	Mat src = imread("my.bmp", IMREAD_GRAYSCALE);                                //���� �� ��� �ҷ�����

	Mat bin;
	threshold(src, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);                    //�Է¿����� �����˰������� �ڵ� ����ȭ ���� bin�� ����

	Mat dst1, dst2;
	morphologyEx(bin, dst1, MORPH_OPEN, Mat());                                  //���⿬��
	morphologyEx(bin, dst2, MORPH_CLOSE, Mat());                                 //�ݱ⿬��

	imshow("src", src);                                                          //����
	imshow("bin", bin);                                                          //����ȭ����
	imshow("opening", dst1);                                                     //���⿬�� ����
	imshow("closing", dst2);                                                     //�ݱ⿬�� ����

	waitKey();
	destroyAllWindows();
}




void labeling_stats()                                                           //���̺�
{
	Mat src = imread("my.bmp", IMREAD_GRAYSCALE);                               //���� �� ��� �ҷ�����

	Mat bin;
	threshold(src, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);                   //�Է¿����� �����˰������� �ڵ� ����ȭ ���� bin�� ����
	Mat dst1;
	morphologyEx(bin, dst1, MORPH_OPEN, Mat());                                 //���⿬�� dst1�� ����

	Mat labels, stats, centroids;                                               
	int cnt = connectedComponentsWithStats(dst1, labels, stats, centroids);     //���̺� ����,�� ��ü ������ ��������� ����

	Mat dst;
	cvtColor(dst1, dst, COLOR_GRAY2BGR);                                        //3ä�� �÷� ���� �������� ��ȯ�Ͽ� dst�� ����

	for (int i = 1; i < cnt; i++) {                                             //��� ���� ��� ��ü ������ ���ؼ��� for�ݺ��� ����
		int* p = stats.ptr<int>(i);

		if (p[4] < 20) continue;                                                //��ü �ȼ����� 20���� ������ �������� ����

		rectangle(dst, Rect(p[0], p[1], p[2], p[3]), Scalar(0, 255, 255));      //����� ��ü�� �ٿ�� �ڽ��� ��������� �׸�
	}


	imshow("src", src);
	imshow("dst", dst);

	waitKey();
	destroyAllWindows();
}



void contours_hier()                                                            //�ܰ�������
{
	Mat src = imread("my.bmp", IMREAD_GRAYSCALE);                               //���� �� ��� �ҷ�����
	namedWindow("dst1");                                                        //Ʈ���� ������ �̸�
	createTrackbar("Block Size", "dst1", 0, 200, hier_change, (void*)&src);     //Ʈ���� ���� �Լ� , Ʈ���� �� 200����
	setTrackbarPos("Block Size", "dst1", 11);                                   //Ʈ���� 11���� ����

	waitKey(0);
	destroyAllWindows();
}
void hier_change(int pos, void* userdata)										//Ʈ����
{
	Mat src = *(Mat*)userdata;													//void*Ÿ�� ���� userdata�� 
	                                                                            //Mat*Ÿ������ ����ȯ�� �� src1������ ����
	int bsize = pos;
	if (bsize % 2 == 0) bsize--;                                                //���� ¦���϶� -1����
	if (bsize < 3) bsize = 3;                                                   //���� 3����  ũ�� 3���� ����

	Mat dst;
	adaptiveThreshold(src, dst, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY,
		bsize, 2);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(dst, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);    //hierarchy���ڸ� �����Ͽ� ���� ������ ����

	Mat dst1;
	cvtColor(dst, dst1, COLOR_GRAY2BGR);

	for (int idx = 0; idx >= 0; idx = hierarchy[idx][0]) {                      //0�ܰ������� ���� ���� ������ ���� �ܰ����� �̵��ϸ鼭 for�� ����
		Scalar c(rand() & 255, rand() & 255, rand() & 255);
		drawContours(dst1, contours, idx, c, -1, LINE_8, hierarchy);           //hierarchy������ �����Ͽ� �ܰ��� �׸� ���β�-1�� ����(�ܰ��� ���� ä��)
	}
	imshow("dst1", dst1);
}