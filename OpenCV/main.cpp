#include <opencv\cv.h>
#include <opencv\highgui.h>

using namespace cv;

void main(){

	Mat img = imread("Ball.jpg");
	Mat gray(img.rows, img.cols, CV_8UC1);	//얘는 원을 찾기위해서 흑백으로 바꾼거(주의할건 Tresholding이랑 흑백화는 다른개념)
	Mat result(img.rows, img.cols, CV_8UC3, Scalar(0, 0, 0));	//이건 최종적으로 그림그릴것, 3채널짜리 검은색으로 초기화
	
	cvtColor(img, gray, CV_BGR2GRAY);	//원을 찾기위해(HoughCircle을 쓰기위해) 흑백으로 바꿈)

	GaussianBlur(gray, gray, Size(9, 9), 2, 2);	//이걸하면 원을 더 정확하게 찾아(블러준거인데 그중 가우시안블러를쓴거)

	vector<Vec3f> circles;	//HoughCircle은 3짜리벡터에 x,y,r 순으로 집어넣어서 집어넣을 변수를 선언한것

	HoughCircles(gray, circles, CV_HOUGH_GRADIENT, 1, gray.rows / 8, 200, 100, 0, 0);	//각 파라미트에 대한 정보는 인터넷검색해서 봐바

	for (size_t i = 0; i < circles.size(); i++)	//원을 찾은 갯수만큼
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));	//원의 중심 x,y
		int radius = cvRound(circles[i][2]);	//원의 반지름 r
		
		circle(result, center, radius, Scalar(255,255,255), CV_FILLED); // 이것은 최종적으로 그려질 그림에 찾은 원들을 하얗게 그려주는 코드

		//색깔을 bgr로 선언되었으니 bgr로 읽어오기(사실 여기선 원의 중심을 찾았는데 원래라면 원 전체의 평균이나 확률을 사용하여 얻어야함)
		int b = img.at<Vec3b>(center.y, center.x)[0];
		int g = img.at<Vec3b>(center.y, center.x)[1];
		int r = img.at<Vec3b>(center.y, center.x)[2];

		circle(result, center, radius, Scalar(b, g, r), 3, 8, 0);	//이것은 테두리 원을 그리는 코드인데 색깔을 아까 찾은 bgr을 사용
	}

	imshow("display", result);
	waitKey(0);
}