#pragma once
#include <string>

class My_Bmp {

private:

	//イメージサイズ
	int im_x;
	int im_y;
	
	
	
public:
	//イメージデータの先頭ポインタ
	unsigned char* img;

	My_Bmp(int sx, int sy);   //コンストラクタ
	void img_read(std::string imgpath);           //BMP読み込み
	void ucimg_to_double(double* data_out);       //読み込んだunsigned charをdoubleにして格納
	
	//書き込みたいデータを256階調化、unsigned charに変換後imgに格納
	//多重定義
	void data_to_ucimg(double* data_in);
	void data_to_ucimg(int* data_in);
	void data_to_ucimg(float* data_in);
	
	void uc_to_img(unsigned char* data_in);       //unsigned charをimgに格納
	void img_write(std::string imgpath);          //BMP書き込み
	~My_Bmp();                                    //デコンストラクタ

};

