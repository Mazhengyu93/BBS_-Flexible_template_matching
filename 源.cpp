#include<opencv2/opencv.hpp>
//#include<valarray>
//#include<numeric>
#include<Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include<vector>
#include <Eigen/Sparse>
#include<valarray>
#include<numeric>
#include<unsupported/Eigen/CXX11/Tensor>
#if (_MSC_VER >= 1915)
#define no_init_all deprecated
#endif
using namespace Eigen;
int grid_number=3;
double gamma = 2.0;
const int template_patch_number = 15 * 11;
const int image_patch_number = 75 * 75;
const int S_patch_X = 75;// image_cols / grid_number;
const int S_patch_Y = 75;// image_rows / grid_number;
const int T_patch_X = 15;// template_cols / grid_number;
const int T_patch_Y = 11;// template_rows / grid_number;
Eigen::Matrix <double, template_patch_number, template_patch_number>D;
int main()
{
	int grid_number_square = grid_number * grid_number;
	cv::Mat template_ = cv::imread("D:\\11�·��������\\star_test.jpg");
	cv::Mat image_ = cv::imread("D:\\11�·��������\\test_NCC_SSD.jpg");
	int template_cols = template_.cols;
	int template_rows = template_.rows;
	Eigen::Matrix <int, Dynamic, Dynamic,RowMajor>template_grid_index= VectorXi::LinSpaced(T_patch_X*T_patch_Y, 0, T_patch_X*T_patch_Y);
	//�����Լ�������
	template_grid_index.resize(T_patch_X, T_patch_Y);
	template_grid_index.transposeInPlace();
	int image_cols = image_.cols;
	int image_rows = image_.rows;
	Eigen::Matrix <int, Dynamic, Dynamic, RowMajor>image_grid_index = VectorXi::LinSpaced(S_patch_X*S_patch_Y, 0, S_patch_X*S_patch_Y);
	//�����Լ�������
	image_grid_index.resize(S_patch_X, S_patch_Y);
	image_grid_index.transposeInPlace();
	//std::cout << image_grid_index;
	//����template��image,ע������ͨ����
	//��eigen��,matrix�Ƕ�ά�ģ�tensor�Ƕ�ά�ģ�
	std::vector<cv::Mat> template_channels(3), image_channels(3);
	cv::split(template_, template_channels);
	cv::split(image_, image_channels);
	std::vector<Eigen::Matrix<double, Dynamic, Dynamic>> template_eigen_channels(3),image_eigen_channels(3);
	for (int i = 0; i < template_channels.size(); i++)
	{
		cv2eigen(template_channels[i], template_eigen_channels[i]);
		template_eigen_channels[i]=(template_eigen_channels[i].array())/ 255;
		cv2eigen(image_channels[i], image_eigen_channels[i]);
		image_eigen_channels[i] = image_eigen_channels[i].array()/ 255;
	}

	
	std::vector<Eigen::Matrix<double, 9, template_patch_number>>template_feature_map_channel(3);
	std::vector<Eigen::Matrix<double, 9, image_patch_number>>image_feature_map_channel(3);
	for (int i = 0; i < template_eigen_channels.size(); i++)
	{
		int count_T = 0;
		for (int m = 0; m < template_eigen_channels[i].cols() - grid_number + 1; m = m + grid_number)
		   for (int n = 0; n < template_eigen_channels[i].rows()- grid_number+1; n = n + grid_number)
		  
			{
				Eigen::Matrix<double, Dynamic, Dynamic>temp = template_eigen_channels[i].block(n,m, grid_number, grid_number);
				temp.resize(grid_number_square, 1);
				template_feature_map_channel[i].col(count_T)=temp.col(0);
				count_T++;
			}
	}
	for (int i = 0; i < image_eigen_channels.size(); i++)
	{
		int count_T = 0;
		for (int m = 0; m < image_eigen_channels[i].cols() - grid_number + 1; m = m + grid_number)
		   for (int n = 0; n < image_eigen_channels[i].rows() - grid_number + 1; n = n + grid_number)
			
			{
				Eigen::Matrix<double, Dynamic, Dynamic>temp = image_eigen_channels[i].block(n, m, grid_number, grid_number);
				temp.resize(grid_number_square, 1);
				image_feature_map_channel[i].col(count_T) = temp.col(0);
				count_T++;
			}
	}




	//���������feature_Map����������
	//���潨����˹��
	cv::Mat cv_guass_x=cv::getGaussianKernel(grid_number,0.6);
	cv::Mat cv_guass_y = cv::getGaussianKernel(grid_number, 0.6);
	cv::Mat cv_guass = cv_guass_x * cv_guass_y.t();
	cv_guass=cv_guass.reshape(0, grid_number_square);
	//��ɸ�Template patch������С��ͬ�ľ��󣬷������ֱ�����
	cv_guass=cv::repeat(cv_guass,1,template_patch_number);
	Eigen::MatrixXd FMat;
	cv2eigen(cv_guass,FMat);
	//��˹��������Ժ�ʼ��������
	//TMat ��С�ľ���֮��ľ���
	Eigen::Matrix <int, Dynamic, Dynamic> distance_index_= VectorXi::LinSpaced(image_patch_number, 0, image_patch_number);
	distance_index_.resize(S_patch_Y,S_patch_X);
	Eigen::Matrix <double, Dynamic, Dynamic> xx= VectorXd::LinSpaced(T_patch_X, 0, T_patch_X-1)*0.0039*grid_number;
	xx.transposeInPlace();
	Eigen::Matrix <double, Dynamic, Dynamic >xx_=xx.replicate(T_patch_Y, 1);
	Eigen::Matrix <double, Dynamic, Dynamic> yy = VectorXd::LinSpaced(T_patch_Y, 0, T_patch_Y-1)*0.0039*grid_number;
	Eigen::Matrix <double, Dynamic, Dynamic>yy_ = yy.replicate(1, T_patch_X);
	Eigen::MatrixXd X = xx_;
	X.resize(template_patch_number, 1);
	Eigen::MatrixXd Y = yy_;
	Y.resize(template_patch_number, 1);

	Eigen::Matrix <double, template_patch_number, template_patch_number>Dxy;
	for (int i = 0; i < template_patch_number; i++)
	{
		Dxy.col(i) = (X.col(0).array() - X(i,0)).square() + (Y.col(0).array() - Y(i,0)).square();
	}
	//������imageΪĿ�꣬һ��patch��һ��patch�ƶ�������ÿ��λ��ÿ��ͨ����Ϊ����ڵ�patch�ĸ���
	//�����������ݸ�ʽ������

	Eigen::Matrix<double, template_patch_number, template_patch_number>Drgb;
	std::vector<Eigen::Matrix<double, template_patch_number, template_patch_number>>Drgb_buffer(image_cols - template_cols);
	cv::Mat Drgb_test;
	//�������б���image�ϵ�ÿ��patch
	Eigen::Matrix<int,S_patch_Y, S_patch_X>BBS;
	BBS.setZero();
	
	for(int colI=0; colI <S_patch_X-T_patch_X+1; colI++)
		for (int rowI = 0; rowI < S_patch_Y-T_patch_Y+1; rowI++)
		{		
			if (rowI == 0 && colI == 0)
			{
				//������block�����ѷ���template��ͬ��С��һ��patch��indexȡ����				
				//ind = IndMat(rowI:rowI + szT(1) / pz - 1, rowI : rowI + szT(2) / pz - 1);
				Eigen::Matrix<int,Dynamic,Dynamic> ind=image_grid_index.block(rowI,colI,T_patch_Y,T_patch_X);
				ind.resize(template_patch_number,1);
				std::vector<Eigen::Matrix<double,9,template_patch_number>>PMat(3);
				for (int C = 0; C < 3; C++)
				{ 
					for (int i = 0; i < template_patch_number; i++)
						PMat[C].col(i)=image_feature_map_channel[C].col(ind(i,0));
				}
				std::vector<Eigen::Matrix<double, 9, template_patch_number>>temp(3);
				for (int j = 0; j < template_patch_number; j++)
				{
					for (int C = 0; C < 3; C++)
					{
						//	PMat.replicate(1, 2);
						Eigen::VectorXd PMat_sub_v = PMat[C].col(j);
						Eigen::MatrixXd PMat_sub = PMat_sub_v.replicate(1, template_patch_number);
						temp[C] = (PMat_sub - template_feature_map_channel[C]).array()*(FMat.array()); //*FMat;
						temp[C] = temp[C].array()*temp[C].array();
					}
					//����ͨ�����
					Eigen::Matrix<double, 9, template_patch_number>temp_sum = temp[0] + temp[1] + temp[2];
					//ÿ��patch��ֵ���
					Drgb.col(j) = temp_sum.colwise().sum();
				}
			
			}
			else if (rowI > 0 && colI == 0)
			{
				Drgb.block(0, 0, template_patch_number, template_patch_number - 1) =Drgb.block(0,1,template_patch_number, template_patch_number - 1);
				Eigen::Matrix<int, Dynamic, Dynamic> ind = image_grid_index.block(rowI+T_patch_Y-1, colI, 1, T_patch_X);
				
				ind.resize(T_patch_X, 1);
				std::vector<Eigen::Matrix<double, 9, T_patch_X>>R(3);
				for (int C = 0; C < 3; C++)
				{
					for (int i = 0; i < T_patch_X; i++)
						R[C].col(i) = image_feature_map_channel[C].col(ind(i, 0));
				}
				std::vector<Eigen::Matrix<double, 9, template_patch_number>>temp(3);
				int idxP = T_patch_Y - 1;
				for (int j = 0; j < T_patch_X; j++)
				{
					
					for (int C = 0; C < 3; C++)
					{
						
						Eigen::VectorXd RMat_sub_v = R[C].col(j);
						Eigen::MatrixXd RMat_sub = RMat_sub_v.replicate(1, template_patch_number);
						temp[C] = (RMat_sub - template_feature_map_channel[C]).array()*(FMat.array()); //*FMat;
						temp[C] = temp[C].array()*temp[C].array();
					}
					//����ͨ�����
					Eigen::Matrix<double, 9, template_patch_number>temp_sum = temp[0] + temp[1] + temp[2];
					//ÿ��patch��ֵ���
					Drgb.col(idxP) = temp_sum.colwise().sum();
					idxP = idxP + T_patch_Y;
				}
				
			}
			else if (rowI == 0 && colI > 0)
			{
				Eigen::Matrix<double, template_patch_number, template_patch_number>Drgb_prev = Drgb_buffer[0];
				Drgb.block(0, 0, template_patch_number, template_patch_number - T_patch_Y) = Drgb_prev.block(0, T_patch_Y, template_patch_number, template_patch_number - T_patch_Y);
				Eigen::Matrix<int, Dynamic, Dynamic> ind = image_grid_index.block(rowI, colI+T_patch_X-1, T_patch_Y, 1);
				ind.resize(T_patch_Y, 1);
				std::vector<Eigen::Matrix<double, 9, T_patch_Y>>R(3);
				for (int C = 0; C < 3; C++)
				{
					for (int i = 0; i < T_patch_Y; i++)
						R[C].col(i) = image_feature_map_channel[C].col(ind(i, 0));
				}
				std::vector<Eigen::Matrix<double, 9, template_patch_number>>temp(3);
				int idxP = (T_patch_X - 1)*T_patch_Y;
				for (int j = 0; j < T_patch_Y; j++)
				{
					
					for (int C = 0; C < 3; C++)
					{
						Eigen::VectorXd RMat_sub_v = R[C].col(j);
						Eigen::MatrixXd RMat_sub = RMat_sub_v.replicate(1, template_patch_number);
						temp[C] = (RMat_sub - template_feature_map_channel[C]).array()*(FMat.array()); //*FMat;
						temp[C] = temp[C].array()*temp[C].array();
					}
					//����ͨ�����
					Eigen::Matrix<double, 9, template_patch_number>temp_sum = temp[0] + temp[1] + temp[2];
					//ÿ��patch��ֵ���
					Drgb.col(idxP) = temp_sum.colwise().sum();
					idxP = idxP + 1;
				}
				
				
			}
			else
			{
				Drgb.block(0, 0, template_patch_number, template_patch_number - 1) = Drgb.block(0, 1, template_patch_number, template_patch_number - 1);
				Eigen::Matrix<double, template_patch_number, template_patch_number>Drgb_prev = Drgb_buffer[rowI];
				for (int k = T_patch_Y - 1; k < template_patch_number - 1; k = k + T_patch_Y)
				{
					Drgb.col(k)=Drgb_prev.col(k+T_patch_Y);
				}
				
				std::vector<Eigen::Matrix<double, 9, 1>>R(3);
				for (int C = 0; C < 3; C++)
				{
					
						R[C].col(0) = image_feature_map_channel[C].col(image_grid_index(rowI+T_patch_Y-1,colI+T_patch_X-1));
				}
				std::vector<Eigen::Matrix<double, 9, template_patch_number>>temp(3);
				
					for (int C = 0; C < 3; C++)
					{

						Eigen::VectorXd RMat_sub_v = R[C].col(0);
						Eigen::MatrixXd RMat_sub = RMat_sub_v.replicate(1, template_patch_number);
						temp[C] = (RMat_sub - template_feature_map_channel[C]).array()*(FMat.array()); //*FMat;
						temp[C] = temp[C].array()*temp[C].array();
					}
					//����ͨ�����
					Eigen::Matrix<double, 9, template_patch_number>temp_sum = temp[0] + temp[1] + temp[2];
					//ÿ��patch��ֵ���
					Drgb.col(template_patch_number-1) = temp_sum.colwise().sum();
			
			}
			Drgb_buffer[rowI]=Drgb;			
			D = gamma * Dxy.array() + Drgb.array();
			D= (D.array()> 1e-4).cast<double>() *D.array();
			//ÿ��patch��9������λ��һ��165��patch������patch�˴���������õ�Drgb,����λ�þ���õ�D
			//�ҵ��˴˵��������Ҫ
			//����ȡ��Сֵ��A patch��B patch����ģ�ÿ�п��Եõ�һ����i,j)
			//����ȡ��Сֵ��B��patch��A��patch����Сֵ��ÿ�п��Եõ�(m,n)
			//�����(i,j)==(m,n)����ճ�һ��
			MatrixXd::Index minRow[template_patch_number], minCol[template_patch_number];
			for (int i = 0; i < template_patch_number; i++)
			{
				D.col(i).minCoeff(&minCol[i]);
				D.row(i).minCoeff(&minRow[i]);
			}
			int nearest_patch_number = 0;
			for (int i = 0; i < template_patch_number; i++)
			{
				if (minRow[minCol[i]] == i)
					nearest_patch_number++;
			}
			BBS(rowI, colI) = nearest_patch_number;
		}
	Eigen::MatrixXd::Index colMax, rowMax;
	BBS.maxCoeff(&rowMax, &colMax);
	//��������BBS
	cv::Mat BBS_mat;
	cv::eigen2cv(BBS,BBS_mat);
	double inMaxVal, inMinVal;
	cv::minMaxLoc(BBS_mat, &inMinVal, &inMaxVal, NULL, NULL);
	// ���ͼ��������Сֵ
	double outMaxVal = 255, outMinVal = 0;
	// ���� alpha �� b
	double alpha = (outMaxVal - outMinVal) / (inMaxVal - inMinVal);
	double b = outMinVal - alpha * inMinVal;
	cv::convertScaleAbs(BBS_mat, BBS_mat, alpha, b);
	int rowMax_ = rowMax;
	int colMax_ = colMax;
	cv::rectangle(image_, cv::Point(colMax*grid_number, rowMax_*grid_number), cv::Point(colMax*grid_number+template_cols, rowMax_*grid_number + template_rows), cv::Scalar(255, 255, 0));
	return 1;
}