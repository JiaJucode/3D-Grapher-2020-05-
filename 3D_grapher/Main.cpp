
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<Eigen/Dense>
#include<iostream>
#include<cmath>
#define _USE_MATH_DEFINES
#include<math.h>

using namespace Eigen;
using namespace cv;

typedef Matrix<double, Dynamic, Dynamic> dmat;
typedef Matrix<int, 1, 4> sfI;
typedef Matrix<double, 3, 1> cdt;
typedef Matrix<double, 1, Dynamic> hvec;
typedef Matrix<cdt, 1, Dynamic> Pvec;
typedef Matrix<Point, 1, Dynamic> Ppvec;
typedef Matrix<sfI, 1, Dynamic> Svec;
typedef Matrix<cdt, Dynamic, Dynamic> matcd;

const double PI = M_PI;
bool FLIPZ = false;
int SCREEN_LENGTH = 1000, SCREEN_WIDTH = 1000;
double SIZE_MOLDIFIER = 0.03;
//Scalar background_color = Scalar(255, 255, 255);
const double CVALUE_P2 = 90000;
const double CVALUE = sqrt(CVALUE_P2);
double PREV_Z;
hvec PREV_XY(2);
Point origin(SCREEN_LENGTH / 2, SCREEN_WIDTH / 2);
double DX = 1, DY = 1;
double SIZE_CHANGE_MUTIPLIER = 1.5;

void Sort_Matrix(Svec& matrix2, hvec& matrix3, hvec matrix, int size)
{
	for (int counter = 1; counter != size; counter++)
	{
		int count = counter;
		while (matrix[count] > matrix[count - 1])
		{
			double buffer = matrix[count];
			matrix[count] = matrix[count - 1];
			matrix[count - 1] = buffer;
			buffer = matrix3[count];
			matrix3[count] = matrix3[count - 1];
			matrix3[count - 1] = buffer;
			sfI bufferI = matrix2[count];
			matrix2[count] = matrix2[count - 1];
			matrix2[count - 1] = bufferI;
			count--;
			if (count == 0) { break; }

		}
	}

}

inline double length(cdt vec)
{
	return sqrt(pow(vec[0], 2) + pow(vec[1], 2) + pow(vec[2], 2));
}

void Calculate_span(double x, double y, double z, dmat& space)
{
	if (x == 0 && y == 0)
	{
		space.col(0) = cdt(PREV_XY[0], PREV_XY[1], 0) / length(cdt(PREV_XY[0], PREV_XY[1], 0)) * (((FLIPZ == false) ^ (z < 0))? -1 : 1);
		space.col(1) = cdt(-PREV_XY[1], PREV_XY[0], 0) / length(cdt(-PREV_XY[1], PREV_XY[0], 0)) * (FLIPZ == false ? 1 : -1);

	}
	else {
		double length1 = length(cdt(x, y, z));
		cdt Perpendicular_vector = cdt(x / length1, y / length1, z / length1);
		//std::cout << prev_xy << std::endl << std::endl;
		cdt buffer;
		if (z == 0)
			buffer = cdt(-x, -y, 5);
		else
		{
			buffer = cdt(-x, -y, z > 0 ? z : -z * 1.1);
		}

		space.col(0) = buffer - buffer.dot(Perpendicular_vector) * Perpendicular_vector;
		space.col(0) = space.col(0) / length(space.col(0));
		//std::cout << buffer.dot(Perpendicular_vector) * Perpendicular_vector << std::endl << std::endl;
		if (FLIPZ == true)
		{
			space.col(0) *= -1;
		}
		dmat b(2, 3);
		b.row(0) = Perpendicular_vector.transpose();
		b.row(1) = space.col(0).transpose();
		FullPivLU<dmat> null_space2(b);
		if ((x <= y && FLIPZ == false) || (x >= y && FLIPZ == true)) {
			space.col(1) = null_space2.kernel() / length(null_space2.kernel()) * -1;
		}
		else
		{
			space.col(1) = null_space2.kernel() / length(null_space2.kernel());
		}
	}
}

void calculate_coordinate(double& x, double& y, double& z, double radian, double height)
{
	if ((PREV_XY[0] != 0 && PREV_XY[1] != 0) || (x != 0 || y != 0)) {
		PREV_XY << x, y;
	}
	z = height;
	double ratio = tan(radian);
	if (radian == PI / 2)
	{
		x = 0, y = sqrt(CVALUE_P2 - pow(z, 2));
	}
	else {
		if (radian == 3 * PI / 2)
		{
			x = 0, y = -sqrt(CVALUE_P2 - pow(z, 2));
		}
		else {
			if (radian == PI)
			{
				x = -sqrt(CVALUE_P2 - pow(z, 2)), y = 0;
			}
			else {
				if (radian == 0)
				{
					x = sqrt(CVALUE_P2 - pow(z, 2)), y = 0;
				}
				else {
					if (radian <= PI) {
						y = sqrt((CVALUE_P2 - pow(z, 2)) / (1 + 1 / pow(ratio, 2)));
						x = y / ratio;
					}
					else
					{
						x = sqrt((CVALUE_P2 - pow(z, 2)) / (1 + pow(ratio, 2)));
						y = x * ratio;
					}
				}
			}
		}

	}
	if ((radian > PI && radian < 3 * PI / 2) )
		x = -x, y = -y;

}

void plot_axis(dmat space, Mat img) 
{
	double max_value = 0.5 / (SIZE_MOLDIFIER*1.1);
	Pvec max_points(4);
	max_points << cdt(0, 0, 0), cdt(max_value,0,0), cdt(0, max_value, 0), cdt(0, 0, max_value);
	Ppvec proj_points(4);
	proj_points[0] = Point((space.transpose() * max_points[0])[1] * SCREEN_LENGTH * SIZE_MOLDIFIER + origin.x,
		(origin.y - (space.transpose() * max_points[0])[0] * SCREEN_WIDTH * SIZE_MOLDIFIER));
	proj_points[1] = Point((space.transpose() * max_points[1])[1] * SCREEN_LENGTH * SIZE_MOLDIFIER + origin.x,
		(origin.y - (space.transpose() * max_points[1])[0] * SCREEN_WIDTH * SIZE_MOLDIFIER));
	proj_points[2] = Point((space.transpose() * max_points[2])[1] * SCREEN_LENGTH * SIZE_MOLDIFIER + origin.x,
		(origin.y - (space.transpose() * max_points[2])[0] * SCREEN_WIDTH * SIZE_MOLDIFIER));
	proj_points[3] = Point((space.transpose() * max_points[3])[1] * SCREEN_LENGTH * SIZE_MOLDIFIER + origin.x,
		(origin.y - (space.transpose() * max_points[3])[0] * SCREEN_WIDTH * SIZE_MOLDIFIER));
	line(img, proj_points[0], proj_points[1], Scalar(255, 20, 20), 3);
	line(img, proj_points[0], proj_points[2], Scalar(20, 255, 20), 3);
	line(img, proj_points[0], proj_points[3], Scalar(20, 20, 255), 3);
}

void plot_squares(Ppvec new_edges, Svec Surfaces, hvec color, Mat img) 
{
	
	for (int spindex = 0; spindex != Surfaces.cols(); spindex++)
	{
		Point block_points[1][4];
		block_points[0][0] = new_edges[Surfaces[spindex][0]];
		block_points[0][1] = new_edges[Surfaces[spindex][1]];
		block_points[0][2] = new_edges[Surfaces[spindex][2]];
		block_points[0][3] = new_edges[Surfaces[spindex][3]];
		const Point* ppt[1] = { block_points[0] };
		int npt[] = { Surfaces[spindex].cols() };
		fillPoly(img,
			ppt,
			npt,
			1,
			Scalar(color[spindex], color[spindex], 0),
			LINE_8);
	}
}

inline double equation(double Ix, double Iy) 
{
	dmat cov(2, 2);
	cov << 0.0311993, 0.0413964, 0.0413964, 0.0565222;
	//cov << 0.0269761, 0.0392137, 0.0392137, 0.0570479;
	return (1 / (sqrt(cov.determinant()) * sqrt(2 * M_PI))) * exp(-0.5 * (Matrix<double,2,1>(Ix,Iy)).transpose() * cov.inverse() * (Matrix<double, 2, 1>(Ix, Iy)));
	/*
	if ((pow(Ix, 2) + pow(Iy, 2)) == 0) 
	{
		return cvalue_p2;
		system("pause");
	}
	else
		return -1/sqrt(pow(Ix,2)+pow(Iy,2));
	*/
	//3.985760576*pow(10,14)
}

void calculate_edges(matcd& edges) 
{

	int x=10, y=10;
	edges.resize(2*x+1, 2*y+1);
	matcd edgeaa(x, y);
	matcd edgeab(x, y);
	matcd edgeba(x, y);
	matcd edgebb(x, y);
	matcd x_edgep(1, y+1);
	matcd x_edgen(1, y);
	matcd y_edgep(x, 1);
	matcd y_edgen(x, 1);

	x_edgep(0, 0)=cdt(0,0,equation(0,0));
	for (double xindex = 1; xindex != edgeaa.rows() + 1; xindex++)
	{
		for (double yindex = 1; yindex != edgeaa.cols() + 1; yindex++)
		{
			edgeaa(edgeab.rows() - yindex, xindex-1) = cdt(xindex, yindex, equation(xindex*DX, yindex*DY));
			edgeab(yindex-1, xindex-1) = cdt(xindex, -yindex, equation(xindex*DX, -yindex*DY));
			edgeba(edgeab.cols() - yindex, edgeab.rows() -xindex) = cdt(-xindex, yindex, equation(-xindex*DX, yindex*DY));
			edgebb(yindex-1, edgeab.rows() - xindex) = cdt(-xindex, -yindex, equation(-xindex*DX,-yindex*DY));
		}
		x_edgep(0, xindex) = cdt(xindex, 0, equation(xindex*DX, 0));
		x_edgen(0, x_edgen.cols() - xindex) = cdt(-xindex, 0, equation(-xindex*DX, 0));
		y_edgep(x_edgen.cols() - xindex, 0) = cdt(0, xindex, equation(0, xindex*DY));
		y_edgen(xindex-1, 0) = cdt(0, -xindex, equation(0, -xindex*DY));


	}
	edges << edgeba, y_edgep, edgeaa, x_edgen, x_edgep, edgebb, y_edgen, edgeab;
	/*
	for (int xindex = 0; xindex != edges.rows() - 1; xindex++)
	{
		for (int yindex = 0; yindex != edges.cols() - 1; yindex++)
		{
			std::cout << " ( " << edges(yindex,xindex).transpose() << " ) ";
		}
		std::cout << std::endl;
	}
	*/

	
}

void calculate_surface(Svec& surfaces, matcd edges) 
{
	surfaces.resize((edges.cols() - 1) * (edges.rows() - 1));
	int surface_index = 0;
	for (int xindex = 1; xindex != edges.rows(); xindex++) 
	{
		for (int yindex = 1; yindex != edges.cols(); yindex++) 
		{
			surfaces(surface_index) = sfI(yindex + xindex * edges.cols(), yindex + (xindex-1) * edges.cols(), 
				(yindex-1) + (xindex-1) * edges.cols(), yindex-1 + xindex * edges.cols());
			surface_index++;
		}
	}
	/*
	for (int xindex = 0; xindex != edges.rows()-1; xindex++)
	{
		for (int yindex = 0; yindex != edges.cols()-1; yindex++)
		{
			std::cout << " ( " << surfaces(0, xindex*(edges.rows()-1)+yindex) << " ) ";
		}
		std::cout << std::endl;
	}
	*/
}

int main()
{
	double radian = 0, height = 0;
	double x = 0, y = 0, z = 0;
	while (true) {
		char key;
		matcd edges_b;
		Svec Surfaces;
		calculate_edges(edges_b);
		calculate_surface(Surfaces, edges_b);
		hvec color(Surfaces.cols());
		for (int xi = 0; xi != edges_b.rows() - 1; xi++)
		{
			for (int yi = 0; yi != edges_b.cols() - 1; yi++)
			{
				color(xi * (edges_b.rows() - 1) + yi) = 100 + ((yi < edges_b.cols() / 2) ? 8 * ((yi + xi) / 2 + 1) : 8 * (edges_b.cols() - 1 - (yi + edges_b.cols() - 1 - xi) / 2));
			}
		}
		//color << hvec::Ones(Surfaces.cols())*100;
		edges_b.transposeInPlace();
		Pvec edges(Map<Pvec>(edges_b.data(), edges_b.cols() * edges_b.rows()));
		/*
		Svec Surfaces(6);
		hvec order(Surfaces.cols());
		hvec color(6);
		Pvec edges(8);
		color << 0, 50, 100, 150, 200, 250;
		order << 0, 1, 2, 3, 4, 5;
		edges << cdt(2, 2, 2), cdt(2, 2, -2), cdt(2, -2, 2), cdt(2, -2, -2), cdt(-2, 2, 2), cdt(-2, 2, -2), cdt(-2, -2, 2), cdt(-2, -2, -2);
		Surfaces << sfI(0, 1, 3, 2), sfI(4, 5, 7, 6), sfI(0, 1, 5, 4), sfI(2, 3, 7, 6), sfI(0, 2, 6, 4), sfI(1, 3, 7, 5);
		*/
		while (true) {
			//std::cout << cdt(radian, height,0) << std::endl << std::endl;
			int lt = getTickCount();
			if (abs(height) == CVALUE) {
				calculate_coordinate(PREV_XY[0], PREV_XY[1], z, radian, PREV_Z);
				x = 0, y = 0, z = height;
			}
			else
				calculate_coordinate(x, y, z, radian, height);
			//std::cout << cdt(x, y, z) << std::endl;
			cdt viewer_cdt = cdt(x, y, z);
			hvec distance(edges.cols());
			distance = hvec::Zero(Surfaces.cols());
			for (int index = 0; index != Surfaces.cols(); index++)
			{
				double total_distance = 0;
				for (int sindex = 0; sindex != Surfaces[index].cols(); sindex++)
				{

					total_distance += sqrt(pow(edges[Surfaces[index][sindex]][0] - viewer_cdt[0], 2) + pow(edges[Surfaces[index][sindex]][1] - viewer_cdt[1], 2)
						+ pow(edges[Surfaces[index][sindex]][2] - viewer_cdt[2], 2));
				}
				distance[index] = total_distance / Surfaces[index].cols();
			}
			//std::cout << distance << std::endl;
			dmat space(3, 2);
			Calculate_span(x, y, z, space);
			//std::cout << space;
			Ppvec new_edges(edges.cols());
			for (int eindex = 0; eindex != edges.cols(); eindex++)
			{
				new_edges[eindex] = Point((space.transpose() * edges[eindex])[1] * SCREEN_LENGTH * SIZE_MOLDIFIER + origin.x,
					origin.y - (space.transpose() * edges[eindex])[0] * SCREEN_LENGTH * SIZE_MOLDIFIER);
			}
			//std::cout << std::endl << new_edges;
			Sort_Matrix(Surfaces, color, distance, Surfaces.cols());
			Mat img(SCREEN_LENGTH, SCREEN_WIDTH, CV_8UC3, Scalar(0,0,0));
			Mat imgp(SCREEN_LENGTH, SCREEN_WIDTH, CV_8UC3, Scalar(0,0,0));
			Mat img_b(SCREEN_LENGTH, SCREEN_WIDTH, CV_8UC3, Scalar(0, 0, 0));
			plot_squares(new_edges, Surfaces, color, imgp);
			plot_axis(space, img_b);
			img = imgp + img_b;
			cv::namedWindow("image", WINDOW_NORMAL);
			cv::imshow("image", img);
			key = (char)waitKey(0);
			switch (key)
			{
			case't':
			{
				SIZE_MOLDIFIER *= SIZE_CHANGE_MUTIPLIER;
				break;
			}
			case'g':
			{
				SIZE_MOLDIFIER /= SIZE_CHANGE_MUTIPLIER;
				break;
			}
			case'r':
			{
				DX *= SIZE_CHANGE_MUTIPLIER;
				break;
			}
			case'f':
			{
				DX /= SIZE_CHANGE_MUTIPLIER;
				break;
			}
			case'y':
			{
				DY *= SIZE_CHANGE_MUTIPLIER;
				break;
			}
			case'h':
			{
				DY /= SIZE_CHANGE_MUTIPLIER;
				break;
			}
			case'i':
			{
				origin.y -= SCREEN_WIDTH / 10;
				break;
			}
			case'k':
			{
				origin.y += SCREEN_WIDTH / 10;
				break;
			}
			case'j':
			{
				origin.x -= SCREEN_LENGTH / 10;
				break;
			}
			case'l':
			{
				origin.x += SCREEN_LENGTH / 10;
				break;
			}
			case 'w':
			{
				if (height >= -CVALUE && height + CVALUE / 10 <= CVALUE && FLIPZ == false)
				{
					if (height + CVALUE / 10 <= CVALUE)
						PREV_Z = z;
					height += CVALUE / 10;
				}
				else
				{
					if (height - CVALUE / 10 >= -CVALUE && height <= CVALUE && FLIPZ == true)
					{
						if (height - CVALUE / 10 <= -CVALUE)
							PREV_Z = z;
						height -= CVALUE / 10;
					}
					else {

						if (FLIPZ == false)
						{
							height = CVALUE * 2 - (height + CVALUE / 10);
							radian += (radian > PI) ? (-PI) : PI;
							FLIPZ = true;
						}
						else
						{
							height = -CVALUE * 2 - (height - CVALUE / 10);
							radian += (radian > PI) ? (-PI) : PI;
							FLIPZ = false;
						}
					}
				}
				break;
			}
			case 's':
			{
				if (height - CVALUE / 10 >= -CVALUE && height <= CVALUE && FLIPZ == false)
				{
					if (height - CVALUE / 10 <= -CVALUE)
						PREV_Z = z;
					height -= CVALUE / 10;
				}
				else
				{
					if (height >= -CVALUE && height + CVALUE / 10 <= CVALUE && FLIPZ == true)
					{
						if (height + CVALUE / 10 <= CVALUE)
							PREV_Z = z;
						height += CVALUE / 10;
					}
					else {

						if (FLIPZ == true)
						{
							height = CVALUE * 2 - (height + CVALUE / 10);
							radian += (radian > PI) ? (-PI) : PI;
							FLIPZ = false;
						}
						else
						{
							height = -CVALUE * 2 - (height - CVALUE / 10);
							radian += (radian > PI) ? (-PI) : PI;
							FLIPZ = true;
						}
					}
				}
				break;
			}
			case 'd':
			{
				if (radian + PI / 20 <= 2 * PI)
				{
					if (FLIPZ == false) {
						radian += PI / 20;
					}
					else
						radian -= PI / 20;
				}
				else
				{
					radian = radian + PI / 20 - 2 * PI;
				}
				break;
			}
			case 'a':
			{
				if (radian - PI / 20 >= 0)
				{
					if (FLIPZ == false)
						radian -= PI / 20;
					else
						radian += PI / 20;
				}
				else
				{
					radian = 2 * PI - (PI / 20 - radian);
				}
				break;
			}
		Default:
			break;
			}
			if (key == 27 || key == 'r' || key == 'f' || key == 'y' || key == 'h')
				break;
		}
		if (key == 27) 
			break;
	}
	return EXIT_SUCCESS;
}
