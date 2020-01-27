#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define __CL_ENABLE_EXCEPTIONS

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <iostream>
#include <fstream>
#include "draw.cpp" 
#include "cl.hpp"
#include "util.hpp"
#include "err_code.h"

// pick up device type from compiler command line or from the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

#define STRLEN 32
#define LONGSTRLEN 256

struct Point{
  double x;
  double y;
  double z;
  double color;
};

struct Triangle{
  int point1;
  int point2;
  int point3;
  double normX;
  double normY;
  double normZ;
  double textureA;
  double textureB;
};

//-----------------------------------------------------------------------------

void set(std::vector<double>* M, int columns, int i, int j, double value){
  M->at(i*columns + j) = value;
}

double get(std::vector<double>* M, int columns, int i, int j){
  return M->at(i*columns + j);
}

void readM(FILE* f,  std::vector<double>* M, int r, int c)
{
  char str[STRLEN];

  for (int i=0 ; i<r ; i++)
    for (int j=0 ; j<c ; j++)
    {
      fscanf(f,"%s",&str);
      set(M,c,i,j,atof(str));
    }
}

void writeM(FILE* f, std::vector<double>* M, int r, int c)
{
  char str[STRLEN];

  for (int i=0 ; i<r ; i++)
  {
    for (int j=0 ; j<c ; j++)
    {
      sprintf(str, "%.6f", get(M,c,i,j));
      fprintf(f,"%s ",str);
    }
    fprintf(f,"\n");
  }
}

void readGISInfo(FILE* f, int &r, int &c, double &xllcorner, double &yllcorner, double &cellsize, double &nodata)
{
  char str[STRLEN];
  int cont = -1;
  fpos_t position;

  //Reading the header
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); c = atoi(str);         //ncols
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); r = atoi(str);         //nrows
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); xllcorner = atof(str); //xllcorner
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); yllcorner = atof(str); //yllcorner
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); cellsize = atof(str);  //cellsize
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); nodata = atof(str);    //NODATA_value 

  //Checks if actually there are ncols x nrows values into the file
  fgetpos (f, &position);
  while (!feof(f))
  {
    fscanf(f,"%s",&str);
    cont++;
  }
  fsetpos (f, &position);
  /*if (r * c != cont)
  {
    printf("File corrupted\n");
    exit(0);
  }*/
}

void saveGISInfo(FILE* f, int r, int c, double xllcorner, double yllcorner, double cellsize, double nodata)
{
  char str[STRLEN];

  //ncols
  fprintf(f,"ncols\t\t");
  sprintf(str,"%d", c);
  fprintf(f,"%s\n", str);
  //nrows
  fprintf(f,"nrows\t\t");
  sprintf(str,"%d", r);
  fprintf(f,"%s\n", str);
  //xllcorner
  fprintf(f,"xllcorner\t");
  sprintf(str,"%f", xllcorner);
  fprintf(f,"%s\n", str);
  //yllcorner
  fprintf(f,"yllcorner\t");
  sprintf(str,"%f", yllcorner);
  fprintf(f,"%s\n", str);
  //cellsize
  fprintf(f,"cellsize\t");
  sprintf(str,"%f", cellsize);
  fprintf(f,"%s\n", str);
  //NODATA_value
  fprintf(f,"NODATA_value\t");
  sprintf(str,"%f", nodata);
  fprintf(f,"%s\n", str);
}

void loadM(char* path, int &r, int &c, double &xllcorner, double &yllcorner, double &cellsize, double &nodata,  std::vector<double> *M)
{
  FILE *f;

  if ( (f = fopen(path,"r") ) == 0){
    printf("Configuration not found\n");
    exit(0);
  }

  readGISInfo(f, r, c, xllcorner, yllcorner, cellsize, nodata);
  M->resize(r * c);
  readM(f, M, r, c);

  fclose(f);
}

void saveM(char* path, int r, int c, double xllcorner, double yllcorner, double cellsize, double nodata, std::vector<double>* M)
{
  FILE *f;
  char str[STRLEN];
  char out_path[LONGSTRLEN];

  strcpy(out_path, path);
  strcat(out_path, ".out");

  f = fopen(out_path, "w");

  saveGISInfo(f, r, c, xllcorner, yllcorner, cellsize, nodata);
  writeM(f, M, r, c);

  fclose(f);
}

void globalTransitionFunction(std::vector<double>* h_Sz, std::vector<double>* h_Sh, std::vector<double>* h_So, double dumping_factor, int r, int c, double nodata, int steps)
{
  cl::Buffer d_Sz;                    
  cl::Buffer d_Sh;    
  cl::Buffer d_So;

  try 
    {
    	// Create a context
        cl::Context context(DEVICE);
        
        // Load in kernel source, creating a program object for the context

        cl::Program program(context, util::loadProgram("../include/flow.cl"), true);

        // Get the command queue
        cl::CommandQueue queue(context);

        // Create the kernel functor
 
        auto kernel1 = cl::make_kernel<int, int, cl::Buffer, cl::Buffer, cl::Buffer, double, double>(program, "kernel1_outflow_computation");
        auto kernel2 = cl::make_kernel<int, int, double, cl::Buffer, cl::Buffer, cl::Buffer>(program, "kernel2_mass_balance");
        auto kernel3 = cl::make_kernel<int, int, double, cl::Buffer, cl::Buffer>(program, "kernel3_outflow_reset");

        d_Sz   = cl::Buffer(context, begin(*h_Sz), end(*h_Sz), true);
        d_Sh   = cl::Buffer(context, begin(*h_Sh), end(*h_Sh), false);

        d_So  = cl::Buffer(context, begin(*h_So), end(*h_So), false);

        util::Timer timer;

        for(int i = 0; i < steps; i++){
          
            kernel1(
                cl::EnqueueArgs(
                    queue,
                    cl::NDRange(r, c)),
                c,
                r,
                d_Sz,
                d_Sh,
                d_So,
                nodata,
                dumping_factor);

           kernel2(
                cl::EnqueueArgs(
                    queue,
                    cl::NDRange(r, c)),
                c,
                r,
                nodata,
                d_Sz,
                d_Sh,
                d_So
                );

            kernel3(
                cl::EnqueueArgs(
                    queue,
                    cl::NDRange(r, c)),
                c,
                r,
                nodata,
                d_Sz,
                d_So
                );
        }

        cl::copy(queue, d_Sh, begin(*h_Sh), end(*h_Sh));
        
        queue.finish();

        double rtime = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
        printf("The kernels ran in %lf seconds\n", rtime);
    }
    catch (cl::Error err) {
        std::cout << "Exception\n";
        std::cerr 
            << "ERROR: "
            << err.what()
            << "("
            << err_code(err.err())
           << ")"
           << std::endl;
           
    }
}

void createPoints(std::vector<double>* Sz, std::vector<double>* Sh, int r, int c, double cellsize, double nodata, Point* &points, Point* &pointsLava, double minimumColorLava, double maxmimumColorLava)
{
  double minimumColor = 0;
  double maxmimumColor = 0;

  double minimumAbsolute = 0;
  double maxmimumAbsolute = 0;

  for(int i = 1; i < r*c; i++){
    if(Sz->at(i) != nodata){
      if(Sz->at(i) < minimumColor)
        minimumColor = Sz->at(i);
      if(Sz->at(i) > maxmimumColor)
        maxmimumColor = Sz->at(i);
    }
  }

  if(minimumColor < minimumAbsolute)
    minimumAbsolute = minimumColor;

  if(maxmimumColor > c * cellsize){
    if(maxmimumColor > r * cellsize){
      maxmimumAbsolute = maxmimumColor;
    }else{
      maxmimumAbsolute = r * cellsize;
    }
  }else if(c * cellsize > maxmimumAbsolute){
    maxmimumAbsolute = c * cellsize;
  }

  for(int i = 0; i < r*c; i++){
    Point point;
    point.x = ((i % c) * cellsize) / (c * cellsize);
    point.y = ((floor(i/c)) * cellsize) / (r * cellsize);
    point.z = (Sz->at(i) - minimumAbsolute) / (maxmimumAbsolute - minimumAbsolute);

    if(Sz->at(i) != nodata){
      point.color = (Sz->at(i) - minimumColor) / (maxmimumColor - minimumColor);
    }else{
      
      point.color = -1;
    }
    points[i] = point;

    Point pointLava;
    pointLava.x = ((i % c) * cellsize) / (c * cellsize);
    pointLava.y = ((floor(i/c)) * cellsize) / (r * cellsize);
    pointLava.z = (Sh->at(i) - minimumAbsolute) / (maxmimumAbsolute - minimumAbsolute) + point.z;

    if(Sh->at(i) != nodata && Sh->at(i) != 0){
      pointLava.color = (Sh->at(i) - minimumColorLava) / (maxmimumColorLava - minimumColorLava);
    } else if(Sh->at(i) == 0){
      pointLava.color = 0;
    }else{
      pointLava.color = -1;
    }
    pointsLava[i] = pointLava;
  }
}

void createTriangles(Point* points, Point* pointsLava, std::vector<Triangle> &triangles, std::vector<Triangle> &trianglesLava, double cellsize, int r, int c){
  for(int i = 0; i < r*c; i++){
    if((i + 1) % c != 0 && (i + 1) < r * (c - 1) && points[i].color != -1 && i + c + 1 <= r * c - 1){
      Point point1 = points[i];
      bool boolPoint2a2b = false;
      bool boolPoint3a = false;
      bool boolPoint3b = false;

      if(points[i + c].color != -1){
        boolPoint3a = true;
      }
      if(points[i + c + 1].color != -1){
        boolPoint2a2b = true;
      }
      if(points[i + 1].color != -1){
        boolPoint3b = true;
      }

      if(boolPoint2a2b){
        if(boolPoint3a){
          Triangle triangle;
          triangle.point1 = i;
          triangle.point2 = i + c + 1;
          triangle.point3 = i + c;

          double x1 = points[i].x;
          double y1 = points[i].y;
          double z1 = points[i].z;

          double x2 = points[i + c + 1].x;
          double y2 = points[i + c + 1].y;
          double z2 = points[i + c + 1].z;

          double x3 = points[i + c].x;
          double y3 = points[i + c].y;
          double z3 = points[i + c].z;

          triangle.normX = (y2-y1)*(z3-z1) - (z2-z1)*(y3-y1);
          triangle.normY = (z2-z1)*(x3-x1) - (x2-x1)*(z3-z1);
          triangle.normZ = (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1);

          triangles.push_back(triangle);

          if(pointsLava[i].color != -1 && pointsLava[i].color != 0){
            Triangle triangleLava;
            triangleLava.point1 = i;
            triangleLava.point2 = i + c + 1;
            triangleLava.point3 = i + c;

            double x1 = pointsLava[i].x;
            double y1 = pointsLava[i].y;
            double z1 = pointsLava[i].z;

            double x2 = pointsLava[i + c + 1].x;
            double y2 = pointsLava[i + c + 1].y;
            double z2 = pointsLava[i + c + 1].z;

            double x3 = pointsLava[i + c].x;
            double y3 = pointsLava[i + c].y;
            double z3 = pointsLava[i + c].z;

            triangleLava.normX = (y2-y1)*(z3-z1) - (z2-z1)*(y3-y1);
            triangleLava.normY = (z2-z1)*(x3-x1) - (x2-x1)*(z3-z1);
            triangleLava.normZ = (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1);

            trianglesLava.push_back(triangleLava);
          }
        }
        if(boolPoint3b){
          Triangle triangle;
          triangle.point1 = i;
          triangle.point2 = i + c + 1;
          triangle.point3 = i + 1;
          
          double x1 = points[i].x;
          double y1 = points[i].y;
          double z1 = points[i].z;

          double x2 = points[i + c + 1].x;
          double y2 = points[i + c + 1].y;
          double z2 = points[i + c + 1].z;

          double x3 = points[i + 1].x;
          double y3 = points[i + 1].y;
          double z3 = points[i + 1].z;

          triangle.normX = (y2-y1)*(z3-z1) - (z2-z1)*(y3-y1);
          triangle.normY = (z2-z1)*(x3-x1) - (x2-x1)*(z3-z1);
          triangle.normZ = (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1);

          triangles.push_back(triangle);

          if(pointsLava[i].color != -1 && pointsLava[i].color != 0){
            Triangle triangleLava;
            triangleLava.point1 = i;
            triangleLava.point2 = i + c + 1;
            triangleLava.point3 = i + 1;

            double x1 = pointsLava[i].x;
            double y1 = pointsLava[i].y;
            double z1 = pointsLava[i].z;

            double x2 = pointsLava[i + c + 1].x;
            double y2 = pointsLava[i + c + 1].y;
            double z2 = pointsLava[i + c + 1].z;

            double x3 = pointsLava[i + 1].x;
            double y3 = pointsLava[i + 1].y;
            double z3 = pointsLava[i + 1].z;

            triangleLava.normX = (y2-y1)*(z3-z1) - (z2-z1)*(y3-y1);
            triangleLava.normY = (z2-z1)*(x3-x1) - (x2-x1)*(z3-z1);
            triangleLava.normZ = (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1);

            trianglesLava.push_back(triangleLava);
          }
        }
      }
    }
  }
}

void transformVertices(std::vector<Triangle> triangles, Point* points, int sizeTriangles, std::vector<Triangle> trianglesLava, Point* pointsLava, int sizeTrianglesLava, float* vertices){
  for(int i = 0; i < sizeTriangles; i++){
    Triangle triangle = triangles[i];

    vertices[i * 21] = points[triangle.point1].x;
    vertices[i * 21 + 1] = points[triangle.point1].y;
    vertices[i * 21 + 2] = points[triangle.point1].z;
    vertices[i * 21 + 3] = triangle.normX;
    vertices[i * 21 + 4] = triangle.normY;
    vertices[i * 21 + 5] = triangle.normZ;
    vertices[i * 21 + 6] = -3;

    vertices[i * 21 + 7] = points[triangle.point2].x;
    vertices[i * 21 + 8] = points[triangle.point2].y;
    vertices[i * 21 + 9] = points[triangle.point2].z;
    vertices[i * 21 + 10] = triangle.normX;
    vertices[i * 21 + 11] = triangle.normY;
    vertices[i * 21 + 12] = triangle.normZ;
    vertices[i * 21 + 13] = -3;

    vertices[i * 21 + 14] = points[triangle.point3].x;
    vertices[i * 21 + 15] = points[triangle.point3].y;
    vertices[i * 21 + 16] = points[triangle.point3].z;
    vertices[i * 21 + 17] = triangle.normX;
    vertices[i * 21 + 18] = triangle.normY;
    vertices[i * 21 + 19] = triangle.normZ;
    vertices[i * 21 + 20] = -3;
  }

  for(int i = 0; i < sizeTrianglesLava; i++){
    Triangle triangle = trianglesLava[i];

    vertices[(i + sizeTriangles) * 21] = pointsLava[triangle.point1].x;
    vertices[(i + sizeTriangles) * 21 + 1] = pointsLava[triangle.point1].y;
    vertices[(i + sizeTriangles) * 21 + 2] = pointsLava[triangle.point1].z;
    vertices[(i + sizeTriangles) * 21 + 3] = triangle.normX;
    vertices[(i + sizeTriangles) * 21 + 4] = triangle.normY;
    vertices[(i + sizeTriangles) * 21 + 5] = triangle.normZ;
    vertices[(i + sizeTriangles) * 21 + 6] = pointsLava[triangle.point1].color;

    vertices[(i + sizeTriangles) * 21 + 7] = pointsLava[triangle.point2].x;
    vertices[(i + sizeTriangles) * 21 + 8] = pointsLava[triangle.point2].y;
    vertices[(i + sizeTriangles) * 21 + 9] = pointsLava[triangle.point2].z;
    vertices[(i + sizeTriangles) * 21 + 10] = triangle.normX;
    vertices[(i + sizeTriangles) * 21 + 11] = triangle.normY;
    vertices[(i + sizeTriangles) * 21 + 12] = triangle.normZ;
    vertices[(i + sizeTriangles) * 21 + 13] = pointsLava[triangle.point2].color;

    vertices[(i + sizeTriangles) * 21 + 14] = pointsLava[triangle.point3].x;
    vertices[(i + sizeTriangles) * 21 + 15] = pointsLava[triangle.point3].y;
    vertices[(i + sizeTriangles) * 21 + 16] = pointsLava[triangle.point3].z;
    vertices[(i + sizeTriangles) * 21 + 17] = triangle.normX;
    vertices[(i + sizeTriangles) * 21 + 18] = triangle.normY;
    vertices[(i + sizeTriangles) * 21 + 19] = triangle.normZ;
    vertices[(i + sizeTriangles) * 21 + 20] = pointsLava[triangle.point3].color;
  }
}

//-----------------------------------------------------------------------------

int main(int argc, char **argv)
{
  int r = 0;
  int c = 0;
  double xllcorner = 0.0;
  double yllcorner = 0.0;
  double cellsize = 0.0;
  double nodata = 0.0;
  std::vector<double> Sz;
  std::vector<double> Sh;
  std::vector<double> So;
  double dumping_factor = 0.75;
  int steps = 0;
  int applyTexture;

  if (argc != 5)
  {
    printf("\n");
    printf("The application have to be executed as follows:\n");
    printf("./flow Sz_path Sh_path number_of_steps texture/noTexture\n");
    printf("\n");
    return(0);
  }

  if(atoi(argv[4]) == 1){
    applyTexture = 1;
  }else if(atoi(argv[4]) == 0){
    applyTexture = 0;
  }else{
    printf("Texture value must be 1 or 0\n");
    return(0);
  }

  printf("Loading initial configuration...\n");
  printf("Loading Sz...\n");
  loadM(argv[1], r, c, xllcorner, yllcorner, cellsize, nodata, &Sz);
  printf("Loading Sh...\n");
  loadM(argv[2], r, c, xllcorner, yllcorner, cellsize, nodata, &Sh);
  So.resize(r * c * 5, 0);
  printf("Done!\n");
  
  printf("Running simultaion...\n");
  steps = atoi(argv[3]);
  globalTransitionFunction(&Sz, &Sh, &So, dumping_factor, r, c, nodata, steps);
  printf("Done!\n");

  //OPENGL PART -----------------
  Point* points = (Point*)malloc(sizeof(Point)*r*c);
  Point* pointsLava = (Point*)malloc(sizeof(Point)*r*c);

  double minimumColorLava = 0;
  double maxmimumColorLava = 0;

  // HERE WE FIXE THE MAXIMUM LAVA COLOR BECAUSE IT WOULD DECREASE
  for(int i = 1; i < r*c; i++){
    if(Sh.at(i) != nodata){
      if(Sh.at(i) < minimumColorLava)
        minimumColorLava = Sh.at(i);
      if(Sh.at(i) > maxmimumColorLava)
        maxmimumColorLava = Sh.at(i);
    }
  }

  createPoints(&Sz, &Sh, r, c, cellsize, nodata, points, pointsLava, minimumColorLava, maxmimumColorLava);

  std::vector<Triangle> triangles;
  std::vector<Triangle> trianglesLava;
  createTriangles(points, pointsLava, triangles, trianglesLava, cellsize, r, c);
  
  float* vertices = (float*)malloc(triangles.size() * 3 * 7 * sizeof(float));

  transformVertices(triangles, points, triangles.size(), trianglesLava, pointsLava, trianglesLava.size(), vertices);

  draw(vertices, (trianglesLava.size() + triangles.size()) * 3 * 7 * sizeof(float), applyTexture);

  //-----------------------

  printf("Writing output...\n");
  saveM(argv[2], r, c, xllcorner, yllcorner, cellsize, nodata, &Sh);
  printf("Done!\n");

  return 0;
}
