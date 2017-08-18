#include <iostream>
#include <string>
#include <sstream>
#include <stdio.h>
using namespace std;

// OpenCV includes
#include "opencv2/opencv_modules.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/calib3d/calib3d.hpp"
 #include "opencv2/core/utility.hpp"
#include <opencv2/imgproc.hpp>

using namespace cv;

int main( int argc, const char** argv )
{
  /*   Lectura de imágenes   */
  Mat img_objeto= imread("logoipn_escala.jpg", CV_LOAD_IMAGE_GRAYSCALE);
  Mat img_escena= imread("escena_cartel_c_21.jpg", CV_LOAD_IMAGE_GRAYSCALE);
  Mat out, out2;
  Mat desc_objeto, desc_escena; /* Matrices para detectar o calcular descriptores   */
  Mat salida_objeto, salida_escena;
  float nndrRatio = 0.7f;
  
  
//////// 500, 1.2, 8, 31,0, 2, harris, 31 20
  /*   se inicializa detector ORB    */
  
  Ptr<ORB> orb_detector = ORB::create(3000,1.2,8,31,0,2,ORB::HARRIS_SCORE, 31,20);  //http://docs.opencv.org/trunk/db/d95/classcv_1_1ORB.html
  std::vector<KeyPoint> kp_objeto; 
  //orb_detector->detect(img_objeto, kp_objeto);
  orb_detector->detectAndCompute(img_objeto, Mat(), kp_objeto, desc_objeto );
  drawKeypoints(img_objeto, kp_objeto, salida_objeto, Scalar::all(255));
  //namedWindow("Objeto", WINDOW_NORMAL );
  //imshow("Objeto", salida_objeto);
  
  
  Ptr<ORB> orb_detector2 = ORB::create(3000,1.2,8,31,0,2,ORB::HARRIS_SCORE, 31,20);
  std::vector<KeyPoint> kp_escena;
  //orb_detector2->detect(img_escena, kp_escena);
  orb_detector2->detectAndCompute(img_escena, Mat(), kp_escena, desc_escena );
  drawKeypoints(img_escena, kp_escena, salida_escena, Scalar::all(255));
  //namedWindow("Escena", WINDOW_NORMAL );
  //imshow("Escena", salida_escena);
  

  /*  Se inicializa descriptor BFMatcher */
  
    ////////////////////////
    ////////////////////////
    
    BFMatcher matcher(NORM_HAMMING);
    std::vector<vector<DMatch > > matches;
    matcher.knnMatch( desc_objeto, desc_escena, matches,2);
    

    vector< DMatch > good_matches;
    good_matches.reserve(matches.size());
    
    for (size_t i = 0; i < matches.size(); ++i)
  { 
      if (matches[i].size() < 2)
                  continue;
     
      const DMatch &m1 = matches[i][0];
      const DMatch &m2 = matches[i][1];
     
      if(m1.distance <= nndrRatio * m2.distance)        
      good_matches.push_back(m1);     
  }
  cout << "\n Cantidad de buenos puntos " << good_matches.size();
  if( (good_matches.size() >=7))
  {
 
    cout << "\n¡Objeto encontrado!" << endl;
    Mat outImg;
    drawMatches(img_objeto,kp_objeto,img_escena,kp_escena, good_matches ,
		outImg,Scalar::all(-1), Scalar::all(-1),vector<char>(),DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
 
    std::vector< Point2f > obj;
    std::vector< Point2f > scene;
 
    for( unsigned int i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( kp_objeto[ good_matches[i].queryIdx ].pt );
        scene.push_back( kp_escena[ good_matches[i].trainIdx ].pt );
    }
 
    Mat H = findHomography( obj, scene, CV_RANSAC );
    std::vector< Point2f > obj_corners(4);
    obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_objeto.cols, 0 );
    obj_corners[2] = cvPoint( img_objeto.cols, img_objeto.rows ); obj_corners[3] = cvPoint( 0, img_objeto.rows );
    std::vector< Point2f > scene_corners(4);
 
    perspectiveTransform( obj_corners, scene_corners, H);
         
    
    bool objectFound = false;

    
    //-- Draw lines between the corners (the mapped object in the scene - image_2 ) 
    Point2f offset ( (float)img_objeto.cols, 0);
    line( outImg, scene_corners[0] + offset , scene_corners[1] + offset, Scalar(0, 255, 0), 10 ); //TOP line
    line( outImg, scene_corners[1] + offset, scene_corners[2]+ offset, Scalar(0, 255, 0), 10 );
    line( outImg, scene_corners[2] + offset, scene_corners[3]+ offset, Scalar(0, 255, 0), 10 );
    line( outImg, scene_corners[3] + offset, scene_corners[0]+ offset , Scalar(0, 255, 0), 10 ); 
    objectFound=true;
    namedWindow("Coincidencias", WINDOW_NORMAL );
     imshow( "Coincidencias", outImg);
    
  }
  else {
    cout << "\n¡Objeto no encontrado!" << endl;
  }
  
    ////////////////////////////
    ////////////////////////////
    
  /*    
        
    FlannBasedMatcher matcher(new flann::LshIndexParams(20, 10, 2));
    std::vector< DMatch > matches;
    matcher.match( desc_objeto, desc_escena, matches);
    double max_dist = 0; double min_dist = 100;
    
    
    for( int i = 0; i < desc_objeto.rows; i++ )
      { double dist = matches[i].distance;
	if( dist < min_dist ) min_dist = dist;
	if( dist > max_dist ) max_dist = dist;
      }
      
    cout << "\n Max dist : " << max_dist;
    cout << "\n Min dist : " << min_dist;
    
    std::vector< DMatch > good_matches;
    
    for (int i = 0; i < desc_objeto.rows; i++){
      if (matches[i].distance <= max(2*min_dist, 0.02)){
	good_matches.push_back( matches[i]);
      }
    }
    
    cout << "\n Coincidencias " << good_matches.size();
    
    Mat img_matches;
    drawMatches( img_objeto, kp_objeto, img_escena, kp_escena,
      good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
      vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    std::vector<Point2f> objeto;
    std::vector<Point2f> escena;
    
      for( int i = 0; i < (int)good_matches.size(); i++ )
      { //printf( "\n-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx );
	objeto.push_back( kp_objeto[ good_matches[i].queryIdx ].pt);
	escena.push_back( kp_escena[ good_matches[i].trainIdx ].pt);
      }
      
      Mat H = findHomography(objeto, escena, CV_RANSAC); 
      
      
      std::vector<Point2f> esquinas_objeto(4);
      esquinas_objeto[0] = cvPoint(0,0); esquinas_objeto[1] = cvPoint( img_objeto.cols, 0);
      esquinas_objeto[2] = cvPoint(img_objeto.cols, img_objeto.rows); esquinas_objeto[3] = cvPoint(0, img_objeto.rows);
      std::vector<Point2f> esquinas_escena(4);
      
      perspectiveTransform( esquinas_objeto, esquinas_escena, H);
      
      Point2f offset ( (float)img_objeto.cols, 0);
      line( img_matches, esquinas_escena[0] + offset, esquinas_escena[1] + offset, Scalar(0, 255, 0), 4 );
      line( img_matches, esquinas_escena[1] + offset, esquinas_escena[2] + offset, Scalar(0, 255, 0), 4 );
      line( img_matches, esquinas_escena[2] + offset, esquinas_escena[3] + offset, Scalar(0, 255, 0), 4 );
      line( img_matches, esquinas_escena[3] + offset, esquinas_escena[0] + offset, Scalar(0, 255, 0), 4 );
      //line(img_matches, 100, 100,, Scalar(0,255,0), 4);
      namedWindow("Good", WINDOW_NORMAL );
      imshow( "Good", img_matches);
      */

  waitKey(0);
  return 0;
}