// get_frame.cpp: define el punto de entrada de la aplicación de consola.
//
#include "precomp.hpp"
#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>
#include <opencv2/highgui/highgui.hpp> // OpenCV window I/O
#include <opencv2/imgproc/imgproc.hpp> // OpenCV image transformations
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "opencv2/reg/mapaffine.hpp"
#include "opencv2/reg/mapshift.hpp"
#include "opencv2/reg/mapprojec.hpp"
#include "opencv2/reg/mappergradshift.hpp"
#include "opencv2/reg/mappergradeuclid.hpp"
#include "opencv2/reg/mappergradsimilar.hpp"
#include "opencv2/reg/mappergradaffine.hpp"
#include "opencv2/reg/mappergradproj.hpp"
#include "opencv2/reg/mapperpyramid.hpp"

static const char* DIFF_IM = "Image difference";
static const char* DIFF_REGPIX_IM = "Image difference: pixel registered";

using namespace cv;
using namespace cv::reg;
using namespace std;


void showDifference(const Mat& image1, const Mat& image2, const char* title)
{
    Mat img1, img2;
    image1.convertTo(img1, CV_32FC3);
    image2.convertTo(img2, CV_32FC3);
    cvtColor(img1, img1, CV_RGB2GRAY);
    cvtColor(img2, img2, CV_RGB2GRAY);

    Mat imgDiff;
    img1.copyTo(imgDiff);
    imgDiff -= img2;
    imgDiff /= 2.f;
    imgDiff += 128.f;

    Mat imgSh;
    imgDiff.convertTo(imgSh, CV_8UC3);
    imshow(title, imgSh);
}

void testShift(const Mat& img1)
{
    Mat img2;

    // Warp original image
    Vec<double, 2> shift(5., 5.);
    MapShift mapTest(shift);
    mapTest.warp(img1, img2);
    showDifference(img1, img2, DIFF_IM);

    // Register
    MapperGradShift mapper;
    MapperPyramid mappPyr(mapper);
    Ptr<Map> mapPtr(0);
    mappPyr.calculate(img1, img2, mapPtr);

    // Print result
    MapShift* mapShift = dynamic_cast<MapShift*>(mapPtr.obj);
    cout << endl << "--- Testing shift mapper ---" << endl;
    cout << Mat(shift) << endl;
    cout << Mat(mapShift->getShift()) << endl;

    // Display registration accuracy
    Mat dest;
    mapShift->inverseWarp(img2, dest);
    showDifference(img1, dest, DIFF_REGPIX_IM);

    waitKey(0);
    cvDestroyWindow(DIFF_IM);
    cvDestroyWindow(DIFF_REGPIX_IM);
}

void testEuclidean(const Mat& img1)
{
    Mat img2;

    // Warp original image
    double theta = 3*M_PI/180;
    double cosT = cos(theta);
    double sinT = sin(theta);
    Matx<double, 2, 2> linTr(cosT, -sinT, sinT, cosT);
    Vec<double, 2> shift(5., 5.);
    MapAffine mapTest(linTr, shift);
    mapTest.warp(img1, img2);
    showDifference(img1, img2, DIFF_IM);

    // Register
    MapperGradEuclid mapper;
    MapperPyramid mappPyr(mapper);
    Ptr<Map> mapPtr(0);
    mappPyr.calculate(img1, img2, mapPtr);

    // Print result
    MapAffine* mapAff = dynamic_cast<MapAffine*>(mapPtr.obj);
    cout << endl << "--- Testing Euclidean mapper ---" << endl;
    cout << Mat(linTr) << endl;
    cout << Mat(shift) << endl;
    cout << Mat(mapAff->getLinTr()) << endl;
    cout << Mat(mapAff->getShift()) << endl;

    // Display registration accuracy
    Mat dest;
    mapAff->inverseWarp(img2, dest);
    showDifference(img1, dest, DIFF_REGPIX_IM);

    waitKey(0);
    cvDestroyWindow(DIFF_IM);
    cvDestroyWindow(DIFF_REGPIX_IM);
}

void testSimilarity(const Mat& img1)
{
    Mat img2;

    // Warp original image
    double theta = 3*M_PI/180;
    double scale = 0.95;
    double a = scale*cos(theta);
    double b = scale*sin(theta);
    Matx<double, 2, 2> linTr(a, -b, b, a);
    Vec<double, 2> shift(5., 5.);
    MapAffine mapTest(linTr, shift);
    mapTest.warp(img1, img2);
    showDifference(img1, img2, DIFF_IM);

    // Register
    MapperGradSimilar mapper;
    MapperPyramid mappPyr(mapper);
    Ptr<Map> mapPtr(0);
    mappPyr.calculate(img1, img2, mapPtr);

    // Print result
    MapAffine* mapAff = dynamic_cast<MapAffine*>(mapPtr.obj);
    cout << endl << "--- Testing similarity mapper ---" << endl;
    cout << Mat(linTr) << endl;
    cout << Mat(shift) << endl;
    cout << Mat(mapAff->getLinTr()) << endl;
    cout << Mat(mapAff->getShift()) << endl;

    // Display registration accuracy
    Mat dest;
    mapAff->inverseWarp(img2, dest);
    showDifference(img1, dest, DIFF_REGPIX_IM);

    waitKey(0);
    cvDestroyWindow(DIFF_IM);
    cvDestroyWindow(DIFF_REGPIX_IM);
}

void testAffine(const Mat& img1)
{
    Mat img2;

    // Warp original image
    Matx<double, 2, 2> linTr(1., 0.1, -0.01, 1.);
    Vec<double, 2> shift(1., 1.);
    MapAffine mapTest(linTr, shift);
    mapTest.warp(img1, img2);
    showDifference(img1, img2, DIFF_IM);

    // Register
    MapperGradAffine mapper;
    MapperPyramid mappPyr(mapper);
    Ptr<Map> mapPtr(0);
    mappPyr.calculate(img1, img2, mapPtr);

    // Print result
    MapAffine* mapAff = dynamic_cast<MapAffine*>(mapPtr.obj);
    cout << endl << "--- Testing affine mapper ---" << endl;
    cout << Mat(linTr) << endl;
    cout << Mat(shift) << endl;
    cout << Mat(mapAff->getLinTr()) << endl;
    cout << Mat(mapAff->getShift()) << endl;

    // Display registration accuracy
    Mat dest;
    mapAff->inverseWarp(img2, dest);
    showDifference(img1, dest, DIFF_REGPIX_IM);

    waitKey(0);
    cvDestroyWindow(DIFF_IM);
    cvDestroyWindow(DIFF_REGPIX_IM);
}

void testProjective(const Mat& img1)
{
    Mat img2;

    // Warp original image
    Matx<double, 3, 3> projTr(1., 0., 0., 0., 1., 0., 0.0001, 0.0001, 1);
    MapProjec mapTest(projTr);
    mapTest.warp(img1, img2);
    showDifference(img1, img2, DIFF_IM);

    // Register
    MapperGradProj mapper;
    MapperPyramid mappPyr(mapper);
    Ptr<Map> mapPtr(0);
    mappPyr.calculate(img1, img2, mapPtr);

    // Print result
    MapProjec* mapProj = dynamic_cast<MapProjec*>(mapPtr.obj);
    mapProj->normalize();
    cout << endl << "--- Testing projective transformation mapper ---" << endl;
    cout << Mat(projTr) << endl;
    cout << Mat(mapProj->getProjTr()) << endl;

    // Display registration accuracy
    Mat dest;
    mapProj->inverseWarp(img2, dest);
    showDifference(img1, dest, DIFF_REGPIX_IM);

    waitKey(0);
    cvDestroyWindow(DIFF_IM);
    cvDestroyWindow(DIFF_REGPIX_IM);
}

//
// Following an example from
// http:// ramsrigoutham.com/2012/11/22/panorama-image-stitching-in-opencv/
//
void calcHomographyFeature(const Mat& image1, const Mat& image2)
{
    static const char* difffeat = "Difference feature registered";

    Mat gray_image1;
    Mat gray_image2;
    // Convert to Grayscale
    cvtColor(image1, gray_image1, CV_RGB2GRAY);
    cvtColor(image2, gray_image2, CV_RGB2GRAY);

    //-- Step 1: Detect the keypoints using SURF Detector
    int minHessian = 400;

    SurfFeatureDetector detector(minHessian);

    std::vector<KeyPoint> keypoints_object, keypoints_scene;

    detector.detect(gray_image1, keypoints_object);
    detector.detect(gray_image2, keypoints_scene);

    //-- Step 2: Calculate descriptors (feature vectors)
    SurfDescriptorExtractor extractor;

    Mat descriptors_object, descriptors_scene;

    extractor.compute(gray_image1, keypoints_object, descriptors_object);
    extractor.compute(gray_image2, keypoints_scene, descriptors_scene);

    //-- Step 3: Matching descriptor vectors using FLANN matcher
    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( descriptors_object, descriptors_scene, matches );

    double max_dist = 0; double min_dist = 100;

    //-- Quick calculation of max and min distances between keypoints
    for(int i = 0; i < descriptors_object.rows; i++)
    {
        double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

    //-- Use only "good" matches (i.e. whose distance is less than 3*min_dist )
    std::vector<DMatch> good_matches;

    for(int i = 0; i < descriptors_object.rows; i++) {
        if(matches[i].distance < 3*min_dist) {
            good_matches.push_back( matches[i]);
        }
    }
    std::vector< Point2f > obj;
    std::vector< Point2f > scene;

    for(size_t i = 0; i < good_matches.size(); i++)
    {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
    }

    // Find the Homography Matrix
    Mat H = findHomography( obj, scene, CV_RANSAC );
    // Use the Homography Matrix to warp the images
    Mat result;
    Mat Hinv = H.inv();
    warpPerspective(image2, result, Hinv, image1.size());

    cout << "--- Feature method\n" << H << endl;
    
    Mat imf1, resf;
    image1.convertTo(imf1, CV_64FC3);
    result.convertTo(resf, CV_64FC3);
    showDifference(imf1, resf, difffeat);
}

void calcHomographyPixel(const Mat& img1, const Mat& img2)
{
    static const char* diffpixel = "Difference pixel registered";

    // Register using pixel differences
    MapperGradProj mapper;
    MapperPyramid mappPyr(mapper);
    Ptr<Map> mapPtr(0);
    mappPyr.calculate(img1, img2, mapPtr);

    // Print result
    MapProjec* mapProj = dynamic_cast<MapProjec*>(mapPtr.obj);
    mapProj->normalize();
    cout << "--- Pixel-based method\n" << Mat(mapProj->getProjTr()) << endl;

    // Display registration accuracy
    Mat dest;
    mapProj->inverseWarp(img2, dest);
    showDifference(img1, dest, diffpixel);
}

void comparePixelVsFeature(const Mat& img1_8b, const Mat& img2_8b)
{
    static const char* difforig = "Difference non-registered";

    // Show difference of images
    Mat img1, img2;
    img1_8b.convertTo(img1, CV_64FC3);
    img2_8b.convertTo(img2, CV_64FC3);
    showDifference(img1, img2, difforig);
    cout << endl << "--- Comparing feature-based with pixel difference based ---" << endl;

    // Register using SURF keypoints
    calcHomographyFeature(img1_8b, img2_8b);

    // Register using pixel differences
    calcHomographyPixel(img1, img2);

    waitKey(0);
}


int main(int argc, char* argv[])
{
    Mat img1;
    
    if(argc < 2) {
        cout <<  "Usage: map_test <picture>" << endl;
        return -1;
    }
    img1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    if(!img1.data) {
        cout <<  "Could not open or find " << argv[1] << endl;
        return -1;
    }

    // Convert to double, 3 channels
    img1.convertTo(img1, CV_64FC3);

    testShift(img1);
    testEuclidean(img1);
    testSimilarity(img1);
    testAffine(img1);
    testProjective(img1);

    Mat imgcmp1 = imread("LR_05.png", CV_LOAD_IMAGE_COLOR);
    Mat imgcmp2 = imread("LR_06.png", CV_LOAD_IMAGE_COLOR);
    comparePixelVsFeature(imgcmp1, imgcmp2);

    return 0;
}
