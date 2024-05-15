/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "ORBmatcher.h"

#include <limits.h>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "ThirdParty/DBoW3/src/FeatureVector.h"

// #include <stdint-gcc.h>

using namespace std;

namespace ORB_SLAM2
{

    const int ORBmatcher::TH_HIGH = 100;
    const int ORBmatcher::TH_LOW = 50;
    const int ORBmatcher::HISTO_LENGTH = 30;

    ORBmatcher::ORBmatcher(float nnratio, bool checkOri) : mfNNratio(nnratio), mbCheckOrientation(checkOri)
    {
    }

    float ORBmatcher::RadiusByViewingCos(const float &viewCos)
    {
        if (viewCos > 0.998)
            return 2.5;
        else
            return 4.0;
    }

    bool ORBmatcher::CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &F12, const std::vector<float> &vLevelSigma2)
    {
        // Epipolar line in second image l = x1'F12 = [a b c]
        const float a = kp1.pt.x * F12.at<float>(0, 0) + kp1.pt.y * F12.at<float>(1, 0) + F12.at<float>(2, 0);
        const float b = kp1.pt.x * F12.at<float>(0, 1) + kp1.pt.y * F12.at<float>(1, 1) + F12.at<float>(2, 1);
        const float c = kp1.pt.x * F12.at<float>(0, 2) + kp1.pt.y * F12.at<float>(1, 2) + F12.at<float>(2, 2);

        const float num = a * kp2.pt.x + b * kp2.pt.y + c;

        const float den = a * a + b * b;

        if (den == 0)
            return false;

        const float dsqr = num * num / den;

        return dsqr < 3.84 * vLevelSigma2[kp2.octave];
    }

    float ORBmatcher::GetDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &F12)
    {
        float dsqr = -1.0;

        // Epipolar line in second image l = x1'F12 = [a b c]
        const float a = kp1.pt.x * F12.at<float>(0, 0) + kp1.pt.y * F12.at<float>(1, 0) + F12.at<float>(2, 0);
        const float b = kp1.pt.x * F12.at<float>(0, 1) + kp1.pt.y * F12.at<float>(1, 1) + F12.at<float>(2, 1);
        const float c = kp1.pt.x * F12.at<float>(0, 2) + kp1.pt.y * F12.at<float>(1, 2) + F12.at<float>(2, 2);

        const float num = a * kp2.pt.x + b * kp2.pt.y + c;

        const float den = a * a + b * b;

        if (den == 0)
            dsqr = -1.0;
        else
            dsqr = std::sqrt(num * num / den);

        return dsqr;
    }

    // int ORBmatcher::SearchForInitialization(lo::cloudblock_t &F1, lo::cloudblock_t &F2, vector<cv::Point2f> &vbPrevMatched, vector<int> &vnMatches12, int windowSize)
    // {
    //     int nmatches=0;
    //     vnMatches12 = vector<int>(F1.mvKeysUn.size(),-1);

    //     vector<int> rotHist[HISTO_LENGTH];
    //     for(int i=0;i<HISTO_LENGTH;i++)
    //         rotHist[i].reserve(500);
    //     const float factor = 1.0f/HISTO_LENGTH;

    //     vector<int> vMatchedDistance(F2.mvKeysUn.size(),INT_MAX);
    //     vector<int> vnMatches21(F2.mvKeysUn.size(),-1);

    //     for(size_t i1=0, iend1=F1.mvKeysUn.size(); i1<iend1; i1++)
    //     {
    //         cv::KeyPoint kp1 = F1.mvKeysUn[i1];
    //         int level1 = kp1.octave;
    //         if(level1>0)
    //             continue;

    //         vector<size_t> vIndices2 = F2.GetFeaturesInArea(vbPrevMatched[i1].x,vbPrevMatched[i1].y, windowSize,level1,level1);

    //         if(vIndices2.empty())
    //             continue;

    //         cv::Mat d1 = F1.mDescriptors.row(i1);

    //         int bestDist = INT_MAX;
    //         int bestDist2 = INT_MAX;
    //         int bestIdx2 = -1;

    //         for(vector<size_t>::iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
    //         {
    //             size_t i2 = *vit;

    //             cv::Mat d2 = F2.mDescriptors.row(i2);

    //             int dist = DescriptorDistance(d1,d2);

    //             if(vMatchedDistance[i2]<=dist)
    //                 continue;

    //             if(dist<bestDist)
    //             {
    //                 bestDist2=bestDist;
    //                 bestDist=dist;
    //                 bestIdx2=i2;
    //             }
    //             else if(dist<bestDist2)
    //             {
    //                 bestDist2=dist;
    //             }
    //         }

    //         if(bestDist<=TH_LOW)
    //         {
    //             if(bestDist<(float)bestDist2*mfNNratio)
    //             {
    //                 if(vnMatches21[bestIdx2]>=0)
    //                 {
    //                     vnMatches12[vnMatches21[bestIdx2]]=-1;
    //                     nmatches--;
    //                 }
    //                 vnMatches12[i1]=bestIdx2;
    //                 vnMatches21[bestIdx2]=i1;
    //                 vMatchedDistance[bestIdx2]=bestDist;
    //                 nmatches++;

    //                 if(mbCheckOrientation)
    //                 {
    //                     float rot = F1.mvKeysUn[i1].angle-F2.mvKeysUn[bestIdx2].angle;
    //                     if(rot<0.0)
    //                         rot+=360.0f;
    //                     int bin = round(rot*factor);
    //                     if(bin==HISTO_LENGTH)
    //                         bin=0;
    //                     assert(bin>=0 && bin<HISTO_LENGTH);
    //                     rotHist[bin].push_back(i1);
    //                 }
    //             }
    //         }

    //     }

    //     if(mbCheckOrientation)
    //     {
    //         int ind1=-1;
    //         int ind2=-1;
    //         int ind3=-1;

    //         ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

    //         for(int i=0; i<HISTO_LENGTH; i++)
    //         {
    //             if(i==ind1 || i==ind2 || i==ind3)
    //                 continue;
    //             for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
    //             {
    //                 int idx1 = rotHist[i][j];
    //                 if(vnMatches12[idx1]>=0)
    //                 {
    //                     vnMatches12[idx1]=-1;
    //                     nmatches--;
    //                 }
    //             }
    //         }

    //     }

    //     //Update prev matched
    //     for(size_t i1=0, iend1=vnMatches12.size(); i1<iend1; i1++)
    //         if(vnMatches12[i1]>=0)
    //             vbPrevMatched[i1]=F2.mvKeysUn[vnMatches12[i1]].pt;

    //     return nmatches;
    // }

    int ORBmatcher::SearchByBoW(lo::cloudblock_t &F1, const lo::cloudblock_t &F2, std::vector<cv::DMatch> &vpMatches12)
    {
        vpMatches12 = vector<cv::DMatch>(F1.NP, cv::DMatch(-1, -1, -1));

        const DBoW3::FeatureVector &vFeatVec1 = F1.mFeatVec;
        const DBoW3::FeatureVector &vFeatVec2 = F2.mFeatVec;

        vector<int> rotHist[HISTO_LENGTH];
        for (int i = 0; i < HISTO_LENGTH; i++)
            rotHist[i].reserve(500);

        const float factor = 1.0f / HISTO_LENGTH;

        int nmatches = 0;

        DBoW3::FeatureVector::const_iterator f1it = vFeatVec1.begin();
        DBoW3::FeatureVector::const_iterator f2it = vFeatVec2.begin();
        DBoW3::FeatureVector::const_iterator f1end = vFeatVec1.end();
        DBoW3::FeatureVector::const_iterator f2end = vFeatVec2.end();

        while (f1it != f1end && f2it != f2end)
        {
            if (f1it->first == f2it->first)
            {
                for (size_t i1 = 0, iend1 = f1it->second.size(); i1 < iend1; i1++)
                {
                    const size_t idx1 = f1it->second[i1];

                    const cv::Mat &d1 = F1.mDescriptors.row(idx1);

                    int bestDist1 = 256;
                    int bestIdx2 = -1;
                    int bestDist2 = 256;

                    for (size_t i2 = 0, iend2 = f2it->second.size(); i2 < iend2; i2++)
                    {
                        const size_t idx2 = f2it->second[i2];

                        const cv::Mat &d2 = F2.mDescriptors.row(idx2);

                        int dist = DescriptorDistance(d1, d2);

                        if (dist < bestDist1)
                        {
                            bestDist2 = bestDist1;
                            bestDist1 = dist;
                            bestIdx2 = idx2;
                        }
                        else if (dist < bestDist2)
                        {
                            bestDist2 = dist;
                        }
                    }

                    if (bestDist1 < TH_LOW)
                    {
                        if (static_cast<float>(bestDist1) < mfNNratio * static_cast<float>(bestDist2))
                        {
                            // vpMatches12[idx1]=vpMapPoints2[bestIdx2];
                            // vbMatched2[bestIdx2]=true;

                            cv::DMatch match;
                            match.queryIdx = idx1;
                            match.trainIdx = bestIdx2;
                            match.distance = bestDist1;
                            vpMatches12[idx1] = match;

                            if (mbCheckOrientation)
                            {
                                float rot = F1.mvKeysUn[idx1].angle - F2.mvKeysUn[bestIdx2].angle;
                                if (rot < 0.0)
                                    rot += 360.0f;
                                int bin = round(rot * factor);
                                if (bin == HISTO_LENGTH)
                                    bin = 0;
                                assert(bin >= 0 && bin < HISTO_LENGTH);
                                rotHist[bin].push_back(idx1);
                            }
                            nmatches++;
                        }
                    }
                }

                f1it++;
                f2it++;
            }
            else if (f1it->first < f2it->first)
            {
                f1it = vFeatVec1.lower_bound(f2it->first);
            }
            else
            {
                f2it = vFeatVec2.lower_bound(f1it->first);
            }
        }

        if (mbCheckOrientation)
        {
            int ind1 = -1;
            int ind2 = -1;
            int ind3 = -1;

            ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

            for (int i = 0; i < HISTO_LENGTH; i++)
            {
                if (i == ind1 || i == ind2 || i == ind3)
                    continue;
                for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
                {
                    // vpMatches12[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                    vpMatches12[rotHist[i][j]] = cv::DMatch(-1, -1, -1);
                    nmatches--;
                }
            }
        }

        return nmatches;
    }

    int ORBmatcher::SearchBruteForce(lo::cloudblock_t *frame1, lo::cloudblock_t *frame2,
                                     std::vector<cv::DMatch> &matches)
    {
        matches.reserve(frame1->mDescriptors.rows);

        for (size_t i = 0; i < frame1->mDescriptors.rows; i++)
        {
            const cv::Mat &d1 = frame1->mDescriptors.row(i);
            int min_dist = 9999;
            int min_dist_index = -1;

            for (size_t j = 0; j < frame2->mDescriptors.rows; j++)
            {
                const cv::Mat &d2 = frame2->mDescriptors.row(j);
                int dist = DescriptorDistance(d1, d2);
                if (dist < min_dist)
                {
                    min_dist = dist;
                    min_dist_index = j;
                }
            }

            if (min_dist < TH_LOW)
            {
                matches.push_back(cv::DMatch(i, min_dist_index, min_dist));
            }
        }

        return matches.size();
    }

    void ORBmatcher::ComputeThreeMaxima(vector<int> *histo, const int L, int &ind1, int &ind2, int &ind3)
    {
        int max1 = 0;
        int max2 = 0;
        int max3 = 0;

        for (int i = 0; i < L; i++)
        {
            const int s = histo[i].size();
            if (s > max1)
            {
                max3 = max2;
                max2 = max1;
                max1 = s;
                ind3 = ind2;
                ind2 = ind1;
                ind1 = i;
            }
            else if (s > max2)
            {
                max3 = max2;
                max2 = s;
                ind3 = ind2;
                ind2 = i;
            }
            else if (s > max3)
            {
                max3 = s;
                ind3 = i;
            }
        }

        if (max2 < 0.1f * (float)max1)
        {
            ind2 = -1;
            ind3 = -1;
        }
        else if (max3 < 0.1f * (float)max1)
        {
            ind3 = -1;
        }
    }

    // Bit set count operation from
    // http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
    int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
    {
        const int *pa = a.ptr<int32_t>();
        const int *pb = b.ptr<int32_t>();

        int dist = 0;

        for (int i = 0; i < 8; i++, pa++, pb++)
        {
            unsigned int v = *pa ^ *pb;
            v = v - ((v >> 1) & 0x55555555);
            v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
            dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
        }

        return dist;
    }

} //namespace ORB_SLAM
