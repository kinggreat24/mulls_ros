/*
 * @Author: kinggreat24
 * @Date: 2022-06-05 19:42:15
 * @LastEditTime: 2023-03-06 18:36:43
 * @LastEditors: kinggreat24 kinggreat24@whu.edu.cn
 * @Description:
 * @FilePath: /mulls_ros/src/bev_dbow_loop_node.cpp
 * 可以输入预定的版权声明、个性签名、空行等
 */
#include <dirent.h>
#include <iostream>

#include <ros/ros.h>

#include "ORBVocabulary.h"
#include "ORBextractor.h"

#include "utility.hpp"
#include "netvlad_tf_mulls/CompactImg.h"

#include <Eigen/Core>

void std2EigenVector(const std::vector<float> &in, Eigen::VectorXf &out);
void getSubmapBevImage(const std::string &dir_name, std::vector<std::string>& file_names);

int main(int argc, char **argv)
{
    ros::init(argc, argv, "submap_dbow_node");

    ros::NodeHandle nh, nh_private("~");

    int n_submap = 0;
    std::string submap_dir("");
    nh_private.param("submap_dir", submap_dir, submap_dir);
    std::vector<std::string> submap_files;
    getSubmapBevImage(submap_dir,submap_files);
    
    n_submap = (int)submap_files.size();
    std::sort(submap_files.begin(), submap_files.end());
    ROS_INFO("submap size: %d",n_submap);

    float minAcceptScore = 0.03;
    nh_private.param("minAcceptScore", minAcceptScore, minAcceptScore);

    std::string voc_file;
    nh_private.param("voc_file", voc_file, voc_file);

    int nFeatures = 2000;
    float scale_factor = 1.2;
    int nLevels = 8;
    int ini_ThFast = 20;
    int min_ThFast = 7;
    ORB_SLAM2::ORBextractor *mpORBExtractor = new ORB_SLAM2::ORBextractor(nFeatures, scale_factor, nLevels, ini_ThFast, min_ThFast);

    ORB_SLAM2::ORBVocabulary *mpVocabulary = new ORB_SLAM2::ORBVocabulary();
    mpVocabulary->load_frombin(voc_file);

    std::vector<DBoW3::BowVector> mBowVec;
    mBowVec.resize(n_submap);

    cv::Mat dbow_similarity_matrix_raw(n_submap, n_submap, CV_32FC1, cv::Scalar(0));
    cv::Mat dbow_similarity_matrix = cv::Mat::zeros(n_submap, n_submap, CV_8UC1);

    // Services
    ros::ServiceClient netvlad_client;
    netvlad_client = nh.serviceClient<netvlad_tf_mulls::CompactImg>("/compact_image");

    std::vector<Eigen::VectorXf> submap_netvlad_descs;
    submap_netvlad_descs.resize(n_submap);
    cv::Mat netvlad_similarity_matrix_raw(n_submap, n_submap, CV_32FC1, cv::Scalar(0));
    cv::Mat netvlad_similarity_matrix = cv::Mat::zeros(n_submap, n_submap, CV_8UC1);


    int ni = 0;
    ros::Rate rate(10);

    char file_name[128] = {0};
    cv::Mat subMamImg;
    while (ros::ok())
    {
        if (ni < n_submap)
        {
            ROS_INFO("Load image: %s", submap_files.at(ni).c_str());
            subMamImg = cv::imread(submap_files.at(ni), CV_LOAD_IMAGE_UNCHANGED);
            if(subMamImg.channels() == 1)
            {
                cv::cvtColor(subMamImg,subMamImg,CV_GRAY2BGR);
                cv::imwrite("/home/kinggreat24/pc/test.png",subMamImg);
            }    
            ROS_INFO("image channels: %d, type: %d",subMamImg.channels(),subMamImg.type());
            
            
            cv::imshow("im", subMamImg);
            cv::waitKey(1);

            cv::Mat descriptors;
            std::vector<cv::KeyPoint> vKeys;
            (*mpORBExtractor)(subMamImg, cv::Mat(), vKeys, descriptors);

            // Bag of Words Vector structures.
            DBoW3::BowVector bowVec;
            DBoW3::FeatureVector featVec;
            std::vector<cv::Mat> vCurrentDesc = lo::toDescriptorVector(descriptors);
            mpVocabulary->transform(vCurrentDesc, bowVec, featVec, 4);

            mBowVec[ni] = bowVec;

            // 计算netvlad描述符
            netvlad_tf_mulls::CompactImg img_srv;
            img_srv.request.req_img_name = /*submap_files.at(ni)*/"/home/kinggreat24/pc/test.png";
            if (netvlad_client.call(img_srv))
            {
                ROS_ERROR("Succeed to call service");
                ROS_INFO("Descriptor len %d",((std::vector<float>)img_srv.response.res_des).size());
                Eigen::VectorXf desc;
                std2EigenVector(img_srv.response.res_des, desc);
                submap_netvlad_descs[ni] = desc;
            }
            else
            {
                ROS_ERROR("Failed to call service");
            }

            ni++;
            rate.sleep();
            ros::spinOnce();
        }
        else
        {
            static bool flag = false;
            if (!flag)
            {
                flag = true;

                ROS_INFO("calculate dbow_similarity_matrix");
                for (int i = 0; i < n_submap; i++)
                {
                    // 直接通过词袋计算相似性
                    for (int j = 0; j < n_submap; j++)
                    {
                        double score = mpVocabulary->score(mBowVec[i], mBowVec[j]);
                        dbow_similarity_matrix_raw.at<float>(i, j) = score;

                        if (score > minAcceptScore)
                            dbow_similarity_matrix.at<uchar>(i, j) = 255;

                        // descriptor distance
		                float desc_score = (submap_netvlad_descs[i] - submap_netvlad_descs[j]).norm();
                        netvlad_similarity_matrix_raw.at<float>(i, j) = desc_score;
                    }
                }

                cv::imwrite("/home/kinggreat24/dbow_similarity_matrix.png", dbow_similarity_matrix);
                cv::imwrite("/home/kinggreat24/netvlad_similarity_matrix.png", netvlad_similarity_matrix_raw);
                cv::imshow("sim_img_raw", dbow_similarity_matrix_raw);
                cv::waitKey(0);

                cv::Mat dbow_similarity_uchar, dbow_similarity_uchar_cm;
                dbow_similarity_matrix_raw.convertTo(dbow_similarity_uchar, CV_8UC1, 255);
                cv::imshow("sim_img_gray", dbow_similarity_uchar);
                cv::waitKey(0);

                cv::applyColorMap(dbow_similarity_uchar, dbow_similarity_uchar_cm, cv::COLORMAP_JET);
                cv::imshow("sim_img", dbow_similarity_uchar_cm);
                cv::waitKey(0);

                //计算netvlad相似性
                cv::imshow("netvlad_sim_img_raw", netvlad_similarity_matrix_raw);
                cv::waitKey(0);

                cv::Mat netvlad_similarity_uchar, netvlad_similarity_uchar_cm;
                netvlad_similarity_matrix_raw.convertTo(netvlad_similarity_uchar, CV_8UC1, 255);
                cv::applyColorMap(netvlad_similarity_uchar, netvlad_similarity_uchar_cm, cv::COLORMAP_JET);

                cv::imshow("netvlad-sim_img", netvlad_similarity_uchar_cm);
                cv::waitKey(0);
            }

            rate.sleep();
            ros::spinOnce();
        }
    }

    return 0;
}

void std2EigenVector(const std::vector<float> &in, Eigen::VectorXf &out)
{
    out.resize(in.size());
    for (uint i = 0; i < in.size(); i++)
    {
        out(i) = in[i];
    }
}

void getSubmapBevImage(const std::string &dir_name, std::vector<std::string>& file_names)
{
    DIR *dp; // 创建一个指向root路径下每个文件的指针
    struct dirent *dirp;
    if ((dp = opendir(dir_name.c_str())) == NULL)
        cout << "can't open" << dir_name << endl;
    
    while ((dirp = readdir(dp)) != NULL)
    {
        cout<<"dirp->d_name: "<<dirp->d_name<<std::endl;
        if(dirp->d_name[0] == '.' || dirp->d_name[0] == '..')
        {
            std::cout<<"err: "<<dirp->d_name <<std::endl;
            continue;
        }    
        std::string full_name = dir_name + "/" + dirp->d_name;
        // cout <<full_name << endl; // 输出名字
        file_names.push_back(full_name);
    }
}