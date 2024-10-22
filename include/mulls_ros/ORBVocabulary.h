/*
 * @Author: kinggreat24
 * @Date: 2022-05-31 11:42:05
 * @LastEditTime: 2022-05-31 11:51:34
 * @LastEditors: kinggreat24
 * @Description: 
 * @FilePath: /mulls_ros/include/mulls_ros/ORBVocabulary.h
 * 可以输入预定的版权声明、个性签名、空行等
 */
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


#ifndef ORBVOCABULARY_H
#define ORBVOCABULARY_H

// #include"ThirdParty/DBoW3/FORB.h"
// #include"ThirdParty/DBoW3/TemplatedVocabulary.h"

 #include "ThirdParty/DBoW3/src/DBoW3.h"

namespace ORB_SLAM2
{

// typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>
//   ORBVocabulary;

typedef DBoW3::Vocabulary ORBVocabulary;

} //namespace ORB_SLAM

#endif // ORBVOCABULARY_H
