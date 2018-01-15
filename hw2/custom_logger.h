//
// Created by Salomon Lee on 12/1/16.
//

#ifndef PARALLEL_COMPUTING_CUSTOM_LOGGER_H
#define PARALLEL_COMPUTING_CUSTOM_LOGGER_H

#include <iostream>
#include <sstream>

template <typename T>
std::string stringify(const T& val) {
    std::ostringstream ss;
    ss << val;
    return ss.str();
}

template<> std::string stringify<dim3>(const dim3& val) {
    std::ostringstream ss;
    ss << "(" << val.x << ", " << val.y << ", " << val.z << ")";
    return ss.str();
}

template<> std::string stringify<uchar4>(const uchar4& val) {
    std::ostringstream ss;
    ss << "{" << static_cast<unsigned>(val.x) << ", "
       << static_cast<unsigned>(val.y) << ", "
       << static_cast<unsigned>(val.z) << ", "
       << static_cast<unsigned>(val.w) << "}";
    return ss.str();
}


#endif //PARALLEL_COMPUTING_CUSTOM_LOGGER_H
