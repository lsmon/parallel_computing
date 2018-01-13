//
// Created by Salomon Lee on 12/1/16.
//

#ifndef PARALLEL_COMPUTING_COMPARE_H
#define PARALLEL_COMPUTING_COMPARE_H

void compareImages(std::string reference_filename, std::string test_filename,
                   bool useEpsCheck, double perPixelError, double globalError);


#endif //PARALLEL_COMPUTING_COMPARE_H
