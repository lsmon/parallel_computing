//
// Created by Salomon Lee on 12/1/16.
//

#ifndef PARALLEL_COMPUTING_REFERENCE_CALC_H
#define PARALLEL_COMPUTING_REFERENCE_CALC_H

void referenceCalculation(const uchar4* const rgbaImage,
                          unsigned char *const greyImage,
                          size_t numRows,
                          size_t numCols);

#endif //PARALLEL_COMPUTING_REFERENCE_CALC_H
