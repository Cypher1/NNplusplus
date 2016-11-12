//
//  Matrix.cpp
//  Neural Net
//
//  Created by Gil Dekel on 8/19/16.
//  Last edited by Joshua Pratt on 10/01/16.
//

#include "Matrix.hpp"

/*
 * Private members for reference
 *
 * size_t n_rows;      // (M)xN
 * size_t n_cols;      // Mx(N)
 * double *matrix_;     // A pointer to the array.
 * double **rowPtrs_;   // An array of row pointers.
 *                      // used to avoid repeated arithmetics
 *                      // at each access to the matrix.
 *
 */

/**********************************************************
 * Constructors
 **********************************************************/

Matrix::Matrix(const size_t m, const size_t n) : n_rows{m}, n_cols{n} {
    matrix_ = new double[size()]();
    rowPtrs_ = new double *[nRows()];

    for (size_t i = 0; i < nRows(); ++i) {
        rowPtrs_[i] = matrix_ + i * nCols();
    }
}

Matrix::Matrix(Matrix &&rhs)
    : n_rows{rhs.nRows()},
      n_cols{rhs.nCols()},
      matrix_{rhs.matrix_},
      rowPtrs_{rhs.rowPtrs_} {
    rhs.matrix_ = nullptr;
    rhs.rowPtrs_ = nullptr;
}

Matrix &Matrix::operator=(Matrix &&rhs) {
    std::swap(n_rows, rhs.n_rows);
    std::swap(n_cols, rhs.n_cols);
    std::swap(matrix_, rhs.matrix_);
    std::swap(rowPtrs_, rhs.rowPtrs_);
    return *this;
}

Matrix::~Matrix() {
    delete[] matrix_;
    delete[] rowPtrs_;
}

