//
//  Matrix.hpp
//  Neural Net
//
//  A matrix object, which includes basic operations such as
//  matrix transpose and dot product.
//
//  Created by Gil Dekel on 8/19/16.
//  Last edited by Joshua Pratt on 10/01/16.
//

#ifndef MATRIX_HPP_
#define MATRIX_HPP_

#include <cmath>  // INFINITY
#include <functional>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <utility>  // std::swap, std::move and std::pair
#include "MatrixExceptions.hpp"

template <typename L>
class UnaryMatrix;
template <typename L, typename R>
class DotMatrix;

template <typename E>
class TransposeMatrix;

class UniformMatrix;
class Matrix;

template <typename E>
class MatrixExpr {
   public:
    double operator()(size_t r, size_t c) const {
        return static_cast<E const &>(*this)(r, c);
    }
    size_t nRows() const { return static_cast<E const &>(*this).nRows(); }
    size_t nCols() const { return static_cast<E const &>(*this).nCols(); }
    size_t size() const { return nRows() * nCols(); }

    operator Matrix() const;

    UnaryMatrix<MatrixExpr<E>> apply(double (*functor)(const double &)) {
        return UnaryMatrix<MatrixExpr<E>>(*this, functor);
    }

    TransposeMatrix<MatrixExpr<E>> T() const {
        return TransposeMatrix<MatrixExpr<E>>(*this);
    }

    template <typename R>
    DotMatrix<MatrixExpr<E>, R> dot(const R &rhs) const {
        return DotMatrix<MatrixExpr<E>, R>(*this, rhs);
    }

    template <typename R>
    bool operator==(const R &rhs) const {
        if (rhs.nRows() != nRows() || rhs.nCols() != nCols()) {
            return false;
        }
        for (size_t i = 0; i < nRows(); ++i) {
            for (size_t j = 0; j < nCols(); ++j) {
                double rv = rhs(i, j);
                double lv = (*this)(i, j);
                if (std::fabs((rv - lv) / lv) > 0.0000000001) {
                    return false;
                }
            }
        }
        return true;
    }

    template <typename R>
    bool operator!=(const R &rhs) const {
        return !((*this) == rhs);
    }

    friend std::ostream &operator<<(std::ostream &os, const MatrixExpr &rhs) {
        os << "[";
        for (auto i = 0U; i < rhs.nRows(); ++i) {
            os << "[";
            for (auto j = 0U; j < rhs.nCols(); ++j) {
                if (j != 0) {
                    os << ", ";
                }
                os << rhs(i, j);
            }
            os << "]";
        }
        os << "]";
        return os;
    }

    std::pair<size_t, size_t> getMaxVal() const {
        double curr = -INFINITY;
        std::pair<size_t, size_t> max = std::make_pair(-1, -1);
        for (size_t i = 0; i < nRows(); ++i) {
            for (size_t j = 0; j < nCols(); ++j) {
                double next = (*this)(i, j);
                if (next > curr) {
                    curr = next;
                    max = std::make_pair(i, j);
                }
            }
        }
        return max;
    }

    // The following overload conversions to E, the template argument type;
    // e.g., for VecExpression<VecSum>, this is a conversion to VecSum.
    // operator E &()() { return static_cast<E &>(*this); }
    // operator E const &()() const { return static_cast<const E &>(*this); }
};

template <typename L, typename R>
class CombineMatrix : public MatrixExpr<CombineMatrix<L, R>> {
    L const &l;
    R const &r;
    double (*functor)(const double &, const double &);

   public:
    CombineMatrix(L const &l, R const &r,
                  double (*functor)(const double &, const double &))
        : l(l), r(r), functor(functor) {
        if (l.nRows() != r.nRows() || l.nCols() != r.nCols()) {
            throw MatrixDimensionsMismatch(
                std::make_pair(l.nRows(), l.nCols()),
                std::make_pair(r.nRows(), r.nCols()));
        }
    }

    double operator()(const size_t &row, const size_t &col) const {
        return functor(l(row, col), r(row, col));
    }
    size_t nRows() const { return l.nRows(); }
    size_t nCols() const { return l.nCols(); }
};

template <typename L>
class UnaryMatrix : public MatrixExpr<UnaryMatrix<L>> {
    L const &l;
    double (*functor)(const double &);

   public:
    UnaryMatrix(L const &l, double (*functor)(const double &))
        : l{l}, functor{functor} {}

    double operator()(const size_t &row, const size_t &col) const {
        return functor(l(row, col));
    }
    size_t nRows() const { return l.nRows(); }
    size_t nCols() const { return l.nCols(); }
};

template <typename L>
class TransposeMatrix : public MatrixExpr<TransposeMatrix<L>> {
    L const &l;

   public:
    TransposeMatrix(L const &l) : l{l} {}

    double operator()(const size_t &row, const size_t &col) const {
        return l(col, row);
    }
    size_t nRows() const { return l.nCols(); }
    size_t nCols() const { return l.nRows(); }
};

template <typename L, typename R>
class DotMatrix : public MatrixExpr<DotMatrix<L, R>> {
    L const &l;
    R const &r;

   public:
    DotMatrix(L const &l, R const &r) : l(l), r(r) {
        if (l.nCols() != r.nRows()) {
            throw MatrixDimensionsMismatch(
                std::make_pair(l.nRows(), l.nCols()),
                std::make_pair(r.nRows(), r.nCols()));
        }
    }

    double operator()(const size_t &row, const size_t &col) const {
        double d = 0;
        for (size_t k = 0; k < l.nCols(); ++k) {
            d += l(row, k) * r(k, col);
        }
        return d;
    }
    size_t nRows() const { return l.nRows(); }
    size_t nCols() const { return r.nCols(); }
};

class UniformMatrix : public MatrixExpr<UniformMatrix> {
    const double value;
    const size_t n_rows;
    const size_t n_cols;

   public:
    UniformMatrix(const double &value, const size_t &r = 1, const size_t &c = 1)
        : value{value}, n_rows{r}, n_cols{c} {}

    UniformMatrix resize(const size_t &r, const size_t &c) {
        return UniformMatrix(value, r, c);
    }

    template <typename E>
    UniformMatrix resize(const MatrixExpr<E> &o) {
        return resize(o.nRows(), o.nCols());
    }

    double operator()(const size_t &, const size_t &) const { return value; }
    size_t nRows() const { return n_rows; }
    size_t nCols() const { return n_cols; }
};

template <typename L, typename R>
CombineMatrix<L, R> const operator+(L const &l, R const &r) {
    auto op = [](const double &l, const double &r) { return l + r; };
    return CombineMatrix<L, R>(l, r, op);
}

template <typename L, typename R>
CombineMatrix<L, R> const operator-(L const &l, R const &r) {
    auto op = [](const double &l, const double &r) { return l - r; };
    return CombineMatrix<L, R>(l, r, op);
}

template <typename L, typename R>
CombineMatrix<L, R> const operator*(L const &l, R const &r) {
    auto op = [](const double &l, const double &r) { return l * r; };
    return CombineMatrix<L, R>(l, r, op);
}

template <typename L, typename R>
CombineMatrix<L, R> const operator/(L const &l, R const &r) {
    auto op = [](const double &l, const double &r) { return l / r; };
    return CombineMatrix<L, R>(l, r, op);
}

class Matrix : public MatrixExpr<Matrix> {
   public:
    /**********************************************************
     * Constructors
     **********************************************************/

    // Basic ctor to initialize a matrix of size m by n.
    // All matrix positions will be initialized to 0.
    Matrix(const size_t m = 0, const size_t n = 0);

    // Iterator ctor
    template <typename IT>
    Matrix(const IT begin, const IT end, const size_t m, const size_t n)
        : Matrix(m, n) {
        size_t size_ = std::distance(begin, end);
        if (size() != size_) {
            Matrix other(begin, end, size_, 1);
            throw MatrixDimensionsMismatch(std::make_pair(nRows(), nCols()),
                                           std::make_pair(size_, 1));
        }

        auto it = begin;
        for (size_t i = 0; i < nRows(); ++i) {
            for (size_t j = 0; j < nCols(); ++j) {
                (*this)(i, j) = *it;
                ++it;
            }
        }
    }

    // Initializer list ctor
    Matrix(const std::initializer_list<double> list)
        : Matrix(list.begin(), list.end(),
                 std::distance(list.begin(), list.end()), 1) {}

    // expr convertor / COPY ctor
    template <typename R>
    Matrix(const MatrixExpr<R> &rhs) : Matrix(rhs.nRows(), rhs.nCols()) {
        for (size_t i = 0; i < nRows(); ++i) {
            for (size_t j = 0; j < nCols(); ++j) {
                (*this)(i, j) = rhs(i, j);
            }
        }
    }
    Matrix(const Matrix &rhs) : Matrix(rhs.nRows(), rhs.nCols()) {
        for (size_t i = 0; i < nRows(); ++i) {
            for (size_t j = 0; j < nCols(); ++j) {
                (*this)(i, j) = rhs(i, j);
            }
        }
    }

    // Copy assignment operator
    Matrix &operator=(const Matrix &rhs) {
        if (this != &rhs) {
            if (nRows() != rhs.nRows() || nCols() != rhs.nCols()) {
                Matrix copy(rhs);
                std::swap(*this, copy);
            } else {
                for (size_t i = 0; i < nRows(); ++i) {
                    for (size_t j = 0; j < nCols(); ++j) {
                        (*this)(i, j) = rhs(i, j);
                    }
                }
            }
        }
        return *this;
    }

    // MOVE ctor
    Matrix(Matrix &&rhs);

    // Move assignment operator
    Matrix &operator=(Matrix &&rhs);

    // dealloc matrix_ (dtor)
    ~Matrix();

    /**********************************************************
     * Operator Overloads
     **********************************************************/

    // Get number of rows (M)xN
    size_t nRows() const { return n_rows; }

    // Get number of columns Mx(N)
    size_t nCols() const { return n_cols; }

    double operator()(const size_t row, const size_t col) const {
        return rowPtrs_[row][col];
    }
    double &operator()(const size_t row, const size_t col) {
        return rowPtrs_[row][col];
    }

   private:
    size_t n_rows;      // (M)xN
    size_t n_cols;      // Mx(N)
    double *matrix_;    // A pointer to the array.
    double **rowPtrs_;  // An array of row pointers.
                        // used to avoid repeated arithmetics
                        // at each access to the matrix.
};

template <typename E>
MatrixExpr<E>::operator Matrix() const {
    Matrix n = Matrix(nRows(), nCols());
    for (size_t i = 0; i < nRows(); ++i) {
        for (size_t j = 0; j < nCols(); ++j) {
            n(i, j) = (*this)(i, j);
        }
    }
    return n;
}

#endif /* MATRIX_HPP_ */
