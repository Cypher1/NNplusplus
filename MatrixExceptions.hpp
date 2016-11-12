//
//  MatrixExceptions.h
//  Neural Net
//
//  Created by Gil Dekel on 8/28/16.
//  Last edited by Joshua Pratt on 10/01/16.
//

#ifndef MATRIX_EXCEPTIONS_HPP
#define MATRIX_EXCEPTIONS_HPP

#include <exception>
#include <string>
#include <utility>

template<typename E>
class MatrixExpr;

class MatrixDimensionsMismatch : public std::exception {
   public:
    MatrixDimensionsMismatch(const std::pair<size_t, size_t> &expected,
                             const std::pair<size_t, size_t> &actual)
        : expected{expected}, actual{actual} {}

    const char *what() const noexcept {
        // const char *MatrixDimensionsMismatch::what() const noexcept {
        std::string expected_s = std::to_string(expected.first) + "x" +
                                 std::to_string(expected.second);
        std::string actual_s =
            std::to_string(actual.first) + "x" + std::to_string(actual.second);
        std::string what = "Matrix dimensions must be equal. \n\tGot [" +
                           actual_s + "] expected [" + expected_s + "].\n";
        return what.c_str();
    }

   private:
    const std::pair<size_t, size_t> &expected;
    const std::pair<size_t, size_t> &actual;
};

#endif /* MATRIX_EXCEPTIONS_HPP */
