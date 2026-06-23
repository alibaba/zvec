// Temporary LNK1189 probe: heavy Eigen template instantiation.
// If the component library is still built as SHARED with
// WINDOWS_EXPORT_ALL_SYMBOLS on MSVC, these symbols will push
// the .def file past the PE/COFF 65535-section limit → LNK1189.
// After confirming the fix works, REMOVE this file and revert
// the rabitqlib addition in CMakeLists.txt.

#include <rabitqlib/third/Eigen/Dense>
#include <rabitqlib/third/Eigen/Sparse>
#include <vector>

namespace zvec {
namespace core {

// Each explicit instantiation of Eigen operations generates many
// COMDAT sections (matrix ops, decompositions, solvers, etc.)

#define PROBE_EIGEN_TAG(SCALAR, TAG, ROWS, COLS)                         \
  void _probe_eigen_impl_##SCALAR##_##TAG(                               \
      const Eigen::Matrix<SCALAR, ROWS, COLS>& m) {                      \
    auto a = m + m;                                                      \
    auto b = m * 2.0f;                                                   \
    auto c = m.transpose() * m;                                          \
    auto d = m.array() * m.array();                                      \
    (void)a; (void)b; (void)c; (void)d;                                  \
  }

#define PROBE_EIGEN(SCALAR, ROWS, COLS) \
  PROBE_EIGEN_TAG(SCALAR, ROWS##x##COLS, ROWS, COLS)

// Generate instantiations for various matrix sizes
// Each generates ~hundreds of COMDAT sections with Eigen
PROBE_EIGEN(float, 2, 2)
PROBE_EIGEN(float, 3, 3)
PROBE_EIGEN(float, 4, 4)
PROBE_EIGEN(float, 8, 8)
PROBE_EIGEN(float, 16, 16)
PROBE_EIGEN(float, 32, 32)
PROBE_EIGEN(float, 64, 64)
PROBE_EIGEN(float, 128, 128)
PROBE_EIGEN(float, 256, 256)
PROBE_EIGEN(float, 512, 512)
PROBE_EIGEN(float, 768, 768)
PROBE_EIGEN(float, 1024, 1024)
PROBE_EIGEN(float, 1536, 1536)
PROBE_EIGEN(float, 2048, 2048)
PROBE_EIGEN_TAG(float, Dyn, Eigen::Dynamic, Eigen::Dynamic)
PROBE_EIGEN(double, 2, 2)
PROBE_EIGEN(double, 3, 3)
PROBE_EIGEN(double, 4, 4)
PROBE_EIGEN(double, 8, 8)
PROBE_EIGEN(double, 16, 16)
PROBE_EIGEN(double, 32, 32)
PROBE_EIGEN(double, 64, 64)
PROBE_EIGEN(double, 128, 128)
PROBE_EIGEN_TAG(double, Dyn, Eigen::Dynamic, Eigen::Dynamic)

#undef PROBE_EIGEN

// More heavy operations: decompositions, solvers
void _probe_decomp_f(const Eigen::MatrixXf& m) {
  Eigen::LLT<Eigen::MatrixXf> llt(m);
  Eigen::JacobiSVD<Eigen::MatrixXf> svd(m);
  Eigen::HouseholderQR<Eigen::MatrixXf> qr(m);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eigen(m);
  (void)llt; (void)svd; (void)qr; (void)eigen;
}

void _probe_decomp_d(const Eigen::MatrixXd& m) {
  Eigen::LLT<Eigen::MatrixXd> llt(m);
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(m);
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(m);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(m);
  (void)llt; (void)svd; (void)qr; (void)eigen;
}

// Sparse matrix operations
void _probe_sparse_f(const Eigen::SparseMatrix<float>& m) {
  Eigen::SparseLU<Eigen::SparseMatrix<float>> lu(m);
  (void)lu;
}

void _probe_sparse_d(const Eigen::SparseMatrix<double>& m) {
  Eigen::SparseLU<Eigen::SparseMatrix<double>> lu(m);
  (void)lu;
}

// Exported function to ensure this TU is linked
extern "C" int lnk1189_probe_trigger() { return 42; }

}  // namespace core
}  // namespace zvec
