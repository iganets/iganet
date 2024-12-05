//! [Include namespace]
#include <iganet.h>

using namespace iganet;
//! [Include namespace]

int main() {
  //! [Initialize internals]
  init();
  //! [Initialize internals]

  //! [Univariate uniform B-spline of degree 2]
  UniformBSpline<double, 1, 2> C({6});
  //! [Univariate uniform B-spline of degree 2]

  //! [Print univariate uniform B-spline of degree 2]
  Log() << C << std::endl;
  //! [Print univariate uniform B-spline of degree 2]

  //! [Print verbose univariate uniform B-spline of degree 2]
  verbose(std::cout);
  Log() << C << std::endl;
  //! [Print verbose univariate uniform B-spline of degree 2]

  //! [Print knots of univariate uniform B-spline of degree 2]
  Log() << C.nknots() << "\n" << C.knots() << "\n";
  //! [Print knots of univariate uniform B-spline of degree 2]

  //! [Print knots of univariate uniform B-spline of degree 2 per dimension]
  Log() << C.nknots(0) << "\n" << C.knots(0) << "\n";
  //! [Print knots of univariate uniform B-spline of degree 2 per dimension]

  //! [Multivariate uniform B-splines]
  UniformBSpline<double, 3, 2, 3>    S({6, 8});
  UniformBSpline<double, 3, 2, 3, 2> V({6, 8, 5});
  //! [Multivariate uniform B-splines]

  Log() << S << "\n" << V << "\n";
  
  //! [Clean up internals]
  finalize();
  //! [Clean up internals]
  
  return 0;
}
