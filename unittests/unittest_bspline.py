from geomdl import BSpline
from geomdl.visualization import VisMPL

# Define univariate B-spline of degree 1
# u     : parameter value where the B-spline is evaluated
# order : order of the derivative
def UniformBSpline_eval_degrees1(u, order):

    bspline = BSpline.Curve()
    bspline.degree = 1
    bspline.ctrlpts = [[0.0, 0.0],
                       [0.1, 0.0],
                       [0.2, 0.0],
                       [0.3, 0.0],
                       [0.4, 0.0],
                       [0.5, 0.0],
                       [0.6, 0.0],
                       [0.7, 0.0],
                       [0.8, 0.0],
                       [0.9, 0.0],
                       [1.0, 0.0]]
    bspline.knotvector = [0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0]

    # bspline.delta = 0.01
    # bspline.vis = VisMPL.VisCurve2D()
    # bspline.render()

    return bspline.derivatives(u=u, order=order)

# Define univariate B-spline of degree 2
# u     : parameter value where the B-spline is evaluated
# order : order of the derivative
def UniformBSpline_eval_degrees2(u, order):

    bspline = BSpline.Curve()
    bspline.degree = 2
    bspline.ctrlpts = [[0.0, 0],
                       [1/9, 0],
                       [2/9, 0],
                       [3/9, 0],
                       [4/9, 0],
                       [5/9, 0],
                       [6/9, 0],
                       [7/9, 0],
                       [8/9, 0],
                       [1.0, 0]]
    bspline.knotvector = [0.0, 0.0, 0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.0, 1.0]

    # bspline.delta = 0.01
    # bspline.vis = VisMPL.VisCurve2D()
    # bspline.render()

    return bspline.derivatives(u=u, order=order)

# Define univariate B-spline of degree 3
# u     : parameter value where the B-spline is evaluated
# order : order of the derivative
def UniformBSpline_eval_degrees3(u, order):

    bspline = BSpline.Curve()
    bspline.degree = 3
    bspline.ctrlpts = [[0.000, 0],
                       [0.125, 0],
                       [0.250, 0],
                       [0.375, 0],
                       [0.500, 0],
                       [0.625, 0],
                       [0.750, 0],
                       [0.875, 0],
                       [1.000, 0]]

    bspline.knotvector = [0.0, 0.0, 0.0, 0.0, 1.0/6.0, 1.0/3.0, 0.5, 2.0/3.0, 5.0/6.0, 1.0, 1.0, 1.0, 1.0]
    
    # bspline.delta = 0.01
    # bspline.vis = VisMPL.VisCurve2D()
    # bspline.render()

    return bspline.derivatives(u=u, order=order)

# Define univariate B-spline of degree 4
# u     : parameter value where the B-spline is evaluated
# order : order of the derivative
def UniformBSpline_eval_degrees4(u, order):

    bspline = BSpline.Curve()
    bspline.degree = 4
    bspline.ctrlpts = [[0.000, 0],
                       [0.125, 0],
                       [0.250, 0],
                       [0.375, 0],
                       [0.500, 0],
                       [0.625, 0],
                       [0.750, 0],
                       [0.875, 0],
                       [1.000, 0]]

    bspline.knotvector = [0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0]
    
    # bspline.delta = 0.01
    # bspline.vis = VisMPL.VisCurve2D()
    # bspline.render()

    return bspline.derivatives(u=u, order=order)

for order in [5]:
    # UniformBSpline_eval_degrees1
    for u in [0.0, 0.2, 0.5, 0.75, 1.0]:
        print('Degree1, u=' + str(u) + ', order=' + str(order) + " : " + str(UniformBSpline_eval_degrees1(u,  order)))

    print('####################')

    # UniformBSpline_eval_degrees2
    for u in [0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0]:
        print('Degree2, u=' + str(u) + ', order=' + str(order) + " : " + str(UniformBSpline_eval_degrees2(u,  order)))

    print('####################')

    # UniformBSpline_eval_degrees3
    for u in [0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0]:
        print('Degree3, u=' + str(u) + ', order=' + str(order) + " : " + str(UniformBSpline_eval_degrees3(u,  order)))

    print('####################')

    # UniformBSpline_eval_degrees
    for u in [0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0]:
        print('Degree4, u=' + str(u) + ', order=' + str(order) + " : " + str(UniformBSpline_eval_degrees4(u,  order)))
