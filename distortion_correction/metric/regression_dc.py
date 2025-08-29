import numpy as np
from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.linear_model import Lasso, LinearRegression, Ridge, RidgeCV


from distortion_correction.base_model import DistortionCorrection
from common.geometry import distancePoint




class U_PM(DistortionCorrection):
    def __init__(self, deg=7, name='metric_poly_dc'):
        """
        deg: the highest degree of polynomial
        """
        super().__init__(name=name)
        self.set_degree(deg)
        self.scaler = StandardScaler()
        self.model = LinearRegression()  

    def set_degree(self, deg):
        self.degree = deg
        self.poly = PolynomialFeatures(self.degree, include_bias=False) #bias is included in linear regression

    def estimate(self, source, dest, **kwargs):
        features = self.poly.fit_transform(source)
        std_features = self.scaler.fit_transform(features)
        assert(std_features.shape[1] == (self.degree + 1) * (self.degree + 2) / 2 - 1)
        # print("NaN in features:", np.isnan(std_features).any())
        # print("Inf in features:", np.isinf(std_features).any())
        # print("NaN in target:", np.isnan(dest).any())
        # print("Inf in target:", np.isinf(dest).any())
        # svals = np.linalg.svd(std_features, full_matrices=False, compute_uv=False)
        # cond = svals.max() / svals.min()
        # print("condition number:", cond)
        # exit()
        # import time
        # start = time.time()
        self.model.fit(std_features, dest)

        # end = time.time()
        # print("training time pm:", end-start)
    
    def undistort(self, points):
        features = self.poly.transform(points)
        std_features = self.scaler.transform(features)
        assert(std_features.shape[1] == (self.degree + 1) * (self.degree + 2) / 2 - 1)
        # Xp = np.asarray(std_features, dtype=np.float64)   # whatever you feed into predict
        # print("predict dtype:", Xp.dtype)
        # print("predict finite:", np.isfinite(Xp).all())
        # print("predict max|X|:", np.abs(Xp).max())
        # print("any object dtype upstream?", getattr(getattr(std_features, 'dtypes', None), 'unique', lambda:[])())
        
        undist = self.model.predict(std_features)
        # exit()
        return undist

class U_RFM_DC(DistortionCorrection):
    def __init__(self, deg=7, name='metric_rfm_dc'):
        """
        deg: the highest degree of polynomial
        """
        super().__init__(name=name)
        
        self.set_degree(deg)
        self.scaler = StandardScaler()
        self.model = LinearRegression(fit_intercept=False)
        self.homo = None

    def set_degree(self, deg):
        self.degree = deg
        self.poly = PolynomialFeatures(self.degree) #bias is included in linear regression

    def estimate(self, source, dest, **kwargs):
        super().estimate(source, dest)
        #
        features = self.poly.fit_transform(source)
        std_features = self.scaler.fit_transform(features)
        std_features[:, 0] = 1 #常数项的特征scaler以后会变成0，需要置为1
        assert(std_features.shape[1] == (self.degree + 1) * (self.degree + 2) / 2)
        features_x = -std_features[:, 1:] * dest[:, :1] # 分母多项式少一项
        features_y = -std_features[:, 1:] * dest[:, 1:]
        zeros = np.zeros_like(std_features)
        mat_x = np.column_stack([std_features, zeros, features_x])
        mat_y = np.column_stack([zeros, std_features, features_y])
        input_features = np.concatenate([mat_x, mat_y])
        target = np.concatenate([dest[:, 0], dest[:, 1]])
        # print(input_features.shape)
        # import time
        # start = time.time()
        self.model.fit(input_features, target)
        # end = time.time()
        # print("training time rfm:", end-start)
        return
        predictions = self.model.predict(input_features)
        x_coords = predictions[:1380]
        y_coords = predictions[1380:]

        # Stack the x and y coordinates along the second axis
        points_array = np.column_stack((x_coords, y_coords))
 
        residuals = np.linalg.norm(points_array - dest, axis=1)
        std_residuals = np.std(residuals)
        outlier_indices = residuals > 5*std_residuals
        # cleaned_source = source[~outlier_indices]
        # import matplotlib.pyplot as plt
        # plt.scatter(cleaned_source[:, 0], cleaned_source[:, 1])
        # plt.show()
        # exit()

        outlier_indices = np.concatenate([outlier_indices, outlier_indices])
        # Remove outliers from the dataset
        cleaned_input_features = input_features[~outlier_indices]
        cleaned_target = target[~outlier_indices]

        
        # Refit the model using cleaned dataset
        self.model.fit(cleaned_input_features, cleaned_target)
   
    
    def undistort(self, points):
        features = self.poly.transform(points)
        std_features = self.scaler.transform(features)
        std_features[:, 0] = 1
        param_count = std_features.shape[1]
        param_a = self.model.coef_[:param_count]
        param_b = self.model.coef_[param_count:2*param_count]
        param_c = self.model.coef_[2 * param_count:]
        den = 1 + np.dot(std_features[:, 1:], param_c)
        num_x = np.dot(std_features, param_a)
        num_y = np.dot(std_features, param_b)
        xu = num_x / den
        yu = num_y / den
        return np.column_stack([xu, yu])


# 中心畸变模型实际上需要畸变图像的cod_d和零畸变图像的cod_u两个cod参数
# RTM模型的cod_u可作为线性项求解，DM模型则是非线性项，需要预估
# 在过去的方法中，通过矩阵拆分获得的cod实际上是
# the model does not assume k0=1 to be generalizable to scaling
class MetricRTMDC(DistortionCorrection):
    def __init__(self, deg=7, tangential=True, name='metric_rtm_dc'):
        """
        deg: the highest degree of polynomial
        """
        super().__init__(name=name)
        self.set_degree(deg)
        self.tangential = tangential
        self.scaler = StandardScaler()
        self.model = LinearRegression(fit_intercept=False)

    def set_degree(self, deg):
        super().set_degree(deg)
        self.terms = (deg - 1) // 2

    def gen_features(self, source, is_train):
        xd = source[:, 0] - self.cod[0]
        yd = source[:, 1] - self.cod[1]
        r2 = xd**2 + yd**2
        
        features_x = list()
        # contains the zero order term of r2
        for i in range(self.terms + 1):
            features_x.append(xd * r2**(i + 0))
        if self.tangential:
            features_x.append(r2 + 2 * xd**2)
            features_x.append(2 * xd * yd)
        features_x = np.column_stack(features_x)
        
        features_y = list()
        for i in range(self.terms + 1):
            features_y.append(yd * r2**(i + 0))
        if self.tangential:
            features_y.append(2 * xd * yd)
            features_y.append(r2 + 2 * yd**2)
        features_y = np.column_stack(features_y)

        features = np.concatenate([features_x, features_y])
        length = len(source)
        features = np.column_stack([features, np.ones((length * 2, 2))])
        features[:length, -1] = 0
        features[length:, -2] = 0

        return features

    def estimate(self, source, dest, cod=(0, 0)):
        self.cod = cod
        features = self.gen_features(source, is_train=True)

        target = np.concatenate([dest[:, 0], dest[:, 1]])

        self.model.fit(features, target)

    
    def undistort(self, points):
        features = self.gen_features(points, is_train=False)
        std_features = features#self.scaler.transform(features)
        undist_x_y = self.model.predict(std_features)
        s = len(undist_x_y)
        x = undist_x_y[0: s // 2]
        y = undist_x_y[s // 2:]
        return np.column_stack([x, y])


class MetricRTMDC1(DistortionCorrection):
    def __init__(self, deg=7, tangential=True, name='metric_rtm_dc'):
        """
        deg: the highest degree of polynomial
        """
        super().__init__(name=name)

        self.set_degree(deg)
        self.tangential = tangential
        self.scaler = StandardScaler()
        # self.scalert = StandardScaler()
        # self.model = Ridge(fit_intercept=False)
        # linearRegression may fail when constant terms cx and cy are included and no normalization is conducted
        # In such cases, use Ridge instead
        self.model = LinearRegression(fit_intercept=False)
        # the model itself shall have no intercept term in the linear model.
        # The cod in the distortion-free coordinates are considored different from the cod in the distorted image
        # This is similar to the intrinsic matrix that shift the pixels by a fixed amount
        # The cod_u are estimated as unknowns, to support translation invariance of results.
        # this is most helpful for decoupled distortion correction, because the cod_u is usually not the same as cod_d
        
        # When the cod_u is not included as unknowns, and the inputs are scaled, intercep must be fitted to address posterior shift
        # otherwise, the target can also be normalized, but not recommended

        # In general, the best implementation is to (1)include cod_u as unkown, (2) scale the input with standard scaler
        # (3) use the linearRegression model with fit_intercept set to False

    def set_degree(self, deg):
        super().set_degree(deg)
        self.terms = (deg - 1) // 2

    def gen_features(self, source, istrain):
        xd = source[:, 0] - self.cod[0]
        yd = source[:, 1] - self.cod[1]
        r2 = xd**2 + yd**2
        features_x = list()
        for i in range(self.terms):
            features_x.append(xd * r2**(i + 1))
        if self.tangential:
            features_x.append(r2 + 2 * xd**2)
            features_x.append(2 * xd * yd)
        features_x = np.column_stack(features_x)

        features_y = list()
        for i in range(self.terms):
            features_y.append(yd * r2**(i + 1))
        if self.tangential:
            features_y.append(2 * xd * yd)
            features_y.append(r2 + 2 * yd**2)
        features_y = np.column_stack(features_y)

        features = np.concatenate([features_x, features_y])

        if istrain:
            std_features = self.scaler.fit_transform(features)
        else:
            std_features = self.scaler.transform(features)
        length = len(source)
        features = np.column_stack([std_features, np.ones((length * 2, 2))])
        features[:length, -1] = 0
        features[length:, -2] = 0

        return features

    def estimate(self, source, dest, cod=(0, 0)):
        self.cod = cod
        features = self.gen_features(source, istrain=True)

        # std_features = self.scaler.fit_transform(features)
        target = np.concatenate([dest[:, 0] - source[:, 0], dest[:, 1] - source[:, 1]])
        # target = self.scalert.fit_transform(target.reshape(-1, 1))
        self.model.fit(features, target)
    
    def undistort(self, points):
        features = self.gen_features(points, istrain=False)
        # std_features = self.scaler.transform(features)
        undist_x_y = self.model.predict(features)
        
        # undist_x_y = self.scalert.inverse_transform(undist_x_y)

        s = len(undist_x_y)
        x = undist_x_y[: s // 2] + points[:, 0]
        y = undist_x_y[s // 2:] + points[:, 1]

        return np.column_stack([x, y])

# RTM model with no given COD for distortion correction
# 
class U_RTM_N(DistortionCorrection):
    """
    RTM model with no given COD for distortion correction.
    """

    def __init__(self, deg=7, tangential=True, name='metric_rtm_dc', fixk1=False):
        """
        Initialize the distortion correction model.

        Parameters:
        deg (int): The highest degree of the polynomial.
        tangential (bool): Whether to include tangential distortion.
        name (str): The name of the distortion correction model.
        """
        super().__init__(name=name)
  
        self.tangential = tangential

        self.fixk1 = fixk1
        self.set_degree(deg)
        
    def set_degree(self, deg):
        """
        Set the degree of the polynomial and calculate the number of parameters.

        Parameters:
        deg (int): The highest degree of the polynomial.
        """
        super().set_degree(deg)
        self.terms = (deg + 1) // 2
        if self.fixk1:
            self.terms -= 1
        self.param_count = 2 + self.terms + 2 # 2 for cod_d, 2 for cod_u

        if self.tangential:
            self.param_count += 2
        # print("number of parameters:", self.param_count)
        self.params:list = [0] * self.param_count

    def model_summary(self):
        cod_d = self.params[:2]
        cod_u = self.params[2:4]
        print("cod_d: ", cod_d)
        print("cod_u: ", cod_u)
        dist_coef = self.params[4:]
        print("dist_coef: ", dist_coef)


    def __undistort(self, xdata, *params):
        """
        Apply undistortion transformation to the input data.

        Parameters:
        xdata (numpy.ndarray): The input data with x and y coordinates.
        params (tuple): The distortion parameters.

        Returns:
        numpy.ndarray: The undistorted coordinates.
        """
        x = xdata[:, 0] - params[0]
        y = xdata[:, 1] - params[1]
        r2 = x**2 + y**2
        ratio = None
        if self.fixk1:
            ratio = np.ones_like(x)
            for i in range(self.terms):
                ratio += r2**(i + 1) * params[i + 4]
        else:
            ratio = np.zeros_like(x)
            for i in range(self.terms):
                ratio += r2**(i) * params[i + 4]
        xu = x * ratio + params[2]
        yu = y * ratio + params[3]
        if self.tangential:
            xu += params[-2] * (r2 + 2 * x**2) + 2 * params[-1] * x * y
            yu += params[-1] * (r2 + 2 * y**2) + 2 * params[-2] * x * y
        result = np.concatenate([xu, yu])
        return result

    def estimate(self, source, dest):
        """
        Estimate distortion parameters by minimizing the reprojection error.

        Parameters:
        source (numpy.ndarray): The source points (distorted).
        dest (numpy.ndarray): The destination points (undistorted).
        """
        # seems to have no way to normaliza data
        from scipy.optimize import curve_fit
        # Flatten destination coordinates for curve fitting         
        dest = np.concatenate([dest[:, 0], dest[:, 1]])
        # Optimize parameters using Levenberg-Marquardt algorithm
        popt, pcov = curve_fit(self.__undistort, source, dest, self.params, method='lm')
        self.params = list(popt)
        # print(self.params)
    
    def undistort(self, points):
        """
        Undistort a set of points using the estimated parameters.

        Parameters:
        points (numpy.ndarray): The points to undistort.

        Returns:
        numpy.ndarray: The undistorted points.
        """
        num = points.shape[0]
        result = self.__undistort(points, *self.params)
        x = result[:num]
        y = result[num:]
        return np.column_stack([x, y])

# DM model with no given COD for distortion correction
# the curvefit does not converge properly with fractional function, interesting
# the parameters are fitted by rearange demonimator to the other side
class U_DM_N(DistortionCorrection):
    """
    RTM model with no given COD for distortion correction.
    """

    def __init__(self, deg=7, name='metric_dm_dc', fixk1=False):
        """
        Initialize the distortion correction model.

        Parameters:
        deg (int): The highest degree of the polynomial.
        tangential (bool): Whether to include tangential distortion.
        name (str): The name of the distortion correction model.
        """
        super().__init__(name=name)

        self.fixk1 = fixk1
        self.set_degree(deg)
        


    def set_degree(self, deg):
        """
        Set the degree of the polynomial and calculate the number of parameters.

        Parameters:
        deg (int): The highest degree of the polynomial.
        """
        super().set_degree(deg)
        self.terms = (deg + 1) // 2
        if self.fixk1:
            self.terms -= 1
        self.param_count = 2 + self.terms + 2 # 2 for cod_d, 2 for cod_u

        # print("number of parameters:", self.param_count)
        self.params:list = [0] * self.param_count
    
    def model_summary(self):
        cod_d = self.params[:2]
        cod_u = self.params[2:4]
        print("cod_d: ", cod_d)
        print("cod_u: ", cod_u)
        dist_coef = self.params[4:]
        print("dist_coef: ", dist_coef)

    def __estimate(self, xdata, *params):
        xyd = xdata[0]
        xyu = xdata[1]
        xdc = xyd[:, 0] - params[0]
        ydc = xyd[:, 1] - params[1]

        xuc = xyu[:, 0] - params[2]
        yuc = xyu[:, 1] - params[3]

        r2 = xdc**2 + ydc**2
        if self.fixk1:
            ratio = np.ones_like(xdc)
            for i in range(self.terms):
                ratio += r2**(i + 1) * params[i + 4]
        else:
            ratio = np.zeros_like(xdc)
            for i in range(self.terms):
                ratio += r2**(i) * params[i + 4]

        xd = xuc * ratio + params[0]
        yd = yuc * ratio + params[1]


        result = np.concatenate([xd, yd])
        return result

    def __undistort(self, xdata, *params):
        """
        Apply undistortion transformation to the input data.

        Parameters:
        xdata (numpy.ndarray): The input data with x and y coordinates.
        params (tuple): The distortion parameters.

        Returns:
        numpy.ndarray: The undistorted coordinates.
        """
        x = xdata[:, 0] - params[0]
        y = xdata[:, 1] - params[1]
        r2 = x**2 + y**2
        ratio = None
        if self.fixk1:
            ratio = np.ones_like(x)
            for i in range(self.terms):
                ratio += r2**(i + 1) * params[i + 4]
        else:
            ratio = np.zeros_like(x)
            for i in range(self.terms):
                ratio += r2**(i) * params[i + 4]

        xu = x / ratio + params[2]
        yu = y / ratio + params[3]

        result = np.concatenate([xu, yu])
        return result

    def estimate(self, source, dest):
        """
        Estimate distortion parameters by minimizing the reprojection error.

        Parameters:
        source (numpy.ndarray): The source points (distorted).
        dest (numpy.ndarray): The destination points (undistorted).
        """
        # seems to have no way to normaliza data
        from scipy.optimize import curve_fit
        # Flatten destination coordinates for curve fitting        
        x = (source, dest)
        y = np.concatenate([source[:, 0], source[:, 1]])
        # Optimize parameters using Levenberg-Marquardt algorithm
        popt, pcov = curve_fit(self.__estimate, x, y, self.params, method='lm')
        self.params = list(popt)
        # print(self.params)
    
    def undistort(self, points):
        """
        Undistort a set of points using the estimated parameters.

        Parameters:
        points (numpy.ndarray): The points to undistort.

        Returns:
        numpy.ndarray: The undistorted points.
        """
        num = points.shape[0]
        result = self.__undistort(points, *self.params)
        x = result[:num]
        y = result[num:]
        return np.column_stack([x, y])

class MetricDMDC(DistortionCorrection):
    def __init__(self, deg=7, name='metric_dm_dc'):
        """
        terms: the number of radial terms
        """
        super().__init__(name=name)
        self.set_degree(deg)
        self.scaler = StandardScaler()
        self.model = LinearRegression(fit_intercept=False)# 只能畸变前后坐标系一致

    def set_degree(self, deg):
        super().set_degree(deg)
        self.terms = (deg - 1) // 2

    def gen_features(self, source):
        xd = source[:, 0] - self.cod[0]
        yd = source[:, 1] - self.cod[1]
        r2 = xd**2 + yd**2
        features = list()
        # for i in range(self.terms):
        #     features.append(r2**(i + 1))
        for i in range(self.terms + 1):
            features.append(r2**(i + 0))
        features = np.column_stack(features)
        return features

    def estimate(self, source, dest, cod=(0, 0)):
        self.cod = cod
        features = self.gen_features(source)
        std_features = self.scaler.fit_transform(features)
        std_features[:, 0] = 1
        # dest坐标不能参与normalization 计算
        features_x = std_features * (dest[:, :1] - cod[0])
        features_y = std_features * (dest[:, 1:] - cod[1])
        input_features = np.concatenate([features_x, features_y])
        # target = np.concatenate([source[:, 0] - dest[:, 0], source[:, 1] - dest[:, 1]])
        target = np.concatenate([source[:, 0] - cod[0], source[:, 1] - cod[1]])
        self.model.fit(input_features, target)
    
    def undistort(self, points):
        features = self.gen_features(points)
        std_features = self.scaler.transform(features)
        std_features[:, 0] = 1
        # den = 1 + np.dot(std_features, self.model.coef_)
        den = np.dot(std_features, self.model.coef_)
        xu = (points[:, 0] - self.cod[0]) / den + self.cod[0]
        yu = (points[:, 1] - self.cod[1]) / den + self.cod[1]
        return np.column_stack([xu, yu])

    def evaluate(self, source, dest):
        pred = self.undistort(source)
        mse = np.mean(distancePoint(dest, pred))
        return mse
    

# not a good idea
class MetricRCPMDC(DistortionCorrection):
    def __init__(self, deg=7, name='metric_rcpm_dc'):
        """
        deg: the highest degree of polynomial
        """
        super().__init__(name=name)
        self.set_degree(deg)
        
        self.scaler_radial = StandardScaler()
        self.scaler_poly = StandardScaler()
        # self.model = LinearRegression(fit_intercept=False, positive=True)
        self.model = Ridge(fit_intercept=False, alpha=0.0001)
    
    def set_degree(self, deg):
        self.degree = deg
        self.poly = PolynomialFeatures(self.degree) 
        self.terms = (deg - 1) // 2

    def gen_features(self, source):
        xd = source[:, 0] - self.cod[0]
        yd = source[:, 1] - self.cod[1]
        r2 = xd**2 + yd**2
        r = np.sqrt(r2)
        feature_x = list()
        feature_y = list()
        feature_radial = list()
        for i in range(self.terms+1):
            # feature_x.append(xd * r**(i + 1))
            # feature_y.append(yd * r**(i + 1))
            feature_radial.append(r2**(i + 0))
        # feature_x = np.column_stack(feature_x)
        # feature_y = np.column_stack(feature_y)
        feature_radial = np.column_stack(feature_radial)
        features_poly = self.poly.fit_transform(source)
        # features_poly = np.delete(features_poly, [1, 2], 1)

        return feature_radial, features_poly

    def estimate(self, source, dest, cod=(0, 0)):
        self.cod = cod
        xd = source[:, 0] - self.cod[0]
        yd = source[:, 1] - self.cod[1]
        xd = xd[:, np.newaxis]
        yd = yd[:, np.newaxis]
        radial, poly = self.gen_features(source)
        std_radial = self.scaler_radial.fit_transform(radial)
        std_poly = self.scaler_poly.fit_transform(poly)
        std_poly[:, 0] = 1

        zeros = np.zeros_like(std_poly)
        mat_x = np.column_stack([xd * std_radial, std_poly, zeros])
        mat_y = np.column_stack([yd * std_radial, zeros, std_poly])
        input_features = np.concatenate([mat_x, mat_y])
        target = np.concatenate([dest[:, 0], dest[:, 1]])
        self.model.fit(input_features, target)

    def undistort(self, points):
        xd = points[:, 0] - self.cod[0]
        yd = points[:, 1] - self.cod[1]
        xd = xd[:, np.newaxis]
        yd = yd[:, np.newaxis]
        radial, poly = self.gen_features(points)
        std_radial = self.scaler_radial.transform(radial)
        std_poly = self.scaler_poly.transform(poly)
        std_poly[:, 0] = 1

        radial_count = std_radial.shape[1]
        poly_count = std_poly.shape[1]
        param_k = self.model.coef_[:radial_count]
        param_a = self.model.coef_[radial_count:radial_count+poly_count]
        param_b = self.model.coef_[radial_count+poly_count:]
        xu = np.dot(xd * std_radial, param_k) + np.dot(std_poly, param_a)
        yu = np.dot(yd * std_radial, param_k) + np.dot(std_poly, param_b)

        return np.column_stack([xu, yu])

# not a good idea
class MetricRadialPolyDC(DistortionCorrection):
    def __init__(self, deg=7, name='metric_radialpoly_dc'):
        super().__init__(name)
        self.radial = MetricRTMDC(deg, False)
        self.poly = U_RFM_DC(deg)
        self.set_degree(deg)
    
    def set_degree(self, deg):
        self.radial.set_degree(deg)
        self.poly.set_degree(deg)
    
    def estimate(self, source, target, cod):
        self.radial.estimate(source, target, cod)
        image_t = self.radial.undistort(source)
        self.poly.estimate(image_t, target)
    
    def undistort(self, points):
        image_t = self.radial.undistort(points)
        image_u = self.poly.undistort(image_t)
        return image_u

# not a good idea
# class DPMetricMLPDC(DistortionCorrection):
#     def __init__(self, neurons=100, hlayer=1, name='metric_mlp_dc'):
#         super().__init__(name=name)
#         self.hidden_layers = np.ones((hlayer,), dtype='int')*neurons
#         self.scaler = StandardScaler()
#         self.modelx = MLPRegressor(hidden_layer_sizes=self.hidden_layers, learning_rate='adaptive', max_iter=10000)
#         self.modely = MLPRegressor(hidden_layer_sizes=self.hidden_layers, learning_rate='adaptive', max_iter=10000)
    
#     def estimate(self, source, target):
#         std_features = self.scaler.fit_transform(source)
#         print(self.hidden_layers)
#         self.modelx.fit(std_features, target[:, 0])
#         self.modely.fit(std_features, target[:, 1])
    
#     def undistort(self, points):
#         std_features = self.scaler.transform(points)
#         undistx = self.modelx.predict(std_features)
#         undisty = self.modely.predict(std_features)
#         undist = np.column_stack([undistx, undisty])

#         return undist

def train_decoupled(name, path, size, model=U_RFM_DC, reverse=False):
    import cv2
    from common.chessboard import findchessboard, create_ground_truth
    img = cv2.imread(path, 0)

    corners = findchessboard(image=img, size=size)
    target = create_ground_truth(size)
    # import matplotlib.pyplot as plt
    # print(corners)
    # print(target)
    if reverse:
        corners, target = target, corners
    de = U_RFM_DC(3)
    # X_train, X_test, y_train, y_test = train_test_split(corners, target, test_size=0.8)
    de.estimate(corners, target)
    mse = de.evaluate(corners, target)
    print(mse)
    de.save_model(name)


if __name__ == "__main__":
    pass
    # train_decoupled('decoupled_glass_11x8', 'data/decoupled/glass_8x11full.bmp', (8, 11))
    train_decoupled('decoupled_glass_30x46', 'data/decoupled/46x30.bmp', (30, 46))
    # train_decoupled('de_laptop', 'data/decoupled/laptop/46x30.jpg', (30, 46))
    # train_decoupled('de_laptop_c1', 'data/decoupled/laptop/8x11.jpg', (8, 11))

    # train_decoupled('de_laptop_reverse', 'data/decoupled/laptop/46x30.jpg', (30, 46), reverse=True)
