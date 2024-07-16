import numpy as np
from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.linear_model import Lasso, LinearRegression, Ridge, RidgeCV

from distortion_correction.base_model import DistortionCorrection
from common.geometry import distancePoint



class MetricPolyDC(DistortionCorrection):
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
        self.model.fit(std_features, dest)
    
    def undistort(self, points):
        features = self.poly.transform(points)
        std_features = self.scaler.transform(features)
        undist = self.model.predict(std_features)
        return undist

class MetricRFMDC(DistortionCorrection):
    def __init__(self, deg=7, name='metric_rfm_dc'):
        """
        deg: the highest degree of polynomial
        """
        super().__init__(name=name)
        
        self.set_degree(deg)
        self.scaler = StandardScaler()
        self.model = LinearRegression(fit_intercept=False)

    def set_degree(self, deg):
        self.degree = deg
        self.poly = PolynomialFeatures(self.degree) #bias is included in linear regression

    def estimate(self, source, dest, **kwargs):
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
        self.model.fit(input_features, target)      
    
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
class MetricRTMDC(DistortionCorrection):
    def __init__(self, deg=7, tangential=True, name='metric_rtm_dc'):
        """
        deg: the highest degree of polynomial
        """
        super().__init__(name=name)
        self.set_degree(deg)
        self.tangential = tangential
        self.scaler = StandardScaler()
        self.model = LinearRegression(fit_intercept=True)

    def set_degree(self, deg):
        super().set_degree(deg)
        self.terms = (deg - 1) // 2

    def gen_features(self, source):
        xd = source[:, 0] - self.cod[0]
        yd = source[:, 1] - self.cod[1]
        r2 = xd**2 + yd**2
        
        features_x = list()
        # 是否包含0次项
        # for i in range(self.terms):
        #     features_x.append(xd * r2**(i + 1))
        for i in range(self.terms + 1):
            features_x.append(xd * r2**(i + 0))

        if self.tangential:
            features_x.append(r2 + 2 * xd**2)
            features_x.append(2 * xd * yd)
        features_x = np.column_stack(features_x)

        features_y = list()
        # for i in range(self.terms):
        #     features_y.append(yd * r2**(i + 1))
        for i in range(self.terms + 1):
            features_y.append(yd * r2**(i + 0))
        if self.tangential:
            features_y.append(2 * xd * yd)
            features_y.append(r2 + 2 * yd**2)
        features_y = np.column_stack(features_y)

        features = np.concatenate([features_x, features_y])
        return features

    def estimate(self, source, dest, cod=(0, 0)):
        self.cod = cod
        features = self.gen_features(source)
        std_features = self.scaler.fit_transform(features)
        # target = np.concatenate([dest[:, 0] - source[:, 0], dest[:, 1] - source[:, 1]])
        target = np.concatenate([dest[:, 0], dest[:, 1]])

        self.model.fit(std_features, target)

    
    def undistort(self, points):
        features = self.gen_features(points)
        std_features = self.scaler.transform(features)
        undist_x_y = self.model.predict(std_features)
        s = len(undist_x_y)
        x = undist_x_y[0: s // 2]# + points[:, 0]
        y = undist_x_y[s // 2:]# + points[:, 1]
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
        self.model = LinearRegression(fit_intercept=False)

    def set_degree(self, deg):
        super().set_degree(deg)
        self.terms = (deg - 1) // 2

    def gen_features(self, source):
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
        return features

    def estimate(self, source, dest, cod=(0, 0)):
        self.cod = cod
        features = self.gen_features(source)
        std_features = features #self.scaler.fit_transform(features)
        target = np.concatenate([dest[:, 0] - source[:, 0], dest[:, 1] - source[:, 1]])
        self.model.fit(std_features, target)
    
    def undistort(self, points):
        features = self.gen_features(points)
        std_features = features #self.scaler.transform(features)
        undist_x_y = self.model.predict(std_features)
        s = len(undist_x_y)
        x = undist_x_y[0: s // 2] + points[:, 0]
        y = undist_x_y[s // 2:] + points[:, 1]
        return np.column_stack([x, y])
    
class MetricRTMDCN(DistortionCorrection):
    def __init__(self, deg=7, tangential=True, name='metric_rtm_dc'):
        """
        deg: the highest degree of polynomial
        """
        super().__init__(name=name)        
        self.tangential = tangential
        self.set_degree(deg)
        self.scaler = StandardScaler()

    def set_degree(self, deg):
        super().set_degree(deg)
        self.terms = (deg - 1) // 2
        self.param_count = 2 + self.terms
        if self.tangential:
            self.param_count += 2
        self.params:list = [0] * self.param_count
    
    def __undistort(self, xdata, *params):
            x = xdata[:, 0] - params[0]
            y = xdata[:, 1] - params[1]
            r2 = x**2 + y**2
            ratio = np.ones_like(x)
            for i in range(self.terms):
                ratio += r2**(i + 1) * params[i + 2]
            xu = x * ratio + params[0]
            yu = y * ratio + params[1]
            if self.tangential:
                xu += params[-2] * (r2 + 2 * x**2) + 2 * params[-1] * x * y
                yu += params[-1] * (r2 + 2 * y**2) + 2 * params[-2] * x * y
            result = np.concatenate([xu, yu])
            return result

    def estimate(self, source, dest):
        from scipy.optimize import curve_fit
        # Minimize reprojection error with Levenberg-Marquardt              
        dest = np.concatenate([dest[:, 0], dest[:, 1]])
        popt, pcov = curve_fit(self.__undistort, source, dest, self.params)
        self.params = list(popt)
        # print(self.params)
    
    def undistort(self, points):
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

class MetricRadialPolyDC(DistortionCorrection):
    def __init__(self, deg=7, name='metric_radialpoly_dc'):
        super().__init__(name)
        self.radial = MetricRTMDC(deg, False)
        self.poly = MetricRFMDC(deg)
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

