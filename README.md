1. A Configurable Pipeline for Mass Univariate Aggregation Methods:
We present a unified, flexible, and accessible configurable pipeline that can be used for implementing CPM (Shen et al., 2017), PNRS (Byington et al., 2023), and many new MUA configurations facilitated by user-specified parameters. The pipeline’s core is built around two classes—FeatureVectorizer and MUA—both of which are engineered to be fully compatible with the scikit-learn ecosystem.
2. The Implementation Details:
2.1 FeatureVectorizer Class
   The FeatureVectorizer class helps to handle both 2D and 3D input data. In this manner, our configurable pipeline can be used to apply MUA methods on neuroimaging data (connectivity matrices) and any other feature-outcome data.
2.2 MUA Class:
   The MUA class is our main Python class that extends scikit-learn's BaseEstimator and TransformerMixin classes to ensure compatibility with standard machine learning pipelines; the MUA class operates through three steps:
1. Feature Selection and Organization: Determines which features are included and how they are organized (e.g., split into positive/negative networks or combined)
2. Feature Weighting: Specifies how selected features are weighted (e.g., binary, correlation-based, or regression-derived weights)
3. Feature Aggregation: Defines how weighted features are combined into predictive scores (e.g., sum or mean aggregation)   

The MUA class employs seven configurable parameters organized across the three computational steps described above that enable flexible implementation of established methods while facilitating exploration of novel approaches. 

split_by_sign
  True: Separate positive/negative features
  False: Keep all features together

