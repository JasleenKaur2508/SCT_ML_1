# 🏘️ INTELLIGENT HOUSING MARKET ANALYZER
# Multi-Algorithm Approach to California Real Estate Prediction
# Featuring: Random Forest, Gradient Boosting, Neural Networks & Deep Analysis

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from scipy import stats
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
warnings.filterwarnings('ignore')

class HousingMarketAnalyzer:
    def __init__(self):
        self.data = None
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.results = {}
        self.kmeans = None
        print("🚀 Initializing Advanced Housing Market Analyzer")
        print("=" * 60)
    
    def load_and_preprocess_data(self):
        """Load California housing data with advanced preprocessing"""
        print("📊 Loading California Housing Dataset...")
        
        # Load dataset
        housing = fetch_california_housing()
        self.data = pd.DataFrame(housing.data, columns=housing.feature_names)
        self.data['Price'] = housing.target * 100000  # Convert to actual dollars
        
        print(f"✅ Loaded {len(self.data):,} housing records")
        print(f"💰 Price range: ${self.data['Price'].min():,.0f} - ${self.data['Price'].max():,.0f}")
        
        # Advanced outlier detection using IQR method
        self._remove_outliers()
        
        return self.data
    
    def _remove_outliers(self):
        """Remove statistical outliers for better model performance"""
        initial_count = len(self.data)
        
        # Remove outliers using IQR method for price
        Q1 = self.data['Price'].quantile(0.25)
        Q3 = self.data['Price'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        self.data = self.data[(self.data['Price'] >= lower_bound) & 
                             (self.data['Price'] <= upper_bound)]
        
        removed_count = initial_count - len(self.data)
        print(f"🧹 Removed {removed_count:,} outliers ({removed_count/initial_count*100:.1f}%)")
    
    def create_advanced_features(self):
        """Engineer sophisticated features using domain knowledge"""
        print("\n🔬 Creating Advanced Feature Engineering...")
        
        df = self.data.copy()
        
        # 1. Wealth indicators
        df['WealthIndex'] = df['MedInc'] * df['AveRooms'] / df['AveOccup']
        
        # 2. Space efficiency metrics
        df['SpaceEfficiency'] = df['AveRooms'] / df['AveBedrms']
        df['LivingSpaceRatio'] = (df['AveRooms'] - df['AveBedrms']) / df['AveRooms']
        
        # 3. Location desirability (multiple city centers)
        major_cities = {
            'SF': (37.7749, -122.4194),
            'LA': (34.0522, -118.2437),
            'SD': (32.7157, -117.1611),
            'SJ': (37.3382, -121.8863)
        }
        
        for city, (lat, lon) in major_cities.items():
            df[f'Distance_{city}'] = np.sqrt((df['Latitude'] - lat)**2 + 
                                           (df['Longitude'] - lon)**2)
        
        # 4. Market segments using clustering
        location_features = ['Latitude', 'Longitude', 'MedInc']
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        df['MarketSegment'] = self.kmeans.fit_predict(df[location_features])
        
        # 5. Interaction features
        df['Income_Age_Interaction'] = df['MedInc'] * df['HouseAge']
        df['Rooms_Population_Interaction'] = df['AveRooms'] * np.log(df['Population'] + 1)
        
        # 6. Derived ratios
        df['People_Per_Room'] = df['AveOccup'] / df['AveRooms']
        df['Bedroom_Density'] = df['AveBedrms'] * df['Population']
        
        # 7. Geographic zones
        df['Zone_North'] = (df['Latitude'] > 36.5).astype(int)
        df['Zone_Central'] = ((df['Latitude'] >= 34.5) & (df['Latitude'] <= 36.5)).astype(int)
        df['Zone_South'] = (df['Latitude'] < 34.5).astype(int)
        
        print(f"   ✅ Created {len(df.columns) - len(self.data.columns)} new features")
        print(f"   📈 Total features: {len(df.columns)}")
        
        self.data = df
        return df
    
    def perform_feature_selection(self, X, y, method='mutual_info', k=15):
        """Select best features using statistical methods"""
        print(f"\n🎯 Selecting Top {k} Features using {method}...")
        
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=k)
        else:
            selector = SelectKBest(score_func=f_regression, k=k)
        
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        print(f"   🏆 Selected features: {', '.join(selected_features)}")
        
        return X_selected, selected_features, selector
    
    def create_interactive_visualizations(self):
        """Create modern interactive visualizations using Plotly"""
        print("\n📊 Creating Interactive Market Visualizations...")
        os.makedirs('outputs', exist_ok=True)
        
        # 1. 3D Price Distribution
        fig1 = px.scatter_3d(self.data.sample(min(5000, len(self.data)), random_state=42), 
                            x='Longitude', y='Latitude', z='MedInc',
                            color='Price', size='AveRooms',
                            title='California Housing: 3D Market Overview',
                            color_continuous_scale='Viridis')
        try:
            fig1.show()
        except Exception:
            pass
        try:
            fig1.write_html('outputs/3d_market_overview.html')
        except Exception:
            pass
        
        # 2. Correlation Heatmap with Clustering
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        corr_matrix = self.data[numeric_cols].corr()
        
        fig2 = px.imshow(corr_matrix, 
                        title='Feature Correlation Matrix',
                        color_continuous_scale='RdBu',
                        aspect='auto')
        try:
            fig2.show()
        except Exception:
            pass
        try:
            fig2.write_html('outputs/feature_correlation_matrix.html')
        except Exception:
            pass
        
        # 3. Market Segment Analysis
        segment_stats = self.data.groupby('MarketSegment').agg({
            'Price': ['mean', 'median', 'std'],
            'MedInc': 'mean',
            'HouseAge': 'mean'
        }).round(2)
        
        print("\n🏘️ Market Segment Analysis:")
        print(segment_stats)
    
    def build_ensemble_models(self, X, y):
        """Build multiple advanced ML models for comparison"""
        print("\n🤖 Building Advanced ML Model Ensemble...")
        print("=" * 50)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=pd.qcut(y, q=5, duplicates='drop')
        )
        
        # Initialize scalers
        standard_scaler = StandardScaler()
        robust_scaler = RobustScaler()
        
        X_train_std = standard_scaler.fit_transform(X_train)
        X_test_std = standard_scaler.transform(X_test)
        X_train_robust = robust_scaler.fit_transform(X_train)
        X_test_robust = robust_scaler.transform(X_test)
        
        self.scalers = {
            'standard': standard_scaler,
            'robust': robust_scaler
        }
        
        # Define advanced models with optimized parameters
        models = {
            'Random_Forest': {
                'model': RandomForestRegressor(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=-1
                ),
                'X_train': X_train,
                'X_test': X_test,
                'scaler': None
            },
            'Gradient_Boosting': {
                'model': GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=6,
                    min_samples_split=10,
                    min_samples_leaf=4,
                    subsample=0.8,
                    random_state=42
                ),
                'X_train': X_train,
                'X_test': X_test,
                'scaler': None
            },
            'Extra_Trees': {
                'model': ExtraTreesRegressor(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                ),
                'X_train': X_train,
                'X_test': X_test,
                'scaler': None
            },
            'Neural_Network': {
                'model': MLPRegressor(
                    hidden_layer_sizes=(100, 50, 25),
                    activation='relu',
                    solver='adam',
                    alpha=0.01,
                    learning_rate='adaptive',
                    max_iter=1000,
                    random_state=42
                ),
                'X_train': X_train_std,
                'X_test': X_test_std,
                'scaler': 'standard'
            },
            'Support_Vector': {
                'model': SVR(
                    kernel='rbf',
                    C=100,
                    gamma='scale',
                    epsilon=0.1
                ),
                'X_train': X_train_robust,
                'X_test': X_test_robust,
                'scaler': 'robust'
            }
        }
        
        # Train models and collect results
        for name, config in models.items():
            print(f"\n🔄 Training {name.replace('_', ' ')} Model...")
            
            model = config['model']
            model.fit(config['X_train'], y_train)
            predictions = model.predict(config['X_test'])
            
            # Calculate comprehensive metrics
            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, predictions)
            mape = mean_absolute_percentage_error(y_test, predictions)
            
            # Cross-validation score (avoid leakage by fitting scaler inside pipeline)
            if config['scaler'] is None:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            elif config['scaler'] == 'standard':
                pipeline_model = make_pipeline(StandardScaler(), model)
                cv_scores = cross_val_score(pipeline_model, X_train, y_train, cv=5, scoring='r2')
            elif config['scaler'] == 'robust':
                pipeline_model = make_pipeline(RobustScaler(), model)
                cv_scores = cross_val_score(pipeline_model, X_train, y_train, cv=5, scoring='r2')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            self.models[name] = model
            self.results[name] = {
                'predictions': predictions,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mape': mape,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"   ✅ MAE: ${mae:,.0f}")
            print(f"   ✅ RMSE: ${rmse:,.0f}")
            print(f"   ✅ R²: {r2:.4f}")
            print(f"   ✅ MAPE: {mape:.2%}")
            print(f"   ✅ CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return X_test, y_test
    
    def compare_model_performance(self):
        """Create comprehensive model comparison"""
        print("\n🏆 MODEL PERFORMANCE COMPARISON")
        print("=" * 50)
        
        # Create comparison DataFrame
        comparison_data = []
        for name, results in self.results.items():
            comparison_data.append({
                'Model': name.replace('_', ' '),
                'MAE ($)': f"{results['mae']:,.0f}",
                'RMSE ($)': f"{results['rmse']:,.0f}",
                'R² Score': f"{results['r2']:.4f}",
                'MAPE (%)': f"{results['mape']:.2%}",
                'CV Score': f"{results['cv_mean']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        # Ensure numeric sorting for scores
        numeric_cols = ['MAE ($)', 'RMSE ($)', 'R² Score', 'MAPE (%)', 'CV Score']
        for col in numeric_cols:
            if col in comparison_df.columns:
                comparison_df[col] = (
                    comparison_df[col]
                    .astype(str)
                    .str.replace(',', '', regex=False)
                    .str.replace('%', '', regex=False)
                    .astype(float, errors='ignore')
                )
        comparison_df = comparison_df.sort_values('R² Score', ascending=False)

        # Pretty formatting for display
        display_df = comparison_df.copy()
        if 'MAE ($)' in display_df:
            display_df['MAE ($)'] = display_df['MAE ($)'].apply(lambda v: f"{int(round(v)):,}")
        if 'RMSE ($)' in display_df:
            display_df['RMSE ($)'] = display_df['RMSE ($)'].apply(lambda v: f"{int(round(v)):,}")
        if 'R² Score' in display_df:
            display_df['R² Score'] = display_df['R² Score'].apply(lambda v: f"{v:.4f}")
        if 'MAPE (%)' in display_df:
            display_df['MAPE (%)'] = display_df['MAPE (%)'].apply(lambda v: f"{v:.2f}%")
        if 'CV Score' in display_df:
            display_df['CV Score'] = display_df['CV Score'].apply(lambda v: f"{v:.4f}")

        print(display_df.to_string(index=False))
        
        # Find best model
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['r2'])
        print(f"\n🥇 Best Model: {best_model.replace('_', ' ')}")
        print(f"   R² Score: {self.results[best_model]['r2']:.4f}")
        print(f"   Average Error: ${self.results[best_model]['rmse']:,.0f}")
        
        return best_model
    
    def analyze_feature_importance(self, model_name='Random_Forest'):
        """Analyze feature importance using tree-based models"""
        print(f"\n🔍 Feature Importance Analysis ({model_name.replace('_', ' ')})")
        print("=" * 50)
        
        if model_name in self.models and hasattr(self.models[model_name], 'feature_importances_'):
            importances = self.models[model_name].feature_importances_
            feature_importance = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            print("🎯 Top 10 Most Important Features:")
            for rank, (_, row) in enumerate(feature_importance.head(10).iterrows(), start=1):
                print(f"   {rank:2d}. {row['Feature']}: {row['Importance']:.4f}")
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            top_features = feature_importance.head(15)
            plt.barh(range(len(top_features)), top_features['Importance'], 
                    color='steelblue', alpha=0.8)
            plt.yticks(range(len(top_features)), top_features['Feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Feature Importance - {model_name.replace("_", " ")} Model')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            try:
                os.makedirs('outputs', exist_ok=True)
                plt.savefig('outputs/feature_importance.png', dpi=150, bbox_inches='tight')
            except Exception:
                pass
            try:
                plt.show()
            except Exception:
                pass
    
    def create_prediction_interface(self, best_model):
        """Interactive prediction system for new properties"""
        print(f"\n🏡 PROPERTY PRICE PREDICTION SYSTEM")
        print("=" * 50)
        
        # Define realistic test scenarios
        test_scenarios = [
            {
                'name': 'Silicon Valley Tech Hub',
                'MedInc': 12.5, 'HouseAge': 10, 'AveRooms': 7.2, 'AveBedrms': 1.0,
                'Population': 3500, 'AveOccup': 2.5, 'Latitude': 37.4, 'Longitude': -122.1
            },
            {
                'name': 'Orange County Family Area',
                'MedInc': 7.8, 'HouseAge': 20, 'AveRooms': 6.5, 'AveBedrms': 1.3,
                'Population': 2800, 'AveOccup': 3.1, 'Latitude': 33.7, 'Longitude': -117.8
            },
            {
                'name': 'Sacramento Suburban',
                'MedInc': 5.2, 'HouseAge': 35, 'AveRooms': 5.8, 'AveBedrms': 1.2,
                'Population': 2200, 'AveOccup': 2.9, 'Latitude': 38.5, 'Longitude': -121.5
            },
            {
                'name': 'Central Valley Agricultural',
                'MedInc': 3.8, 'HouseAge': 28, 'AveRooms': 5.1, 'AveBedrms': 1.1,
                'Population': 1500, 'AveOccup': 3.4, 'Latitude': 36.8, 'Longitude': -119.8
            }
        ]
        
        for scenario in test_scenarios:
            # Create feature vector (need to engineer all features)
            input_data = pd.DataFrame([scenario])
            
            # Add engineered features
            input_data = self._engineer_features_for_prediction(input_data)
            
            # Select same features used in training
            input_features = input_data[self.feature_names]
            
            # Apply scaling if needed
            model_config = {
                'Random_Forest': None,
                'Gradient_Boosting': None,
                'Extra_Trees': None,
                'Neural_Network': 'standard',
                'Support_Vector': 'robust'
            }
            
            scaler_type = model_config.get(best_model)
            if scaler_type:
                input_features = self.scalers[scaler_type].transform(input_features)
            
            # Make prediction
            predicted_price = self.models[best_model].predict(input_features)[0]
            
            print(f"\n🏠 {scenario['name']}")
            print(f"   📍 Location: {scenario['Latitude']:.1f}°N, {abs(scenario['Longitude']):.1f}°W")
            print(f"   💰 Median Income: ${scenario['MedInc']*10000:,.0f}")
            print(f"   🏡 House Age: {scenario['HouseAge']} years")
            print(f"   📊 Predicted Price: ${predicted_price:,.0f}")
            
            # Add confidence interval (simplified)
            model_rmse = self.results[best_model]['rmse']
            lower_bound = predicted_price - model_rmse
            upper_bound = predicted_price + model_rmse
            print(f"   📈 Price Range: ${lower_bound:,.0f} - ${upper_bound:,.0f}")
    
    def _engineer_features_for_prediction(self, df):
        """Engineer features for new prediction data"""
        # Replicate the feature engineering process
        df['WealthIndex'] = df['MedInc'] * df['AveRooms'] / df['AveOccup']
        df['SpaceEfficiency'] = df['AveRooms'] / df['AveBedrms']
        df['LivingSpaceRatio'] = (df['AveRooms'] - df['AveBedrms']) / df['AveRooms']
        
        # Distance features
        major_cities = {
            'SF': (37.7749, -122.4194),
            'LA': (34.0522, -118.2437),
            'SD': (32.7157, -117.1611),
            'SJ': (37.3382, -121.8863)
        }
        
        for city, (lat, lon) in major_cities.items():
            df[f'Distance_{city}'] = np.sqrt((df['Latitude'] - lat)**2 + 
                                           (df['Longitude'] - lon)**2)
        
        # Market segment assignment (prefer trained clustering if available)
        try:
            if hasattr(self, 'kmeans') and self.kmeans is not None:
                df['MarketSegment'] = self.kmeans.predict(df[['Latitude', 'Longitude', 'MedInc']])
            else:
                df['MarketSegment'] = pd.cut(
                    df['MedInc'], bins=[0, 4, 6, 8, 15], labels=[0, 1, 2, 3]
                ).astype(int)
        except Exception:
            df['MarketSegment'] = pd.cut(
                df['MedInc'], bins=[0, 4, 6, 8, 15], labels=[0, 1, 2, 3]
            ).astype(int)
        
        # Interaction features
        df['Income_Age_Interaction'] = df['MedInc'] * df['HouseAge']
        df['Rooms_Population_Interaction'] = df['AveRooms'] * np.log(df['Population'] + 1)
        
        # Derived ratios
        df['People_Per_Room'] = df['AveOccup'] / df['AveRooms']
        df['Bedroom_Density'] = df['AveBedrms'] * df['Population']
        
        # Geographic zones
        df['Zone_North'] = (df['Latitude'] > 36.5).astype(int)
        df['Zone_Central'] = ((df['Latitude'] >= 34.5) & (df['Latitude'] <= 36.5)).astype(int)
        df['Zone_South'] = (df['Latitude'] < 34.5).astype(int)
        
        return df
    
    def create_advanced_visualizations(self, X_test, y_test, best_model):
        """Create publication-quality visualizations"""
        print("\n📊 Creating Advanced Performance Visualizations...")
        os.makedirs('outputs', exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Advanced Housing Price Prediction Analysis', fontsize=16, fontweight='bold')
        
        best_predictions = self.results[best_model]['predictions']
        
        # 1. Actual vs Predicted with density
        axes[0,0].scatter(y_test, best_predictions, alpha=0.6, s=20, color='navy')
        axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                      'r--', linewidth=2, label='Perfect Prediction')
        axes[0,0].set_xlabel('Actual Price ($)')
        axes[0,0].set_ylabel('Predicted Price ($)')
        axes[0,0].set_title(f'{best_model.replace("_", " ")} - Actual vs Predicted')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Residuals distribution
        residuals = y_test - best_predictions
        axes[0,1].hist(residuals, bins=50, alpha=0.7, color='darkgreen', density=True)
        axes[0,1].axvline(residuals.mean(), color='red', linestyle='--', 
                         label=f'Mean: ${residuals.mean():,.0f}')
        axes[0,1].set_xlabel('Residuals ($)')
        axes[0,1].set_ylabel('Density')
        axes[0,1].set_title('Prediction Error Distribution')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Model comparison
        models = list(self.results.keys())
        r2_scores = [self.results[model]['r2'] for model in models]
        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        
        bars = axes[1,0].bar(range(len(models)), r2_scores, color=colors, alpha=0.8)
        axes[1,0].set_xticks(range(len(models)))
        axes[1,0].set_xticklabels([m.replace('_', '\n') for m in models], rotation=0)
        axes[1,0].set_ylabel('R² Score')
        axes[1,0].set_title('Model Performance Comparison')
        axes[1,0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, score in zip(bars, r2_scores):
            axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                          f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Prediction intervals
        error_percentiles = np.percentile(np.abs(residuals), [50, 80, 95])
        sample_indices = np.random.choice(len(y_test), 100)
        sample_actual = y_test.iloc[sample_indices]
        sample_predicted = best_predictions[sample_indices]
        
        axes[1,1].scatter(range(100), sample_actual, alpha=0.7, s=30, color='blue', label='Actual')
        axes[1,1].scatter(range(100), sample_predicted, alpha=0.7, s=30, color='red', label='Predicted')
        axes[1,1].fill_between(range(100), 
                              sample_predicted - error_percentiles[1], 
                              sample_predicted + error_percentiles[1], 
                              alpha=0.2, color='gray', label='80% Prediction Interval')
        axes[1,1].set_xlabel('Sample Properties')
        axes[1,1].set_ylabel('Price ($)')
        axes[1,1].set_title('Prediction Confidence Intervals (Sample)')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        try:
            plt.savefig('outputs/model_performance.png', dpi=150, bbox_inches='tight')
        except Exception:
            pass
        try:
            plt.show()
        except Exception:
            pass
    
    def generate_market_insights(self):
        """Generate actionable market insights"""
        print("\n💡 CALIFORNIA HOUSING MARKET INSIGHTS")
        print("=" * 60)
        
        # Price distribution analysis
        price_stats = self.data['Price'].describe()
        
        print("📊 Market Statistics:")
        print(f"   • Average Home Price: ${price_stats['mean']:,.0f}")
        print(f"   • Median Home Price: ${price_stats['50%']:,.0f}")
        print(f"   • Standard Deviation: ${price_stats['std']:,.0f}")
        
        # Income correlation
        income_corr = self.data['MedInc'].corr(self.data['Price'])
        print(f"\n💰 Income-Price Relationship:")
        print(f"   • Correlation Coefficient: {income_corr:.3f}")
        print(f"   • Interpretation: {'Strong' if income_corr > 0.7 else 'Moderate'} positive correlation")
        
        # Geographic insights
        north_prices = self.data[self.data['Zone_North'] == 1]['Price'].mean()
        central_prices = self.data[self.data['Zone_Central'] == 1]['Price'].mean()
        south_prices = self.data[self.data['Zone_South'] == 1]['Price'].mean()
        
        print(f"\n🗺️ Regional Price Analysis:")
        print(f"   • Northern California: ${north_prices:,.0f}")
        print(f"   • Central California: ${central_prices:,.0f}")
        print(f"   • Southern California: ${south_prices:,.0f}")
        
        # Investment recommendations
        print(f"\n🎯 Investment Insights:")
        high_value_low_income = self.data[
            (self.data['Price'] > price_stats['75%']) & 
            (self.data['MedInc'] < self.data['MedInc'].median())
        ]
        
        if len(high_value_low_income) > 0:
            print(f"   • Found {len(high_value_low_income)} areas with high prices but lower incomes")
            print(f"   • These may indicate gentrification or investment opportunities")
        
        print("\n" + "=" * 60)
    
    def run_complete_analysis(self):
        """Execute complete housing market analysis pipeline"""
        print("🏠 CALIFORNIA HOUSING MARKET INTELLIGENCE SYSTEM")
        print("Advanced Multi-Model Prediction & Analysis Platform")
        print("=" * 70)
        
        # 1. Load and preprocess data
        self.load_and_preprocess_data()
        
        # 2. Feature engineering
        self.create_advanced_features()
        
        # 3. Prepare features for modeling
        feature_columns = [col for col in self.data.columns if col != 'Price']
        X = self.data[feature_columns]
        y = self.data['Price']
        
        # 4. Feature selection
        X_selected, self.feature_names, selector = self.perform_feature_selection(X, y, k=15)
        
        # 5. Create visualizations
        self.create_interactive_visualizations()
        
        # 6. Build and compare models
        X_test, y_test = self.build_ensemble_models(X_selected, y)
        
        # 7. Compare performance
        best_model = self.compare_model_performance()
        
        # 8. Feature importance analysis
        self.analyze_feature_importance(best_model)
        
        # 9. Create advanced visualizations
        self.create_advanced_visualizations(X_test, y_test, best_model)
        
        # 10. Property prediction system
        self.create_prediction_interface(best_model)
        
        # 11. Generate insights
        self.generate_market_insights()
        
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print(f"Best performing model: {best_model.replace('_', ' ')}")
        print(f"Achieved R² score: {self.results[best_model]['r2']:.4f}")
        print("=" * 70)

# Execute the complete analysis
if __name__ == "__main__":
    analyzer = HousingMarketAnalyzer()
    analyzer.run_complete_analysis()