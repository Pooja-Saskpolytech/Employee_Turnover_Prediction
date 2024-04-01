import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt

# Load the trained XGBoost model
model = joblib.load('xgboost_model.pkl')
# Load the feature names
feature_names = joblib.load('feature_names.pkl')

# Function to preprocess input data
def preprocess_input_data(df):
    label_encoder = LabelEncoder()
    df['department'] = label_encoder.fit_transform(df['department'])
    df['salary'] = label_encoder.fit_transform(df['salary'])
    return df


# Function to make predictions
def predict_employee_turnover(input_data):
    input_data = preprocess_input_data(input_data)
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[:, 1]  # Probability of turnover
    return prediction,probability

# Function to get top contributing features
def get_top_features(input_data):
    input_data = preprocess_input_data(input_data)
    importance = model.feature_importances_
    top_features_indices = importance.argsort()[::-1][:5]  # Get indices of top 5 features
    top_features = [feature_names[i] for i in top_features_indices]
    return top_features
# Function to provide actionable recommendations based on top features and probability
def get_actionable_recommendations(top_features, probability):
    recommendations = []
    
    if probability >= 0.8:
        recommendations.append("High risk of turnover. Recommend immediate intervention such as one-on-one meetings with the employee, offering additional support, or providing opportunities for advancement.")
        if 'avg_hrs_month' in top_features:
           recommendations.append("Monitor employee workload and consider workload balancing strategies.")
        if 'satisfaction' in top_features:
            recommendations.append("Investigate factors contributing to low satisfaction, such as work environment or job responsibilities.")
        if 'review' in top_features:
            recommendations.append("Conduct performance reviews and provide constructive feedback to improve employee performance.")
        if 'bonus' in top_features:
              recommendations.append("Consider offering performance-based bonuses to incentivize and motivate employees.")
        if 'promoted' in top_features:
            recommendations.append("Provide opportunities for career advancement and professional growth.")
    elif probability >= 0.6:
        recommendations.append("Moderate risk of turnover. Consider implementing retention strategies such as performance bonuses, career development programs, or flexible work arrangements.")
    elif probability >= 0.4:
        recommendations.append("Low to moderate risk of turnover. Monitor the employee's engagement and satisfaction closely. Offer recognition and opportunities for skill development.")
    else:
        recommendations.append("Low risk of turnover. Continue to support the employee's growth and development to maintain engagement and job satisfaction.")
    
    return recommendations

def categorize_risk(probability):
    if probability < 0.3:
        return "Low Risk", "far fa-smile", "green"  # Green color for low risk
    elif probability < 0.7:
        return "Moderate Risk", "far fa-meh", "orange"  # Orange color for moderate risk
    else:
        return "High Risk", "far fa-frown", "red"  # Red color for high risk

# Function to get feature importance scores for all features
def get_feature_importance(input_data):
    input_data = preprocess_input_data(input_data)
    importance = model.feature_importances_
    feature_importance = {feature_names[i]: importance[i] for i in range(len(feature_names))}
    return feature_importance    

# Streamlit app
def show_predict_page():
    st.title('Employee Turnover Prediction')
    st.sidebar.header('Retrain Model')
    uploaded_file = st.sidebar.file_uploader("Upload CSV file for Retraining", type=['csv'])
    if uploaded_file is not None:
        retrain_button = st.sidebar.button('Retrain Model')
        if retrain_button:
            input_data = pd.read_csv(uploaded_file)        
            # Encode labels and split data
            label_encoder = LabelEncoder()
            X = input_data.drop(columns=['turnover_Status'])
            X['department'] = label_encoder.fit_transform(X['department'])
            X['salary'] = label_encoder.fit_transform(X['salary'])
            y = label_encoder.fit_transform(input_data['turnover_Status'])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)        
            # Retrain model
            model_retrained = XGBClassifier()
            model_retrained.fit(X_train, y_train)
            joblib.dump(model_retrained, 'xgboost_model.pkl')
            st.sidebar.success('Model retrained successfully!')
    st.sidebar.header('Input Features')
    department = st.sidebar.selectbox('Department', ['IT', 'admin', 'engineering', 'finance', 'logistics', 'marketing', 'operations', 'retail', 'sales', 'support'])
    promoted = st.sidebar.selectbox('Promoted', [0, 1])
    review = st.sidebar.slider('Review', 0.0, 1.0, 0.5)
    projects = st.sidebar.slider('Projects', 1, 10, 5)
    salary = st.sidebar.selectbox('Salary', ['low', 'medium', 'high'])
    tenure = st.sidebar.slider('Tenure', 1, 10, 3)
    satisfaction = st.sidebar.slider('Satisfaction', 0.0, 1.0, 0.5)
    bonus = st.sidebar.selectbox('Bonus', [0, 1])
    avg_hrs_month = st.sidebar.slider('Average Hours per Month', 50, 300, 150)

    input_data = pd.DataFrame({
        'department': [department],
        'promoted': [promoted],
        'review': [review],
        'projects': [projects],
        'salary': [salary],
        'tenure': [tenure],
        'satisfaction': [satisfaction],
        'bonus': [bonus],
        'avg_hrs_month': [avg_hrs_month]
    })

   
    if st.button('Predict'):
        prediction, probability = predict_employee_turnover(input_data)
        risk_level, icon, color = categorize_risk(probability[0])
        top_features = get_top_features(input_data)
        all_feature_importances = get_feature_importance(input_data)

        if prediction[0] == 1:
            st.error(f'Employee is likely to leave with a probability of {probability[0]:.2f}')
        else:
            st.success(f'Employee is likely to stay with a probability of {1 - probability[0]:.2f}')
        st.markdown(f"**Risk Level:** {risk_level} <i class='{icon}' style='color:{color}'></i>", unsafe_allow_html=True)
        st.markdown('**Top Features Contributing to Turnover Prediction:**')
        for feature in top_features:
           feature_label = feature.replace('_', ' ').title()
           st.write(f"- {feature_label}")
        # Get actionable recommendations based on top features and probability
        st.markdown('**Actionable Recommendations:**')
        recommendations = get_actionable_recommendations(top_features, probability[0])
        for rec in recommendations:
            st.write(f"- {rec}")

     # Create a bar chart of all feature importances
        st.markdown('**Visualisation of importance of each feature**')
        fig, ax = plt.subplots()
        ax.barh(list(all_feature_importances.keys()), list(all_feature_importances.values()), color='skyblue')
        ax.set_xlabel('Importance Score')
        ax.set_ylabel('Feature')
        ax.set_title('Feature Importance Scores')
        st.pyplot(fig)
    
   