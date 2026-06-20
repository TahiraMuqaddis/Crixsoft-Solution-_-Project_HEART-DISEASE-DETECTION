"""
CRIXSOFT SOLUTION - Machine Learning Internship
Project 2: Heart Disease Detection
Author: Your Name
Description: Uses ML algorithms to predict likelihood of heart disease from medical data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve, 
                             auc, accuracy_score, precision_score, recall_score, f1_score)
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATASET GENERATION (Medical Data)
# ============================================================================

def generate_heart_disease_data(samples=500, random_state=42):
    """
    Generate synthetic heart disease dataset
    Features:
    - Age: 29-77 years
    - Sex: 0=Female, 1=Male
    - Chest Pain Type: 1-4
    - Resting Blood Pressure: 80-200 mmHg
    - Cholesterol: 100-570 mg/dL
    - Fasting Blood Sugar: 0=<120 mg/dL, 1=>120 mg/dL
    - Resting ECG: 0-2
    - Max Heart Rate: 60-202 bpm
    - Exercise Induced Angina: 0=No, 1=Yes
    - Old Peak (ST depression): 0-6.2
    - ST Slope: 1-3
    - Target: 0=No disease, 1=Disease present
    """
    np.random.seed(random_state)
    
    n_features = 11
    X = np.zeros((samples, n_features))
    y = np.zeros(samples)
    
    # Feature generation with correlation to heart disease
    X[:, 0] = np.random.uniform(29, 77, samples)  # Age
    X[:, 1] = np.random.binomial(1, 0.65, samples)  # Sex
    X[:, 2] = np.random.randint(1, 5, samples)  # Chest Pain Type
    X[:, 3] = np.random.uniform(80, 200, samples)  # Resting BP
    X[:, 4] = np.random.uniform(100, 570, samples)  # Cholesterol
    X[:, 5] = np.random.binomial(1, 0.15, samples)  # Fasting BS
    X[:, 6] = np.random.randint(0, 3, samples)  # Resting ECG
    X[:, 7] = np.random.uniform(60, 202, samples)  # Max HR
    X[:, 8] = np.random.binomial(1, 0.35, samples)  # Exercise Angina
    X[:, 9] = np.random.uniform(0, 6.2, samples)  # Old Peak
    X[:, 10] = np.random.randint(1, 4, samples)  # ST Slope
    
    # Generate target based on risk factors
    risk_score = (
        (X[:, 0] > 55) * 0.3 +  # Age
        (X[:, 1] == 1) * 0.2 +  # Male
        (X[:, 3] > 140) * 0.25 +  # High BP
        (X[:, 4] > 240) * 0.2 +  # High Cholesterol
        (X[:, 8] == 1) * 0.25 +  # Exercise Angina
        (X[:, 9] > 2) * 0.25  # ST depression
    )
    
    y = (risk_score > 1.0).astype(int)
    
    feature_names = ['Age', 'Sex', 'Chest_Pain_Type', 'Resting_BP', 'Cholesterol',
                     'Fasting_BS', 'Resting_ECG', 'Max_Heart_Rate', 'Exercise_Angina',
                     'Old_Peak', 'ST_Slope']
    
    return pd.DataFrame(X, columns=feature_names), pd.Series(y, name='Heart_Disease')

# ============================================================================
# DATA ANALYSIS
# ============================================================================

def analyze_data(X, y):
    """Perform exploratory data analysis"""
    print("\n" + "=" * 70)
    print("DATASET ANALYSIS")
    print("=" * 70)
    
    print(f"\nDataset Shape: {X.shape}")
    print(f"Number of Samples: {X.shape[0]}")
    print(f"Number of Features: {X.shape[1]}")
    
    print(f"\nTarget Distribution:")
    print(f"  • No Disease: {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.1f}%)")
    print(f"  • Disease Present: {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.1f}%)")
    
    print(f"\nFeature Statistics:")
    print(X.describe().round(2))
    
    return True

# ============================================================================
# MODEL TRAINING AND EVALUATION
# ============================================================================

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train multiple models and compare performance"""
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42)
    }
    
    results = {}
    predictions = {}
    
    print("\n" + "=" * 70)
    print("MODEL TRAINING AND EVALUATION")
    print("=" * 70)
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        results[model_name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        predictions[model_name] = {
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"  ✓ Accuracy: {accuracy:.4f}")
        print(f"  ✓ Precision: {precision:.4f}")
        print(f"  ✓ Recall: {recall:.4f}")
        print(f"  ✓ F1-Score: {f1:.4f}")
        print(f"  ✓ Cross-Validation: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    return results, predictions

# ============================================================================
# RISK ASSESSMENT FOR NEW PATIENTS
# ============================================================================

def predict_risk_for_patient(model, scaler, patient_data):
    """Predict heart disease risk for a new patient"""
    patient_scaled = scaler.transform([patient_data])
    risk_probability = model.predict_proba(patient_scaled)[0][1]
    
    risk_level = "Low Risk" if risk_probability < 0.3 else \
                 "Moderate Risk" if risk_probability < 0.7 else "High Risk"
    
    return risk_probability, risk_level

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualizations(X, y, results, predictions):
    """Create comprehensive visualizations"""
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Feature correlation heatmap
    ax1 = plt.subplot(2, 3, 1)
    X_with_target = X.copy()
    X_with_target['Heart_Disease'] = y
    correlation = X_with_target.corr()
    sns.heatmap(correlation['Heart_Disease'].sort_values(ascending=False).iloc[1:].values.reshape(-1, 1),
                annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax1, cbar=False)
    ax1.set_ylabel('Features')
    ax1.set_xlabel('Correlation with Heart Disease')
    ax1.set_title('Feature Correlation with Disease')
    
    # 2. Disease distribution
    ax2 = plt.subplot(2, 3, 2)
    disease_counts = y.value_counts()
    colors = ['#2ecc71', '#e74c3c']
    ax2.bar(['No Disease', 'Disease Present'], disease_counts.values, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Number of Patients')
    ax2.set_title('Target Distribution')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Model comparison
    ax3 = plt.subplot(2, 3, 3)
    model_names = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in model_names]
    f1_scores = [results[m]['f1'] for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    ax3.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
    ax3.bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.8)
    ax3.set_ylabel('Score')
    ax3.set_title('Model Performance Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_names, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. ROC curves
    ax4 = plt.subplot(2, 3, 4)
    for model_name in list(results.keys())[:3]:  # Top 3 models
        y_pred_proba = predictions[model_name]['y_pred_proba']
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        ax4.plot(fpr, tpr, label=f'{model_name} (AUC={roc_auc:.3f})', linewidth=2)
    
    ax4.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    ax4.set_xlabel('False Positive Rate')
    ax4.set_ylabel('True Positive Rate')
    ax4.set_title('ROC Curves')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Best model confusion matrix
    best_model_name = max(results, key=lambda x: results[x]['f1'])
    y_pred_best = predictions[best_model_name]['y_pred']
    cm = confusion_matrix(y, y_pred_best)
    
    ax5 = plt.subplot(2, 3, 5)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5, 
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    ax5.set_ylabel('True Label')
    ax5.set_xlabel('Predicted Label')
    ax5.set_title(f'Confusion Matrix - {best_model_name}')
    
    # 6. Age distribution by disease status
    ax6 = plt.subplot(2, 3, 6)
    ax6.hist(X[y == 0]['Age'], bins=20, alpha=0.6, label='No Disease', color='green', edgecolor='black')
    ax6.hist(X[y == 1]['Age'], bins=20, alpha=0.6, label='Disease', color='red', edgecolor='black')
    ax6.set_xlabel('Age (years)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Age Distribution by Disease Status')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/Heart_Disease_Detection_Analysis.png', dpi=300, bbox_inches='tight')
    print("\n  ✓ Analysis visualization saved as 'Heart_Disease_Detection_Analysis.png'")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 70)
    print("CRIXSOFT SOLUTION - Heart Disease Detection")
    print("=" * 70)
    
    # Step 1: Generate data
    print("\nStep 1: Generating Medical Dataset...")
    X, y = generate_heart_disease_data(samples=500, random_state=42)
    print(f"  ✓ Generated dataset with {len(X)} patients and {X.shape[1]} features")
    
    # Step 2: Analyze data
    print("\nStep 2: Analyzing Dataset...")
    analyze_data(X, y)
    
    # Step 3: Prepare data
    print("\nStep 3: Preparing Data for Modeling...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"  ✓ Training set: {X_train_scaled.shape[0]} samples")
    print(f"  ✓ Testing set: {X_test_scaled.shape[0]} samples")
    print(f"  ✓ Data standardized using StandardScaler")
    
    # Step 4: Train models
    print("\nStep 4: Training Machine Learning Models...")
    results, predictions = train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Step 5: Best model details
    print("\n" + "=" * 70)
    print("BEST MODEL SUMMARY")
    print("=" * 70)
    best_model_name = max(results, key=lambda x: results[x]['f1'])
    best_results = results[best_model_name]
    
    print(f"\nBest Performing Model: {best_model_name}")
    print(f"  • Accuracy: {best_results['accuracy']:.4f}")
    print(f"  • Precision: {best_results['precision']:.4f}")
    print(f"  • Recall: {best_results['recall']:.4f}")
    print(f"  • F1-Score: {best_results['f1']:.4f}")
    
    best_model = best_results['model']
    y_pred_best = predictions[best_model_name]['y_pred']
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred_best, 
                                target_names=['No Disease', 'Disease Present']))
    
    # Step 6: Example predictions
    print("\n" + "=" * 70)
    print("SAMPLE PATIENT RISK PREDICTIONS")
    print("=" * 70)
    
    # Sample patients
    sample_patients = [
        {
            'name': 'Patient A (Low Risk)',
            'data': [35, 0, 1, 120, 200, 0, 0, 180, 0, 0.5, 1]
        },
        {
            'name': 'Patient B (Moderate Risk)',
            'data': [55, 1, 3, 140, 250, 1, 1, 140, 1, 1.5, 2]
        },
        {
            'name': 'Patient C (High Risk)',
            'data': [70, 1, 4, 160, 320, 1, 2, 100, 1, 3.0, 3]
        }
    ]
    
    for patient in sample_patients:
        risk_prob, risk_level = predict_risk_for_patient(best_model, scaler, patient['data'])
        print(f"\n{patient['name']}")
        print(f"  • Risk Probability: {risk_prob:.2%}")
        print(f"  • Risk Level: {risk_level}")
    
    # Step 7: Visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    print("\nStep 7: Creating Analysis Visualizations...")
    create_visualizations(X, y, results, predictions)
    
    print("\n" + "=" * 70)
    print("✓ Project 2 Completed Successfully!")
    print("=" * 70)
    
    return {
        'best_model': best_model,
        'scaler': scaler,
        'results': results,
        'best_model_name': best_model_name
    }

if __name__ == "__main__":
    results = main()
