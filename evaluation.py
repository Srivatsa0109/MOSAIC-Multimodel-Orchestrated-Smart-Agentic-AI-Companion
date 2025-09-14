import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
import json
from datetime import datetime
import os
from typing import List, Dict, Any

class TravelAgentEvaluator:
    def __init__(self):
        self.metrics = {
            'response_time': [],
            'query_complexity': [],
            'success_rate': [],
            'price_accuracy': [],
            'information_completeness': [],
            'user_satisfaction': [],
            'query_diversity': [],
            'recommendation_relevance': [],
            'error_handling': [],
            'context_understanding': [],
            'personalization_score': [],
            'response_consistency': [],
            'language_quality': [],
            'booking_conversion': [],
            'cost_optimization': [],
            'classification_metrics': {}  # New field for classification metrics
        }
        
    def calculate_metrics(self, test_data):
        """Calculate various evaluation metrics"""
        results = {
            'response_times': [],
            'success_rates': [],
            'price_accuracies': [],
            'completeness_scores': [],
            'satisfaction_scores': [],
            'query_diversity_scores': [],
            'recommendation_scores': [],
            'error_handling_scores': [],
            'context_scores': [],
            'personalization_scores': [],
            'consistency_scores': [],
            'language_scores': [],
            'conversion_rates': [],
            'cost_optimization_scores': [],
            'classification_metrics': {}  # New field for classification metrics
        }
        
        # Generate simulated ground truth and predictions for classification metrics
        y_true = np.random.randint(0, 2, size=100)  # Ground truth
        y_pred = np.random.randint(0, 2, size=100)  # Predictions
        
        # Calculate classification metrics
        results['classification_metrics'] = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }
        
        # Generate classification report
        results['classification_metrics']['report'] = classification_report(y_true, y_pred, output_dict=True)
        
        for query in test_data:
            # Response time (simulated)
            response_time = np.random.normal(2.5, 0.5)  # Mean 2.5s, std 0.5s
            results['response_times'].append(response_time)
            
            # Success rate (simulated)
            success = np.random.random() > 0.1  # 90% success rate
            results['success_rates'].append(success)
            
            # Price accuracy (simulated)
            price_accuracy = np.random.normal(0.95, 0.03)  # 95% accuracy with 3% std
            results['price_accuracies'].append(price_accuracy)
            
            # Information completeness (simulated)
            completeness = np.random.normal(0.85, 0.05)  # 85% completeness
            results['completeness_scores'].append(completeness)
            
            # User satisfaction (simulated)
            satisfaction = np.random.normal(4.2, 0.3)  # 4.2/5 rating
            results['satisfaction_scores'].append(satisfaction)
            
            # Query Diversity: Measures variety in types of queries handled
            query_diversity = np.random.normal(0.8, 0.1)
            results['query_diversity_scores'].append(query_diversity)
            
            # Recommendation Relevance: How well recommendations match user preferences
            recommendation_score = np.random.normal(0.88, 0.05)
            results['recommendation_scores'].append(recommendation_score)
            
            # Error Handling: How well the system handles edge cases and errors
            error_handling = np.random.normal(0.92, 0.04)
            results['error_handling_scores'].append(error_handling)
            
            # Context Understanding: How well the system maintains context
            context_score = np.random.normal(0.85, 0.06)
            results['context_scores'].append(context_score)
            
            # Personalization: How well recommendations are personalized
            personalization = np.random.normal(0.82, 0.07)
            results['personalization_scores'].append(personalization)
            
            # Response Consistency: Consistency in responses across similar queries
            consistency = np.random.normal(0.9, 0.04)
            results['consistency_scores'].append(consistency)
            
            # Language Quality: Quality of natural language responses
            language_quality = np.random.normal(0.87, 0.05)
            results['language_scores'].append(language_quality)
            
            # Booking Conversion: Rate of successful bookings
            conversion = np.random.normal(0.75, 0.08)
            results['conversion_rates'].append(conversion)
            
            # Cost Optimization: How well the system optimizes for cost
            cost_optimization = np.random.normal(0.88, 0.06)
            results['cost_optimization_scores'].append(cost_optimization)
        
        return results
    
    def create_visualizations(self, results):
        """Create various visualizations for the evaluation metrics"""
        # Create output directory if it doesn't exist
        os.makedirs('evaluation_results', exist_ok=True)
        
        # 1. Response Time Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(results['response_times'], kde=True)
        plt.title('Response Time Distribution')
        plt.xlabel('Response Time (seconds)')
        plt.ylabel('Frequency')
        plt.savefig('evaluation_results/response_time_dist.png')
        plt.close()
        
        # 2. Success Rate Pie Chart
        plt.figure(figsize=(8, 8))
        success_rate = sum(results['success_rates']) / len(results['success_rates'])
        plt.pie([success_rate, 1-success_rate], 
                labels=['Successful', 'Failed'],
                autopct='%1.1f%%',
                colors=['#2ecc71', '#e74c3c'])
        plt.title('Query Success Rate')
        plt.savefig('evaluation_results/success_rate.png')
        plt.close()
        
        # 3. Price Accuracy Box Plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(y=results['price_accuracies'])
        plt.title('Price Accuracy Distribution')
        plt.ylabel('Accuracy Score')
        plt.savefig('evaluation_results/price_accuracy.png')
        plt.close()
        
        # 4. Information Completeness Heatmap
        plt.figure(figsize=(12, 8))
        completeness_matrix = np.array(results['completeness_scores']).reshape(-1, 1)
        sns.heatmap(completeness_matrix, 
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Completeness Score'})
        plt.title('Information Completeness Heatmap')
        plt.savefig('evaluation_results/completeness_heatmap.png')
        plt.close()
        
        # 5. User Satisfaction Trend
        plt.figure(figsize=(12, 6))
        plt.plot(results['satisfaction_scores'], marker='o')
        plt.title('User Satisfaction Trend')
        plt.xlabel('Query Number')
        plt.ylabel('Satisfaction Score (out of 5)')
        plt.grid(True)
        plt.savefig('evaluation_results/satisfaction_trend.png')
        plt.close()
        
        # 6. Combined Metrics Dashboard
        plt.figure(figsize=(15, 10))
        metrics = ['Response Time', 'Success Rate', 'Price Accuracy', 
                  'Completeness', 'Satisfaction']
        values = [
            np.mean(results['response_times']),
            sum(results['success_rates']) / len(results['success_rates']),
            np.mean(results['price_accuracies']),
            np.mean(results['completeness_scores']),
            np.mean(results['satisfaction_scores']) / 5  # Normalize to 0-1
        ]
        
        plt.bar(metrics, values)
        plt.title('Overall Performance Metrics')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig('evaluation_results/performance_dashboard.png')
        plt.close()
        
        # New visualizations
        self._create_radar_chart(results)
        self._create_correlation_heatmap(results)
        self._create_metric_distributions(results)
        self._create_performance_matrix(results)
        
        # New classification metrics visualizations
        self._create_classification_metrics_plot(results)
        self._create_confusion_matrix_plot(results)
        
        # Save metrics to JSON
        self._save_metrics_summary(results)
    
    def _create_radar_chart(self, results):
        """Create a radar chart for key performance metrics"""
        metrics = ['Success Rate', 'Price Accuracy', 'Completeness', 
                  'Personalization', 'Context', 'Consistency']
        values = [
            np.mean(results['success_rates']),
            np.mean(results['price_accuracies']),
            np.mean(results['completeness_scores']),
            np.mean(results['personalization_scores']),
            np.mean(results['context_scores']),
            np.mean(results['consistency_scores'])
        ]
        
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
        values = np.concatenate((values, [values[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, values)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_title('Performance Metrics Radar Chart')
        plt.savefig('evaluation_results/performance_radar.png')
        plt.close()
    
    def _create_correlation_heatmap(self, results):
        """Create a correlation heatmap between different metrics"""
        metrics_data = pd.DataFrame({
            'Response Time': results['response_times'],
            'Success Rate': results['success_rates'],
            'Price Accuracy': results['price_accuracies'],
            'Completeness': results['completeness_scores'],
            'Satisfaction': results['satisfaction_scores'],
            'Personalization': results['personalization_scores'],
            'Context': results['context_scores'],
            'Consistency': results['consistency_scores']
        })
        
        correlation = metrics_data.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
        plt.title('Metrics Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('evaluation_results/correlation_heatmap.png')
        plt.close()
    
    def _create_metric_distributions(self, results):
        """Create distribution plots for key metrics"""
        metrics = {
            'Response Time': results['response_times'],
            'Price Accuracy': results['price_accuracies'],
            'Personalization': results['personalization_scores'],
            'Context Understanding': results['context_scores']
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for idx, (metric_name, values) in enumerate(metrics.items()):
            sns.histplot(values, kde=True, ax=axes[idx])
            axes[idx].set_title(f'{metric_name} Distribution')
            axes[idx].set_xlabel(metric_name)
            axes[idx].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('evaluation_results/metric_distributions.png')
        plt.close()
    
    def _create_performance_matrix(self, results):
        """Create a performance matrix visualization"""
        metrics = ['Success Rate', 'Price Accuracy', 'Completeness', 
                  'Personalization', 'Context', 'Consistency', 'Conversion']
        values = [
            np.mean(results['success_rates']),
            np.mean(results['price_accuracies']),
            np.mean(results['completeness_scores']),
            np.mean(results['personalization_scores']),
            np.mean(results['context_scores']),
            np.mean(results['consistency_scores']),
            np.mean(results['conversion_rates'])
        ]
        
        plt.figure(figsize=(12, 8))
        plt.barh(metrics, values)
        plt.title('Performance Matrix')
        plt.xlabel('Score')
        plt.xlim(0, 1)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('evaluation_results/performance_matrix.png')
        plt.close()
    
    def _create_classification_metrics_plot(self, results):
        """Create a bar plot for classification metrics"""
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        values = [
            results['classification_metrics']['accuracy'],
            results['classification_metrics']['precision'],
            results['classification_metrics']['recall'],
            results['classification_metrics']['f1']
        ]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, values)
        plt.title('Classification Metrics')
        plt.ylim(0, 1)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('evaluation_results/classification_metrics.png')
        plt.close()
    
    def _create_confusion_matrix_plot(self, results):
        """Create a confusion matrix visualization"""
        # Generate simulated confusion matrix
        cm = np.array([[45, 5], [8, 42]])  # Example values
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Predicted 0', 'Predicted 1'],
                   yticklabels=['Actual 0', 'Actual 1'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('evaluation_results/confusion_matrix.png')
        plt.close()
    
    def _save_metrics_summary(self, results):
        """Save detailed metrics summary to JSON"""
        metrics_summary = {
            'average_response_time': np.mean(results['response_times']),
            'success_rate': np.mean(results['success_rates']),
            'average_price_accuracy': np.mean(results['price_accuracies']),
            'average_completeness': np.mean(results['completeness_scores']),
            'average_satisfaction': np.mean(results['satisfaction_scores']),
            'average_personalization': np.mean(results['personalization_scores']),
            'average_context_understanding': np.mean(results['context_scores']),
            'average_consistency': np.mean(results['consistency_scores']),
            'average_conversion_rate': np.mean(results['conversion_rates']),
            'average_cost_optimization': np.mean(results['cost_optimization_scores']),
            'classification_metrics': results['classification_metrics'],
            'timestamp': datetime.now().isoformat()
        }
        
        with open('evaluation_results/metrics_summary.json', 'w') as f:
            json.dump(metrics_summary, f, indent=4)

def main():
    # Create sample test data
    test_data = [f"Query {i}" for i in range(100)]
    
    # Initialize evaluator
    evaluator = TravelAgentEvaluator()
    
    # Calculate metrics
    results = evaluator.calculate_metrics(test_data)
    
    # Create visualizations
    evaluator.create_visualizations(results)
    
    print("Evaluation completed! Check the 'evaluation_results' directory for visualizations and metrics.")

if __name__ == "__main__":
    main() 