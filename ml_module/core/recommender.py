"""
Intervention Recommender for SACOPOA ML Module
==============================================

Intelligent recommendation system for educational interventions
based on student performance patterns and risk assessment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.filterwarnings('ignore')


class InterventionRecommender:
    """
    Recommendation system for educational interventions and support strategies.
    
    Capabilities:
    - Personalized intervention recommendations
    - Study strategy suggestions
    - Resource allocation optimization
    - Success pattern matching
    """
    
    def __init__(self):
        """Initialize the intervention recommender."""
        self.intervention_types = {
            'tutoring': 'One-on-one tutoring sessions',
            'study_group': 'Collaborative study groups',
            'remedial_classes': 'Additional remedial classes',
            'counseling': 'Academic counseling and guidance',
            'peer_mentoring': 'Peer mentoring program',
            'extra_practice': 'Additional practice materials',
            'time_management': 'Time management training',
            'stress_management': 'Stress management workshops',
            'learning_strategies': 'Learning strategies workshop',
            'assessment_preparation': 'Assessment preparation sessions'
        }
        
        self.intervention_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.success_matcher = NearestNeighbors(n_neighbors=5, metric='euclidean')
        self.intervention_database = []
        self.is_trained = False
        
    def build_intervention_database(self, student_features: np.ndarray,
                                  outcomes: np.ndarray,
                                  interventions_applied: List[List[str]]) -> None:
        """
        Build database of interventions and their outcomes.
        
        Args:
            student_features: Feature matrix of students
            outcomes: Success outcomes (1 = improved, 0 = no improvement)
            interventions_applied: List of interventions for each student
        """
        self.intervention_database = []
        
        for i, (features, outcome, interventions) in enumerate(
            zip(student_features, outcomes, interventions_applied)
        ):
            record = {
                'student_id': i,
                'features': features,
                'outcome': outcome,
                'interventions': interventions,
                'success': outcome > 0.5  # Binary success indicator
            }
            self.intervention_database.append(record)
    
    def train_recommendation_model(self, student_features: np.ndarray,
                                 successful_interventions: np.ndarray) -> Dict[str, float]:
        """
        Train the intervention recommendation model.
        
        Args:
            student_features: Feature matrix
            successful_interventions: Binary matrix of successful interventions
            
        Returns:
            Training metrics
        """
        # Create training labels (most effective intervention type)
        y_train = np.argmax(successful_interventions, axis=1)
        
        # Train the model
        self.intervention_model.fit(student_features, y_train)
        
        # Fit success pattern matcher
        successful_students = student_features[np.any(successful_interventions, axis=1)]
        if len(successful_students) > 0:
            self.success_matcher.fit(successful_students)
        
        # Calculate training accuracy
        accuracy = self.intervention_model.score(student_features, y_train)
        
        self.is_trained = True
        
        return {
            'model_accuracy': accuracy,
            'intervention_types_count': successful_interventions.shape[1],
            'successful_cases': int(np.sum(np.any(successful_interventions, axis=1)))
        }
    
    def recommend_interventions(self, student_features: np.ndarray,
                              risk_levels: np.ndarray,
                              performance_scores: np.ndarray = None) -> List[Dict[str, Any]]:
        """
        Recommend interventions for students.
        
        Args:
            student_features: Feature matrix for students
            risk_levels: Risk assessment levels
            performance_scores: Optional performance scores
            
        Returns:
            List of intervention recommendations for each student
        """
        recommendations = []
        
        for i, (features, risk_level) in enumerate(zip(student_features, risk_levels)):
            # Base recommendations on risk level
            base_recommendations = self._get_base_recommendations(risk_level)
            
            # Personalized recommendations based on features
            if self.is_trained:
                personalized = self._get_personalized_recommendations(features)
                base_recommendations.extend(personalized)
            
            # Pattern-based recommendations
            pattern_recommendations = self._get_pattern_based_recommendations(features)
            
            # Success-based recommendations
            success_recommendations = self._get_success_based_recommendations(features)
            
            # Combine and rank recommendations
            all_recommendations = self._combine_recommendations([
                base_recommendations,
                pattern_recommendations,
                success_recommendations
            ])
            
            # Add timing and priority
            timed_recommendations = self._add_timing_and_priority(
                all_recommendations, risk_level, performance_scores[i] if performance_scores is not None else None
            )
            
            recommendations.append({
                'student_index': i,
                'risk_level': risk_level,
                'recommendations': timed_recommendations,
                'priority_interventions': self._get_priority_interventions(timed_recommendations),
                'estimated_impact': self._estimate_intervention_impact(features, timed_recommendations)
            })
        
        return recommendations
    
    def _get_base_recommendations(self, risk_level: int) -> List[Dict[str, Any]]:
        """Get base recommendations based on risk level."""
        base_recs = []
        
        if risk_level == 0:  # Low risk
            base_recs = [
                {'type': 'extra_practice', 'confidence': 0.7, 'urgency': 'low'},
                {'type': 'study_group', 'confidence': 0.6, 'urgency': 'low'}
            ]
        elif risk_level == 1:  # Medium risk
            base_recs = [
                {'type': 'tutoring', 'confidence': 0.8, 'urgency': 'medium'},
                {'type': 'study_group', 'confidence': 0.7, 'urgency': 'medium'},
                {'type': 'time_management', 'confidence': 0.6, 'urgency': 'medium'}
            ]
        elif risk_level == 2:  # High risk
            base_recs = [
                {'type': 'tutoring', 'confidence': 0.9, 'urgency': 'high'},
                {'type': 'remedial_classes', 'confidence': 0.8, 'urgency': 'high'},
                {'type': 'counseling', 'confidence': 0.7, 'urgency': 'high'},
                {'type': 'peer_mentoring', 'confidence': 0.6, 'urgency': 'medium'}
            ]
        else:  # Critical risk
            base_recs = [
                {'type': 'tutoring', 'confidence': 0.95, 'urgency': 'critical'},
                {'type': 'remedial_classes', 'confidence': 0.9, 'urgency': 'critical'},
                {'type': 'counseling', 'confidence': 0.85, 'urgency': 'critical'},
                {'type': 'stress_management', 'confidence': 0.7, 'urgency': 'high'},
                {'type': 'learning_strategies', 'confidence': 0.8, 'urgency': 'high'}
            ]
        
        return base_recs
    
    def _get_personalized_recommendations(self, features: np.ndarray) -> List[Dict[str, Any]]:
        """Get personalized recommendations based on ML model."""
        if not self.is_trained:
            return []
        
        # Predict best intervention type
        features_reshaped = features.reshape(1, -1)
        predicted_intervention = self.intervention_model.predict(features_reshaped)[0]
        intervention_probs = self.intervention_model.predict_proba(features_reshaped)[0]
        
        # Get top 3 recommendations
        top_indices = np.argsort(intervention_probs)[-3:][::-1]
        
        personalized_recs = []
        intervention_names = list(self.intervention_types.keys())
        
        for idx in top_indices:
            if idx < len(intervention_names):
                personalized_recs.append({
                    'type': intervention_names[idx],
                    'confidence': float(intervention_probs[idx]),
                    'urgency': 'medium',
                    'source': 'ml_model'
                })
        
        return personalized_recs
    
    def _get_pattern_based_recommendations(self, features: np.ndarray) -> List[Dict[str, Any]]:
        """Get recommendations based on performance patterns."""
        pattern_recs = []
        
        # Analyze feature patterns (simplified approach)
        # This would be enhanced with more sophisticated pattern analysis
        
        # Check consistency (assuming it's one of the features)
        if len(features) > 8:  # Assuming consistency is feature index 8
            consistency = features[8]
            if consistency < 0.3:
                pattern_recs.append({
                    'type': 'time_management',
                    'confidence': 0.8,
                    'urgency': 'medium',
                    'reason': 'Low consistency in performance'
                })
        
        # Check improvement trend (assuming it's feature index 7)
        if len(features) > 7:
            trend = features[7]
            if trend < -0.5:
                pattern_recs.append({
                    'type': 'counseling',
                    'confidence': 0.7,
                    'urgency': 'high',
                    'reason': 'Declining performance trend'
                })
            elif trend > 0.5:
                pattern_recs.append({
                    'type': 'peer_mentoring',
                    'confidence': 0.6,
                    'urgency': 'low',
                    'reason': 'Positive improvement trend'
                })
        
        return pattern_recs
    
    def _get_success_based_recommendations(self, features: np.ndarray) -> List[Dict[str, Any]]:
        """Get recommendations based on similar successful students."""
        if not hasattr(self.success_matcher, 'kneighbors') or len(self.intervention_database) == 0:
            return []
        
        try:
            # Find similar successful students
            features_reshaped = features.reshape(1, -1)
            distances, indices = self.success_matcher.kneighbors(features_reshaped)
            
            # Collect interventions from similar successful cases
            successful_interventions = {}
            for idx in indices[0]:
                if idx < len(self.intervention_database):
                    record = self.intervention_database[idx]
                    if record['success']:
                        for intervention in record['interventions']:
                            if intervention in successful_interventions:
                                successful_interventions[intervention] += 1
                            else:
                                successful_interventions[intervention] = 1
            
            # Convert to recommendations
            success_recs = []
            for intervention, count in successful_interventions.items():
                confidence = min(0.9, count / len(indices[0]))
                success_recs.append({
                    'type': intervention,
                    'confidence': confidence,
                    'urgency': 'medium',
                    'source': 'similar_success',
                    'supporting_cases': count
                })
            
            return success_recs
            
        except Exception:
            return []
    
    def _combine_recommendations(self, recommendation_lists: List[List[Dict]]) -> List[Dict[str, Any]]:
        """Combine recommendations from different sources."""
        combined = {}
        
        for rec_list in recommendation_lists:
            for rec in rec_list:
                intervention_type = rec['type']
                
                if intervention_type in combined:
                    # Average confidence scores and take highest urgency
                    existing = combined[intervention_type]
                    existing['confidence'] = (existing['confidence'] + rec['confidence']) / 2
                    
                    # Priority: critical > high > medium > low
                    urgency_priority = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
                    if urgency_priority.get(rec['urgency'], 0) > urgency_priority.get(existing['urgency'], 0):
                        existing['urgency'] = rec['urgency']
                    
                    # Combine sources
                    if 'sources' not in existing:
                        existing['sources'] = [existing.get('source', 'base')]
                    if 'source' in rec:
                        existing['sources'].append(rec['source'])
                else:
                    combined[intervention_type] = rec.copy()
        
        # Convert back to list and sort by confidence
        result = list(combined.values())
        result.sort(key=lambda x: x['confidence'], reverse=True)
        
        return result
    
    def _add_timing_and_priority(self, recommendations: List[Dict], 
                               risk_level: int, performance_score: float = None) -> List[Dict[str, Any]]:
        """Add timing and priority information to recommendations."""
        for rec in recommendations:
            # Set implementation timeline
            if rec['urgency'] == 'critical':
                rec['timeline'] = 'Immediate (within 1 week)'
                rec['priority'] = 1
            elif rec['urgency'] == 'high':
                rec['timeline'] = 'Short-term (within 2 weeks)'
                rec['priority'] = 2
            elif rec['urgency'] == 'medium':
                rec['timeline'] = 'Medium-term (within 1 month)'
                rec['priority'] = 3
            else:
                rec['timeline'] = 'Long-term (within semester)'
                rec['priority'] = 4
            
            # Add detailed description
            rec['description'] = self.intervention_types.get(rec['type'], 'Unknown intervention')
            
            # Add estimated duration
            rec['estimated_duration'] = self._estimate_intervention_duration(rec['type'])
            
            # Add resource requirements
            rec['resources_needed'] = self._estimate_resources_needed(rec['type'])
        
        return recommendations
    
    def _get_priority_interventions(self, recommendations: List[Dict]) -> List[Dict]:
        """Get highest priority interventions."""
        priority_recs = [rec for rec in recommendations if rec.get('priority', 4) <= 2]
        return sorted(priority_recs, key=lambda x: (x.get('priority', 4), -x.get('confidence', 0)))
    
    def _estimate_intervention_impact(self, features: np.ndarray, 
                                    recommendations: List[Dict]) -> Dict[str, float]:
        """Estimate potential impact of interventions."""
        # Simplified impact estimation
        # In practice, this would use historical data and more sophisticated modeling
        
        high_confidence_recs = [rec for rec in recommendations if rec.get('confidence', 0) > 0.7]
        
        estimated_impact = {
            'performance_improvement': len(high_confidence_recs) * 0.1,  # 10% per high-confidence intervention
            'risk_reduction': min(0.5, len(high_confidence_recs) * 0.15),  # Up to 50% risk reduction
            'success_probability': min(0.9, 0.3 + len(high_confidence_recs) * 0.2)  # Base 30% + 20% per intervention
        }
        
        return estimated_impact
    
    def _estimate_intervention_duration(self, intervention_type: str) -> str:
        """Estimate duration for different intervention types."""
        duration_map = {
            'tutoring': '4-8 weeks',
            'study_group': '6-12 weeks',
            'remedial_classes': '8-16 weeks',
            'counseling': '2-6 sessions',
            'peer_mentoring': '4-12 weeks',
            'extra_practice': 'Ongoing',
            'time_management': '2-4 weeks',
            'stress_management': '3-6 sessions',
            'learning_strategies': '2-4 weeks',
            'assessment_preparation': '2-4 weeks'
        }
        
        return duration_map.get(intervention_type, 'Variable')
    
    def _estimate_resources_needed(self, intervention_type: str) -> List[str]:
        """Estimate resources needed for interventions."""
        resource_map = {
            'tutoring': ['Qualified tutor', 'Meeting space', 'Learning materials'],
            'study_group': ['Group facilitator', 'Meeting space', 'Study materials'],
            'remedial_classes': ['Instructor', 'Classroom', 'Curriculum materials'],
            'counseling': ['Academic counselor', 'Private office'],
            'peer_mentoring': ['Peer mentor', 'Meeting coordination'],
            'extra_practice': ['Practice materials', 'Online platform access'],
            'time_management': ['Workshop facilitator', 'Training materials'],
            'stress_management': ['Counselor/therapist', 'Workshop materials'],
            'learning_strategies': ['Educational specialist', 'Strategy guides'],
            'assessment_preparation': ['Subject expert', 'Practice tests']
        }
        
        return resource_map.get(intervention_type, ['General resources'])
    
    def generate_intervention_report(self, recommendations: List[Dict], 
                                   student_ids: List[str] = None) -> Dict[str, Any]:
        """Generate comprehensive intervention report."""
        if student_ids is None:
            student_ids = [f"Student_{i}" for i in range(len(recommendations))]
        
        # Summary statistics
        total_students = len(recommendations)
        critical_cases = sum(1 for rec in recommendations if any(
            r.get('urgency') == 'critical' for r in rec['recommendations']
        ))
        high_priority_cases = sum(1 for rec in recommendations if any(
            r.get('priority', 4) <= 2 for r in rec['recommendations']
        ))
        
        # Intervention frequency
        intervention_counts = {}
        for rec in recommendations:
            for intervention in rec['recommendations']:
                intervention_type = intervention['type']
                intervention_counts[intervention_type] = intervention_counts.get(intervention_type, 0) + 1
        
        # Resource requirements
        total_resources = {}
        for rec in recommendations:
            for intervention in rec['recommendations']:
                resources = intervention.get('resources_needed', [])
                for resource in resources:
                    total_resources[resource] = total_resources.get(resource, 0) + 1
        
        return {
            'summary': {
                'total_students': total_students,
                'critical_cases': critical_cases,
                'high_priority_cases': high_priority_cases,
                'average_recommendations_per_student': np.mean([len(rec['recommendations']) for rec in recommendations])
            },
            'intervention_frequency': intervention_counts,
            'resource_requirements': total_resources,
            'student_details': [
                {
                    'student_id': student_ids[i] if i < len(student_ids) else f"Student_{i}",
                    'risk_level': rec['risk_level'],
                    'total_recommendations': len(rec['recommendations']),
                    'priority_interventions': len(rec['priority_interventions']),
                    'estimated_impact': rec['estimated_impact']
                }
                for i, rec in enumerate(recommendations)
            ]
        }