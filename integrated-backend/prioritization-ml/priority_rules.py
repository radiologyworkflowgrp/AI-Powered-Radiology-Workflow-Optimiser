#!/usr/bin/env python3
"""
Simple rule-based priority calculator for symptoms
This provides better prioritization than the untrained ML model
"""

def calculate_priority_from_symptoms(symptoms_text):
    """
    Calculate priority based on symptom keywords
    Returns: priority (1=high, 2=medium, 3=low)
    """
    if not symptoms_text:
        return 3  # Default to low priority if no symptoms
    
    symptoms_lower = symptoms_text.lower()
    
    # High priority symptoms (Priority 1)
    high_priority_keywords = [
        'chest pain', 'heart attack', 'stroke', 'seizure',
        'unconscious', 'not breathing', 'severe bleeding',
        'head injury', 'severe burn', 'difficulty breathing',
        'choking', 'severe allergic reaction', 'anaphylaxis',
        'severe trauma', 'gunshot', 'stab wound', 'cardiac arrest',
        'respiratory distress', 'severe chest pain', 'crushing chest pain'
    ]
    
    # Medium priority symptoms (Priority 2)
    medium_priority_keywords = [
        'fever', 'vomiting', 'diarrhea', 'abdominal pain',
        'headache', 'dizziness', 'nausea', 'cough',
        'shortness of breath', 'back pain', 'joint pain',
        'rash', 'infection', 'wound', 'fracture',
        'sprain', 'moderate pain', 'bleeding'
    ]
    
    # Check for high priority
    for keyword in high_priority_keywords:
        if keyword in symptoms_lower:
            return 1
    
    # Check for medium priority
    for keyword in medium_priority_keywords:
        if keyword in symptoms_lower:
            return 2
    
    # Default to low priority
    return 3

if __name__ == "__main__":
    # Test cases
    test_symptoms = [
        "chest pain and difficulty breathing",
        "fever and cough",
        "minor headache",
        "severe bleeding from head injury",
        "stomach ache"
    ]
    
    for symptom in test_symptoms:
        priority = calculate_priority_from_symptoms(symptom)
        print(f"Symptoms: '{symptom}' -> Priority: {priority}")
