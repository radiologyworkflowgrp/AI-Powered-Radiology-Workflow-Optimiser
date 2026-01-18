#!/usr/bin/env python3
"""
Convert patient_priority.csv with triage column to ML training format
"""
import pandas as pd

# Read CSV
df = pd.read_csv('patient_priority.csv')

# Map triage colors to priority numbers
# red = 1 (high), yellow/orange = 2 (medium), empty/other = 3 (low)
def map_triage_to_priority(triage):
    if pd.isna(triage) or triage == '':
        return 3  # low priority for empty
    triage = str(triage).lower().strip()
    if 'red' in triage:
        return 1  # high priority
    elif 'yellow' in triage or 'orange' in triage:
        return 2  # medium priority
    else:
        return 3  # low priority

# Create symptoms text from available medical data
def create_symptoms_text(row):
    symptoms = []
    
    # Chest pain type
    if not pd.isna(row.get('chest pain type')):
        pain_types = {1: 'typical angina', 2: 'atypical angina', 3: 'non-anginal pain', 4: 'asymptomatic'}
        symptoms.append(f"chest pain: {pain_types.get(int(row['chest pain type']), 'unknown')}")
    
    # Heart disease and hypertension
    if row.get('heart_disease', 0) == 1:
        symptoms.append('heart disease')
    if row.get('hypertension', 0) == 1:
        symptoms.append('hypertension')
    
    # Exercise angina
    if row.get('exercise angina', 0) == 1:
        symptoms.append('exercise-induced angina')
    
    # Vital signs
    if not pd.isna(row.get('blood pressure')):
        bp = int(row['blood pressure'])
        if bp > 140:
            symptoms.append(f'high blood pressure ({bp} mmHg)')
        elif bp < 90:
            symptoms.append(f'low blood pressure ({bp} mmHg)')
    
    if not pd.isna(row.get('max heart rate')):
        hr = int(row['max heart rate'])
        if hr > 100:
            symptoms.append(f'elevated heart rate ({hr} bpm)')
        elif hr < 60:
            symptoms.append(f'low heart rate ({hr} bpm)')
    
    # Age and gender context
    age = int(row.get('age', 0))
    gender = 'male' if row.get('gender', 0) == 1 else 'female'
    
    if len(symptoms) == 0:
        symptoms.append('routine checkup')
    
    return f"{age} year old {gender}, {', '.join(symptoms)}"

# Create formatted dataframe
formatted_df = pd.DataFrame({
    'symptoms_text': df.apply(create_symptoms_text, axis=1),
    'label': df['triage'].apply(map_triage_to_priority)
})

# Remove any rows with invalid labels
formatted_df = formatted_df[formatted_df['label'].isin([1, 2, 3])]

# Save
formatted_df.to_csv('patient_priority_formatted.csv', index=False)

print(f"✅ Created patient_priority_formatted.csv")
print(f"   Total rows: {len(formatted_df)}")
print(f"\nPriority distribution:")
print(formatted_df['label'].value_counts().sort_index())
print(f"\nSample rows:")
print(formatted_df.head(10))
