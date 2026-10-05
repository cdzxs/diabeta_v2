"""Presentation only: does not select a tier or change eligibility."""
def explain(assessment, hba1c, glucose):
    labs = []
    for name, value, unit, lower, upper, diagnostic in [
        ('HbA1c', hba1c, '%', 5.7, 6.4, 6.5),
        ('Fasting glucose', glucose, 'mg/dL', 100, 125, 126),
    ]:
        if value is None:
            finding = 'Not provided; cannot assess this test.'
        elif value <= 0:
            finding = 'Requires a valid measured value and clinical review.'
        elif value >= diagnostic:
            finding = 'Diabetes-range laboratory value; clinical confirmation is needed.'
        elif lower <= value <= upper:
            finding = 'Prediabetes-range laboratory value under the configured rules.'
        elif value < lower:
            finding = 'Below the prediabetes threshold for this test; does not rule out diabetes.'
        else:
            finding = 'Between the configured bands; discuss this value with a clinician.'
        labs.append(dict(test=name, value=value, unit=unit, finding=finding))
    rule = assessment['rule_triggered']
    tier = assessment['final_risk_level']
    reasons = {
        'rule_1_diagnostic_lab_override': 'Rule 1: a supplied laboratory value reached the diabetes-range threshold. This is a lab rule, not a discovery by the profile model.',
        'rule_2_model1b_high_probability': 'Rule 2: the profile model assigned at least 70% probability to its High class. This project rule takes priority over the remaining lab rules; it is not a diabetes diagnosis.',
        'rule_3_prediabetic_range_lab': 'Rule 3: a supplied laboratory value was in the configured prediabetes range, after the first two rules were checked.',
        'rule_4_model1b_low_high_fallback': 'Rule 4: none of the earlier rules applied. The profile model compared its High and Low probabilities, ignoring its Moderate probability; High wins a tie. This is a project fallback, not a diagnosis.',
    }
    if tier == 'High':
        action = ('Arrange clinical review to confirm the laboratory finding and discuss next steps.'
                  if rule.startswith('rule_1') else
                  'Discuss the profile-based concern and the available or missing lab results with a clinician.')
        action += ' Under the current project policy, every final High result blocks Stage 3.'
    elif tier == 'Moderate':
        action = 'Discuss your lab results, risk factors and a prevention plan with a healthcare professional.'
    else:
        action = 'Continue prevention and follow clinical advice on repeat screening. Low does not exclude diabetes.'
    return dict(
        lab_assessment=labs,
        profile_screening=dict(high_probability=assessment['model1b_probability'].get('High', 0),
            explanation='Stage 1 model probability for its High class, based on eight profile predictors. HbA1c and fasting glucose are not inputs to this score. It is neither a diagnosis nor an approximately three-year risk estimate.'),
        final_recommendation=dict(tier=tier, rule=rule, explanation=reasons[rule], next_step=action))
