import json
from api_llm_scoring import LLMScoring

llm_scoring = LLMScoring('upb-nlp/qwen3_4b_scoring_all_tasks_with_se_improved', feedback_model_name='Qwen/Qwen3-4B-Instruct-2507')

RBC_CONTEXT = "Red blood cells have the vital role of carrying oxygen to all of the cells in the body.  They also pick up waste carbon dioxide for removal.  These cells are the most numerous of the blood cells.  The disk shape of red blood cells results in a large surface area, which enables them to be efficient at gas diffusion.\nRed blood cells contain a large, complex protein called hemoglobin.  Hemoglobin binds to the oxygen and carbon dioxide that the red blood cells transport.  Each red blood cell contains about 250 million hemoglobin molecules, each carrying four molecules of oxygen.  Hemoglobin also contains iron, which gives blood its red color.  Molecular oxygen can also be transported by another route, dissolved in blood plasma. However, oxygen is poorly soluble in water, so only about 1.5% is carried in dissolved form.  Therefore, most oxygen is carried by hemoglobin.\nRed blood cells lack a nucleus and the organelles found in other cells.  Therefore, these cells cannot reproduce or repair themselves.  Red blood cells live for about three or four months before being broken down in the spleen.  Iron from the broken-down cells is returned to the bone marrow to be recycled into new hemoglobin.\nSometimes blood does not transport enough oxygen, resulting in a condition called anemia. This makes a person feel tired and weak.  Anemia can result from too little iron in the diet, loss of blood due to injury or menstruation, or various medical conditions.  One type of anemia, called sickle-cell disease, is characterized by red blood cells that are sickle-shaped instead of disk-shaped.  The shape of the cells causes them to clog blood vessels, preventing oxygen from reaching muscles and other tissues."

HD_CONTEXT = "The heart is the hardest-working organ in the living body. Any disorder that terminates the body's blood supply is a threat to life. More people are killed every year in the U.S. by heart disease than by any other disease.  A congenital disease is one with which a person is born. Most babies are born with perfect hearts, but something can go wrong for approximately one in 200 cases. Sometimes a valve develops the incorrect shape causing it to be too tight or fail to close properly. Sometimes a gap is left in the septal wall between the two sides of the heart. When a baby's heart is badly formed, it cannot work efficiently.  The baby's blood does not receive enough oxygen and cannot eliminate carbon dioxide through the lungs. The blood becomes purplish, and the baby's skin looks blue. The baby is in danger of suffocating. Diseases also cause the heart to form improperly. For example, the disease called rheumatic fever follows a sore throat caused by bacteria called streptococci. The tissues of the heart become inflamed and, if badly affected, can cause it to stop. Usually the heart recovers, but the heart valves are left with scars. Years later, they may fail to work properly and cause the heart to stop. The most common heart problem is a heart attack, or coronary thrombosis, which is caused when a coronary artery becomes blocked. The blood vessels that extend across the heart and supply it with blood are called the coronary arteries. They give the heart the oxygen it needs to carry on working. The blockage of a coronary artery is usually caused by a thrombus, or blood clot. Whether heart disease is congenital, caused by other diseases, or the result of a blood clot, it is a very serious problem that requires medical attention."

# ---------------------------------------------------------------------------
# Define all scenarios
# ---------------------------------------------------------------------------

scenarios = []

# =================== PARAPHRASING ===================

# 1. Good paraphrase with syntactic change -- expect no retry
scenarios.append({
    'name': 'paraphrasing_good_syntactic_change',
    'task': 'paraphrasing',
    'is_retry': False,
    'data': {
        'target_sentence': "One of the most harmful air pollutants is acid rain, a mixture of acid and water that falls to earth.",
        'student_response': "A combination of acid and water that falls upon the ground is a harmful pollutant called acid rain.",
    },
})

# 2. Decent paraphrase -- expect no retry
scenarios.append({
    'name': 'paraphrasing_decent',
    'task': 'paraphrasing',
    'is_retry': False,
    'data': {
        'target_sentence': "Sometimes blood does not transport enough oxygen, resulting in a condition called anemia.",
        'student_response': "Anemia is caused when the blood doesn't transport enough oxygen.",
    },
})

# 3. Garbage input -- expect retry with paraphrase
scenarios.append({
    'name': 'paraphrasing_garbage',
    'task': 'paraphrasing',
    'is_retry': False,
    'data': {
        'target_sentence': "Red blood cells have the vital role of carrying oxygen to all of the cells in the body.",
        'student_response': ",m.",
    },
})

# 4. Garbage input as retry -- should NOT get try-again prompt
scenarios.append({
    'name': 'paraphrasing_garbage_retry',
    'task': 'paraphrasing',
    'is_retry': True,
    'data': {
        'target_sentence': "Red blood cells have the vital role of carrying oxygen to all of the cells in the body.",
        'student_response': "asdf asdf",
    },
})

# 5. Very poor attempt, incomplete meaning -- expect retry with paraphrase
scenarios.append({
    'name': 'paraphrasing_very_poor',
    'task': 'paraphrasing',
    'is_retry': False,
    'data': {
        'target_sentence': "Over two thirds of heat generated by a resting human is created by organs of the thoracic and abdominal cavities and the brain.",
        'student_response': "2/3 OF HUMANS REST.",
    },
})

# 6. Very poor attempt as retry -- should NOT get try-again prompt
scenarios.append({
    'name': 'paraphrasing_very_poor_retry',
    'task': 'paraphrasing',
    'is_retry': True,
    'data': {
        'target_sentence': "Over two thirds of heat generated by a resting human is created by organs of the thoracic and abdominal cavities and the brain.",
        'student_response': "Humans rest a lot and produce heat.",
    },
})

# 7. Frozen / copy-paste -- expect retry with paraphrase
scenarios.append({
    'name': 'paraphrasing_frozen_copypaste',
    'task': 'paraphrasing',
    'is_retry': False,
    'data': {
        'target_sentence': "The disk shape of red blood cells results in a large surface area, which enables them to be efficient at gas diffusion.",
        'student_response': "The disk shape of red blood cells results in a large surface area, which enables them to be efficient at gas diffusion.",
    },
})

# 8. Irrelevant response -- expect retry with paraphrase
scenarios.append({
    'name': 'paraphrasing_irrelevant',
    'task': 'paraphrasing',
    'is_retry': False,
    'data': {
        'target_sentence': "Hemoglobin binds to the oxygen and carbon dioxide that the red blood cells transport.",
        'student_response': "I had pizza for lunch and it was delicious.",
    },
})

# 9. Partial paraphrase, some frozen expressions -- might trigger retry
scenarios.append({
    'name': 'paraphrasing_partial_frozen',
    'task': 'paraphrasing',
    'is_retry': False,
    'data': {
        'target_sentence': "Red blood cells lack a nucleus and the organelles found in other cells.",
        'student_response': "Red blood cells lack a nucleus and the organelles that other cells have.",
    },
})

# 10. Good paraphrase with meaning preserved -- expect no retry
scenarios.append({
    'name': 'paraphrasing_good_meaning_preserved',
    'task': 'paraphrasing',
    'is_retry': False,
    'data': {
        'target_sentence': "Each red blood cell contains about 250 million hemoglobin molecules, each carrying four molecules of oxygen.",
        'student_response': "A single red blood cell holds roughly 250 million hemoglobin molecules, and every one of them transports four oxygen molecules.",
    },
})

# 11. Random characters -- expect retry with paraphrase
scenarios.append({
    'name': 'paraphrasing_random_chars',
    'task': 'paraphrasing',
    'is_retry': False,
    'data': {
        'target_sentence': "Iron from the broken-down cells is returned to the bone marrow to be recycled into new hemoglobin.",
        'student_response': "!!!???@@@###",
    },
})

# 12. Random characters as retry -- should NOT get try-again prompt
scenarios.append({
    'name': 'paraphrasing_random_chars_retry',
    'task': 'paraphrasing',
    'is_retry': True,
    'data': {
        'target_sentence': "Iron from the broken-down cells is returned to the bone marrow to be recycled into new hemoglobin.",
        'student_response': "zzzz zzz zz",
    },
})

# 13. Single word answer -- expect retry
scenarios.append({
    'name': 'paraphrasing_single_word',
    'task': 'paraphrasing',
    'is_retry': False,
    'data': {
        'target_sentence': "Anemia can result from too little iron in the diet, loss of blood due to injury or menstruation, or various medical conditions.",
        'student_response': "anemia",
    },
})

# =================== SELF-EXPLANATION ===================

# 14. Excellent self-explanation with bridging -- expect no retry
scenarios.append({
    'name': 'selfexplanation_excellent_bridging',
    'task': 'selfexplanation',
    'is_retry': False,
    'data': {
        'target_sentence': "The shape of the cells causes them to clog blood vessels, preventing oxygen from reaching muscles and other tissues.",
        'context': RBC_CONTEXT,
        'student_response': "Blood vessels are naturally shaped to transport the disk shaped red blood cell, if the blood cell changes shape it makes sense to say how it will clog the vessels considering the vessels are already shaped for disk shaped blood cells.",
    },
})

# 15. Good self-explanation with elaboration -- expect no retry
scenarios.append({
    'name': 'selfexplanation_good_elaboration',
    'task': 'selfexplanation',
    'is_retry': False,
    'data': {
        'target_sentence': "Sometimes blood does not transport enough oxygen, resulting in a condition called anemia.",
        'context': RBC_CONTEXT,
        'student_response': "You develop a condition because you did not have enough oxygen transported, which could mean your red blood cells aren't functioning right.",
    },
})

# 16. Basic paraphrase only, no deeper processing -- borderline
scenarios.append({
    'name': 'selfexplanation_basic_paraphrase',
    'task': 'selfexplanation',
    'is_retry': False,
    'data': {
        'target_sentence': "A congenital disease is one with which a person is born.",
        'context': HD_CONTEXT,
        'student_response': "There are certain types of diseases and one of the types of diseases that exists is the type of disease in which someone is given to genetically. They are born with said disease, and this type of disease is classified as a congenital disease.",
    },
})

# 17. Detailed explanation with multiple connections -- expect no retry
scenarios.append({
    'name': 'selfexplanation_detailed_connections',
    'task': 'selfexplanation',
    'is_retry': False,
    'data': {
        'target_sentence': "The blood becomes purplish, and the baby's skin looks blue.",
        'context': HD_CONTEXT,
        'student_response': "When a baby's blood does not recieve the amount of oxygen it needs, both the blood and the skin color of the baby becomes affected. Since there is an insufficient amount of oxygen in the blood, the baby's body fails to eliminate carbon dioxide and as a result, the colors of the blood and skin change to abnormal shade of purple and blue, respectively.",
    },
})

# 18. Poor / no effort -- expect retry with paraphrase
scenarios.append({
    'name': 'selfexplanation_poor_no_effort',
    'task': 'selfexplanation',
    'is_retry': False,
    'data': {
        'target_sentence': "Hemoglobin also contains iron, which gives blood its red color.",
        'context': RBC_CONTEXT,
        'student_response': "ok",
    },
})

# 19. Poor / no effort as retry -- should NOT get try-again prompt
scenarios.append({
    'name': 'selfexplanation_poor_no_effort_retry',
    'task': 'selfexplanation',
    'is_retry': True,
    'data': {
        'target_sentence': "Hemoglobin also contains iron, which gives blood its red color.",
        'context': RBC_CONTEXT,
        'student_response': "I don't know",
    },
})

# 20. Off-topic response -- expect retry with paraphrase
scenarios.append({
    'name': 'selfexplanation_offtopic',
    'task': 'selfexplanation',
    'is_retry': False,
    'data': {
        'target_sentence': "Red blood cells live for about three or four months before being broken down in the spleen.",
        'context': RBC_CONTEXT,
        'student_response': "I went to the park yesterday and saw a dog.",
    },
})

# 21. Gibberish response -- expect retry with paraphrase
scenarios.append({
    'name': 'selfexplanation_gibberish',
    'task': 'selfexplanation',
    'is_retry': False,
    'data': {
        'target_sentence': "Therefore, most oxygen is carried by hemoglobin.",
        'context': RBC_CONTEXT,
        'student_response': "asjdkasjd ajsdklasd",
    },
})

# 22. Gibberish as retry -- should NOT get try-again prompt
scenarios.append({
    'name': 'selfexplanation_gibberish_retry',
    'task': 'selfexplanation',
    'is_retry': True,
    'data': {
        'target_sentence': "Therefore, most oxygen is carried by hemoglobin.",
        'context': RBC_CONTEXT,
        'student_response': "blah blah blah",
    },
})

# 23. Minimal but on-topic -- borderline
scenarios.append({
    'name': 'selfexplanation_minimal_ontopic',
    'task': 'selfexplanation',
    'is_retry': False,
    'data': {
        'target_sentence': "The most common heart problem is a heart attack, or coronary thrombosis, which is caused when a coronary artery becomes blocked.",
        'context': HD_CONTEXT,
        'student_response': "Heart attacks happen when arteries get blocked.",
    },
})

# 24. Good self-explanation on heart disease context -- expect no retry
scenarios.append({
    'name': 'selfexplanation_good_hd',
    'task': 'selfexplanation',
    'is_retry': False,
    'data': {
        'target_sentence': "The blockage of a coronary artery is usually caused by a thrombus, or blood clot.",
        'context': HD_CONTEXT,
        'student_response': "A blood clot, also called a thrombus, blocks the coronary artery. This connects to the earlier point about coronary thrombosis being the most common heart problem, because if the artery supplying the heart with oxygen gets blocked, the heart muscle cannot work properly.",
    },
})

# 25. Copy-paste of the target sentence -- poor self-explanation
scenarios.append({
    'name': 'selfexplanation_copypaste',
    'task': 'selfexplanation',
    'is_retry': False,
    'data': {
        'target_sentence': "Usually the heart recovers, but the heart valves are left with scars.",
        'context': HD_CONTEXT,
        'student_response': "Usually the heart recovers, but the heart valves are left with scars.",
    },
})

# 26. Improved retry after poor first attempt -- good content on retry
scenarios.append({
    'name': 'selfexplanation_improved_retry',
    'task': 'selfexplanation',
    'is_retry': True,
    'data': {
        'target_sentence': "Hemoglobin also contains iron, which gives blood its red color.",
        'context': RBC_CONTEXT,
        'student_response': "The protein hemoglobin has iron in it, and that iron is what makes our blood look red. This connects to the earlier idea that hemoglobin binds oxygen, so the iron must play a role in that binding process too.",
    },
})

# ---------------------------------------------------------------------------
# Run all scenarios and collect results
# ---------------------------------------------------------------------------

results = []

for scenario in scenarios:
    result = llm_scoring.feedback(
        scenario['data'],
        scenario['task'],
        is_retry=scenario['is_retry'],
    )
    results.append({
        'name': scenario['name'],
        'task': scenario['task'],
        'is_retry': scenario['is_retry'],
        'student_response': scenario['data']['student_response'],
        'target_sentence': scenario['data'].get('target_sentence', ''),
        'scores': result['scores'],
        'feedback': result['feedback'],
        'try_again': result['try_again'],
    })

with open('feedback_scenarios_results.json', 'w') as f:
    json.dump(results, f, indent=4)
