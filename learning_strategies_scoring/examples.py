from api_llm_scoring import LLMScoring

llm_scoring = LLMScoring('upb-nlp/qwen3_4b_scoring_all_tasks_with_se_improved', feedback_model_name='Qwen/Qwen3-4B-Instruct-2507')

RBC_CONTEXT = "Red blood cells have the vital role of carrying oxygen to all of the cells in the body.  They also pick up waste carbon dioxide for removal.  These cells are the most numerous of the blood cells.  The disk shape of red blood cells results in a large surface area, which enables them to be efficient at gas diffusion.\nRed blood cells contain a large, complex protein called hemoglobin.  Hemoglobin binds to the oxygen and carbon dioxide that the red blood cells transport.  Each red blood cell contains about 250 million hemoglobin molecules, each carrying four molecules of oxygen.  Hemoglobin also contains iron, which gives blood its red color.  Molecular oxygen can also be transported by another route, dissolved in blood plasma. However, oxygen is poorly soluble in water, so only about 1.5% is carried in dissolved form.  Therefore, most oxygen is carried by hemoglobin.\nRed blood cells lack a nucleus and the organelles found in other cells.  Therefore, these cells cannot reproduce or repair themselves.  Red blood cells live for about three or four months before being broken down in the spleen.  Iron from the broken-down cells is returned to the bone marrow to be recycled into new hemoglobin.\nSometimes blood does not transport enough oxygen, resulting in a condition called anemia. This makes a person feel tired and weak.  Anemia can result from too little iron in the diet, loss of blood due to injury or menstruation, or various medical conditions.  One type of anemia, called sickle-cell disease, is characterized by red blood cells that are sickle-shaped instead of disk-shaped.  The shape of the cells causes them to clog blood vessels, preventing oxygen from reaching muscles and other tissues."

HD_CONTEXT = "The heart is the hardest-working organ in the living body. Any disorder that terminates the body's blood supply is a threat to life. More people are killed every year in the U.S. by heart disease than by any other disease.  A congenital disease is one with which a person is born. Most babies are born with perfect hearts, but something can go wrong for approximately one in 200 cases. Sometimes a valve develops the incorrect shape causing it to be too tight or fail to close properly. Sometimes a gap is left in the septal wall between the two sides of the heart. When a baby's heart is badly formed, it cannot work efficiently.  The baby's blood does not receive enough oxygen and cannot eliminate carbon dioxide through the lungs. The blood becomes purplish, and the baby's skin looks blue. The baby is in danger of suffocating. Diseases also cause the heart to form improperly. For example, the disease called rheumatic fever follows a sore throat caused by bacteria called streptococci. The tissues of the heart become inflamed and, if badly affected, can cause it to stop. Usually the heart recovers, but the heart valves are left with scars. Years later, they may fail to work properly and cause the heart to stop. The most common heart problem is a heart attack, or coronary thrombosis, which is caused when a coronary artery becomes blocked. The blood vessels that extend across the heart and supply it with blood are called the coronary arteries. They give the heart the oxygen it needs to carry on working. The blockage of a coronary artery is usually caused by a thrombus, or blood clot. Whether heart disease is congenital, caused by other diseases, or the result of a blood clot, it is a very serious problem that requires medical attention."

# ===== Self-explanation examples =====

# Self-explanation: good response with bridging and elaboration (expected overall ~2)
se_data_1 = {
    'target_sentence': "Sometimes blood does not transport enough oxygen, resulting in a condition called anemia.",
    'context': RBC_CONTEXT,
    'student_response': "You develop a condition because you did not have enough oxygen transported, which could mean your red blood cells aren't functioning right.",
}

print("=== Self-explanation 1 (good, expected overall ~2) ===")
scores = llm_scoring.score(se_data_1, 'selfexplanation')
print(scores)

# Self-explanation: basic paraphrase only, no bridging (expected overall ~1)
se_data_2 = {
    'target_sentence': "A congenital disease is one with which a person is born.",
    'context': HD_CONTEXT,
    'student_response': "There are certain types of diseases and one of the types of diseases that exists is the type of disease in which someone is given to genetically. They are born with said disease, and this type of disease is classified as a congenital disease.",
}

print("\n=== Self-explanation 2 (basic paraphrase, expected overall ~1) ===")
scores = llm_scoring.score(se_data_2, 'selfexplanation')
print(scores)

# Self-explanation: excellent with strong bridging (expected overall ~3)
se_data_3 = {
    'target_sentence': "The shape of the cells causes them to clog blood vessels, preventing oxygen from reaching muscles and other tissues.",
    'context': RBC_CONTEXT,
    'student_response': "Blood vessels are naturally shaped to transport the disk shaped red blood cell, if the blood cell changes shape it makes sense to say how it will clog the vessels considering the vessels are already shaped for disk shaped blood cells.",
}

print("\n=== Self-explanation 3 (excellent bridging, expected overall ~3) ===")
scores = llm_scoring.score(se_data_3, 'selfexplanation')
print(scores)

# Self-explanation: detailed explanation with multiple connections (expected overall ~2)
se_data_4 = {
    'target_sentence': "The blood becomes purplish, and the baby's skin looks blue.",
    'context': HD_CONTEXT,
    'student_response': "When a baby's blood does not recieve the amount of oxygen it needs, both the blood and the skin color of the baby becomes affected. Since there is an insufficient amount of oxygen in the blood, the baby's body fails to eliminate carbon dioxide and as a result, the colors of the blood and skin change to abnormal shade of purple and blue, respectively.",
}

print("\n=== Self-explanation 4 (detailed, expected overall ~2) ===")
scores = llm_scoring.score(se_data_4, 'selfexplanation')
print(scores)

# ===== Paraphrasing examples =====

# Paraphrasing: decent restatement (expected mid-range scores)
para_data_1 = {
    'target_sentence': "Sometimes blood does not transport enough oxygen, resulting in a condition called anemia.",
    'student_response': "anemia is caused when the blood doesn't transport enough oxygen.",
}

print("\n=== Paraphrasing 1 (decent, expected mid-range) ===")
scores = llm_scoring.score(para_data_1, 'paraphrasing')
print(scores)

# Paraphrasing: garbage input (expected very low scores)
para_data_2 = {
    'target_sentence': "Red blood cells have the vital role of carrying oxygen to all of the cells in the body.",
    'student_response': ",m.",
}

print("\n=== Paraphrasing 2 (garbage, expected very low) ===")
scores = llm_scoring.score(para_data_2, 'paraphrasing')
print(scores)

# Paraphrasing: good quality with syntactic change (expected high scores)
para_data_3 = {
    'target_sentence': "One of the most harmful air pollutants is acid rain, a mixture of acid and water that falls to earth.",
    'student_response': "a combination of acid and water that fall upon the ground is a harmful polluntant called acid rain.",
}

print("\n=== Paraphrasing 3 (good quality, expected high) ===")
scores = llm_scoring.score(para_data_3, 'paraphrasing')
print(scores)

# Paraphrasing: poor attempt, incomplete meaning (expected low scores)
para_data_4 = {
    'target_sentence': "Over two thirds of heat generated by a resting human is created by organs of the thoracic and abdominal cavities and the brain.",
    'student_response': "2/3 OF HUMANS REST.",
}

print("\n=== Paraphrasing 4 (poor attempt, expected low) ===")
scores = llm_scoring.score(para_data_4, 'paraphrasing')
print(scores)

# ===== Feedback examples =====

# Feedback on excellent self-explanation
print("\n=== Feedback: excellent self-explanation ===")
result = llm_scoring.feedback(se_data_3, 'selfexplanation')
print("Scores:", result['scores'])
print("Feedback:", result['feedback'])

# Feedback on poor paraphrasing
print("\n=== Feedback: poor paraphrasing ===")
result = llm_scoring.feedback(para_data_4, 'paraphrasing')
print("Scores:", result['scores'])
print("Feedback:", result['feedback'])

# Feedback on poor paraphrasing
print("\n=== Feedback: poor paraphrasing ===")
result = llm_scoring.feedback(para_data_2, 'paraphrasing')
print("Scores:", result['scores'])
print("Feedback:", result['feedback'])

# Feedback on decent paraphrasing
print("\n=== Feedback: decent paraphrasing ===")
result = llm_scoring.feedback(para_data_3, 'paraphrasing')
print("Scores:", result['scores'])
print("Feedback:", result['feedback'])