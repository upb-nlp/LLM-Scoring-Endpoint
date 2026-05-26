"""Send the feedback scenarios to the LLM-Scoring-Endpoint server.

Usage:
    python request_feedback_server.py                          # hit the public endpoint
    python request_feedback_server.py --url http://localhost:5001
    python request_feedback_server.py --only paraphrasing      # filter by task
    python request_feedback_server.py --name paraphrasing_garbage_retry  # run a single scenario
    python request_feedback_server.py --retries-only           # only run retry scenarios
    python request_feedback_server.py --output results.json    # also write structured results

Requires the `requests` package: pip install requests
"""

import argparse
import json
import sys
import time

import requests


DEFAULT_URL = "https://chat.readerbench.com"

RBC_CONTEXT = "Red blood cells have the vital role of carrying oxygen to all of the cells in the body.  They also pick up waste carbon dioxide for removal.  These cells are the most numerous of the blood cells.  The disk shape of red blood cells results in a large surface area, which enables them to be efficient at gas diffusion.\nRed blood cells contain a large, complex protein called hemoglobin.  Hemoglobin binds to the oxygen and carbon dioxide that the red blood cells transport.  Each red blood cell contains about 250 million hemoglobin molecules, each carrying four molecules of oxygen.  Hemoglobin also contains iron, which gives blood its red color.  Molecular oxygen can also be transported by another route, dissolved in blood plasma. However, oxygen is poorly soluble in water, so only about 1.5% is carried in dissolved form.  Therefore, most oxygen is carried by hemoglobin.\nRed blood cells lack a nucleus and the organelles found in other cells.  Therefore, these cells cannot reproduce or repair themselves.  Red blood cells live for about three or four months before being broken down in the spleen.  Iron from the broken-down cells is returned to the bone marrow to be recycled into new hemoglobin.\nSometimes blood does not transport enough oxygen, resulting in a condition called anemia. This makes a person feel tired and weak.  Anemia can result from too little iron in the diet, loss of blood due to injury or menstruation, or various medical conditions.  One type of anemia, called sickle-cell disease, is characterized by red blood cells that are sickle-shaped instead of disk-shaped.  The shape of the cells causes them to clog blood vessels, preventing oxygen from reaching muscles and other tissues."

HD_CONTEXT = "The heart is the hardest-working organ in the living body. Any disorder that terminates the body's blood supply is a threat to life. More people are killed every year in the U.S. by heart disease than by any other disease.  A congenital disease is one with which a person is born. Most babies are born with perfect hearts, but something can go wrong for approximately one in 200 cases. Sometimes a valve develops the incorrect shape causing it to be too tight or fail to close properly. Sometimes a gap is left in the septal wall between the two sides of the heart. When a baby's heart is badly formed, it cannot work efficiently.  The baby's blood does not receive enough oxygen and cannot eliminate carbon dioxide through the lungs. The blood becomes purplish, and the baby's skin looks blue. The baby is in danger of suffocating. Diseases also cause the heart to form improperly. For example, the disease called rheumatic fever follows a sore throat caused by bacteria called streptococci. The tissues of the heart become inflamed and, if badly affected, can cause it to stop. Usually the heart recovers, but the heart valves are left with scars. Years later, they may fail to work properly and cause the heart to stop. The most common heart problem is a heart attack, or coronary thrombosis, which is caused when a coronary artery becomes blocked. The blood vessels that extend across the heart and supply it with blood are called the coronary arteries. They give the heart the oxygen it needs to carry on working. The blockage of a coronary artery is usually caused by a thrombus, or blood clot. Whether heart disease is congenital, caused by other diseases, or the result of a blood clot, it is a very serious problem that requires medical attention."

CLIMATE_MULTITEXT_CONTEXT = (
    "Text #1 of 4: Source A: Natural and Manmade Global Warming Causes\n"
    "Author(s): Patrick Heskestad, Kevin Lerstad, and Hanna Liebich\n"
    "Source: Text obtained from a High School textbook for nature studies: Cosmos; Routledge\n"
    "Date: May 4, 2004\n"
    "The earth's climate has always changed over time. Such climate changes have until recently had natural causes such as changes in the strength of the sun, changes in the earth's orbit around the sun, and volcanic eruptions. It now appears that for the first time mankind is facing a global climate change caused by its own activities.\n"
    "The greenhouse effect is primarily a natural and necessary process.\n"
    "The sun has a surface temperature of approximately 6,000 °C and emits various kinds of radiation. Half of the sunrays that hit the earth's atmosphere penetrate down to the surface of the earth, the rest are reflected by clouds and other gases. Most of the sunrays that reach the earth have short wavelengths. They warm the surface of the earth, which sends back long wavelength streams of heat. A large proportion of these streams returned from the earth are absorbed by the clouds and the gases in the atmosphere, which then send the radiated heat back to us.\n"
    "Some of these gases in the atmosphere are called climate gases. The most important climate gases are water vapour, carbon dioxide and methane. They form a heat shield that slows down the radiation of heat from the earth. This results in the surface of the earth and the air layer being heated up. This is the same that takes place in a greenhouse where sunlight penetrates the glass panes, but radiated heat is restrained on its way out. The result is that the greenhouse is warmer than its surroundings. Without this natural greenhouse effect the average temperature on earth would be -18 °C instead of the 15 °C it is today.\n"
    "In recent times, climate researchers have found that the earth's average temperature rose by approximately 0.5 °C between 1850 and 2004. From around 1900 until the present day the level of carbon dioxide in the air has increased from less than 0.03% to almost 0.04%, and it appears that this increase is continuing. This is due to the fact that we have increased our discharges of CO2 into the atmosphere through the burning of large quantities of oil, gas and coal.\n"
    "Human activities have also resulted in increased discharges of other climate gases.\n"
    "This can result in more of the heat being stopped from escaping from the earth and the average temperature rising even more.\n"
    "Text #2 of 4: Source B: Greenhouse Gases: Manmade Causes\n"
    "Author(s): Professor Martha Olsen\n"
    "Source: Center for International Climate and Environmental Research - Vanderbilt University\n"
    "Date: February 8, 2005\n"
    "The UN's climate panel concludes in its third main report from 2001 that it is highly probable that manmade discharges of climate gases have contributed significantly to the climate changes observed in the last 30 to 50 years.\n"
    "Since pre-industrial times (around 1750) the concentration of carbon dioxide (CO2) has increased by around 31 per cent, the concentration of methane (CH4) has increased by around 151 percent and the concentration of nitrogen oxide (N2O) has increased by around 17 percent. These increases are due to manmade discharges and have resulted in a stronger greenhouse effect. Human activities have also introduced into the atmosphere smaller quantities of a number of climate gases that do not exist in the atmosphere naturally.\n"
    "The increase in the concentration of CO2 in the atmosphere forms the primary constituent (around 60%) of the strengthening of the greenhouse effect for which mankind is responsible. These manmade discharges of CO2 are first and foremost due to the consumption of fossil fuels (coal, oil and gas) and the deforestation of tropical regions.\n"
    "Mankind's discharges amount to only a small part of the quantity of climate gases released into the atmosphere and the effect is minor in relation to, for example, the effect of naturally occurring water vapour. The problem is that the climate system is very complex and sensitive, and even small changes in the system can trigger major consequences. Nature's own discharges of climate gases form part of a cycle in which, for example, rotting trees release CO2 and living trees absorb CO2 through photosynthesis. Our CO2 discharges from, among other things, the burning of fossil fuels do not form part of this cycle and result in surplus CO2 which remains in the atmosphere for a long time.\n"
    "Text (#3 / 4) Title: Source C: Disastrous Side Effects from Global Warming\n"
    "Author(s): Journalist Gustav Jensen\n"
    "Source: Online Periodical: National Enquiry\n"
    "Date: January 16, 2005\n"
    "Stronger storms, more hurricanes and increasingly tumultuous weather are just a few of the negative consequences we can expect in the next few years. Global warming may also weaken the Gulf Stream and result in serious cooling in Northern Europe.\n"
    "A number of oceanographers fear highly uncomfortable side effects due to global warming. It may weaken the ocean currents in the North Atlantic to such a degree that there is a genuine risk of serious and long-term cooling both in the Nordic Region and large parts of Europe and North America. The Nordic Region would be significantly colder without the Gulf Stream.\n"
    "Oceanographers know all to well that the warnings will cause surprise because we are reminded almost daily of the opposite, namely that global warming will raise the earth's average temperature. However, paradoxically, both things could well occur at the same time.\n"
    "If the circulation of the Atlantic is disturbed, we could have a fall in the average temperature of 3-5 °C. This will have a dramatic effect on farming and forestry, while at the same time there will be a greater need for heating.\n"
    "And there is much that indicates that the disturbances are well underway. More ice is melting due to global warming and more precipitation is falling over Russia, among other places. This is resulting in greater outward flows of freshwater from the major Russian rivers into the Arctic Ocean. At the same time, we risk losing the Western Arctic ice and Greenland ice.\n"
    "When the ice surrounding the poles melts, this will not just result in an increased mass of water, it will also result in increased evaporation from the oceans. This will provide hurricanes with energy. Time magazine reports that hurricanes have increased in both number and intensity since 1995.\n"
    "According to the UN's climate panel, an increased greenhouse effect resulted in water levels rising between 10 and 20 cm in the last century and by 2100 ocean levels will rise by between 9 and 88 cm. This will be catastrophic for many coastal communities - especially in developing countries.\n"
    "Text (#4 / 4) Title: Source D: Polar Exploration and Global Warming\n"
    "Author(s): Journalist John Hultgren\n"
    "Source: Online Periodical: Bay Area Times\n"
    "Date: October 13, 2004\n"
    "Regions that are now becoming accessible due to global warming conceal enormous riches. The melting of the ice permits the exploitation of resources in the northerly regions.\n"
    "Temperatures around the North Pole are increasing at double the rate of other places around the globe according to UN experts. The Arctic ice is melting so quickly that a sea passage between the Atlantic Ocean and the Pacific Ocean may be accessible to ordinary ships during the summer by 2050. The route through the Northwest Passage to Asia will reduce the journey distance between London and Tokyo from 21,000 to 16,000 kilometres.\n"
    "The northerly regions that are becoming accessible also conceal enormous riches. The oil and gas deposits that are concealed there are estimated to amount to 30 per cent of the earth's deposits.\n"
    "And there is more to be found in the northerly regions than petroleum. There is also gold, diamonds, copper and zinc. There will be a lot of traffic due to such exploration says Frederic Lasserre, a geographer at Laval University in Quebec in Canada who is a specialist in Arctic regions.\n"
    "The director of the Nansen Environmental and Remote Sensing Center, also points out positive consequences of global warming, which occurs in the Arctic in particular: A warmer climate could result in better growing conditions and lower heating costs. The ice in the Barents Sea will be pushed northwards and eastwards due to increasing south-westerly winds and warmer weather. This will expand winter fishing grounds and make it easier for the gas and oil industry to operate during the winter season."
)


# ---------------------------------------------------------------------------
# Reusable "previous attempt" payloads for retry scenarios.
#
# Each retry scenario passes a `previous_answer` dict to the /feedback/<task>
# endpoint containing the student's prior response, the scores it received, and
# the feedback they were shown. The server uses this to emit a one-sentence
# "evolution" comment comparing the two attempts.
# ---------------------------------------------------------------------------

PREV_PARA_GARBAGE = {
    "student_response": ",m.",
    "scores": {
        "Garbage Content": "Too much",
        "Frozen Expressions": "Not present",
        "Irrelevant": "Irrelevant",
        "Elaboration": "Not present/Poor",
        "Semantic Completeness": "Not present/Poor",
        "Entailment": "Not present/Poor",
        "Syntactic Similarity": "Not present",
        "Lexical Similarity": "Not present",
        "Paraphrase Quality": "Poor",
        "Writing Quality": "Poor",
    },
    "feedback": "Your response is very short and does not appear to address the target sentence. It may help to read the sentence again and write a complete rephrasing in your own words. Can you try again?",
}

PREV_PARA_VERY_POOR = {
    "student_response": "2/3 OF HUMANS REST.",
    "scores": {
        "Garbage Content": "Not present",
        "Frozen Expressions": "Not present",
        "Irrelevant": "Somewhat relevant",
        "Elaboration": "Not present/Poor",
        "Semantic Completeness": "Not present/Poor",
        "Entailment": "Not present/Poor",
        "Syntactic Similarity": "Not present",
        "Lexical Similarity": "Not present",
        "Paraphrase Quality": "Poor",
        "Writing Quality": "Poor",
    },
    "feedback": "Your response appears to drop most of the meaning of the original sentence. It may help to include all the key ideas — what generates the heat and where. Can you try again?",
}

PREV_PARA_RANDOM_CHARS = {
    "student_response": "!!!???@@@###",
    "scores": {
        "Garbage Content": "Too much",
        "Frozen Expressions": "Not present",
        "Irrelevant": "Irrelevant",
        "Elaboration": "Not present/Poor",
        "Semantic Completeness": "Not present/Poor",
        "Entailment": "Not present/Poor",
        "Syntactic Similarity": "Not present",
        "Lexical Similarity": "Not present",
        "Paraphrase Quality": "Poor",
        "Writing Quality": "Poor",
    },
    "feedback": "Your response does not appear to contain any words from the target sentence. It may help to write the rephrasing as a full sentence. Can you try again?",
}

PREV_PARA_IMPROVABLE = {
    "student_response": "Red blood cells lack a nucleus and the organelles in other cells.",
    "scores": {
        "Garbage Content": "Not present",
        "Frozen Expressions": "Not present",
        "Irrelevant": "Relevant",
        "Elaboration": "Not present/Poor",
        "Semantic Completeness": "Good",
        "Entailment": "Good",
        "Syntactic Similarity": "Not present",
        "Lexical Similarity": "Too much",
        "Paraphrase Quality": "Poor",
        "Writing Quality": "Good",
    },
    "feedback": "Your response keeps the meaning but reuses most of the wording. It may help to replace shared words with synonyms and restructure the sentence.",
}

PREV_SE_OK = {
    "student_response": "ok",
    "scores": {
        "Paraphrase presence": "Not present",
        "Lexical change": "Not applicable",
        "Syntactic change": "Not applicable",
        "Misconception": "Not applicable",
        "Monitoring": "Not applicable",
        "Bridge presence": "Not present/Poor",
        "Bridge contribution": "Not applicable",
        "Elaboration presence": "Not present",
        "Life event": "Not present",
        "Overall": "Poor",
    },
    "feedback": "Your response is very short and does not appear to engage with the target sentence. It may help to explain what the sentence means in your own words. Can you try again?",
}

PREV_SE_GIBBERISH = {
    "student_response": "asjdkasjd ajsdklasd",
    "scores": {
        "Paraphrase presence": "Not present",
        "Lexical change": "Not applicable",
        "Syntactic change": "Not applicable",
        "Misconception": "Not applicable",
        "Monitoring": "Not applicable",
        "Bridge presence": "Not present/Poor",
        "Bridge contribution": "Not applicable",
        "Elaboration presence": "Not present",
        "Life event": "Not present",
        "Overall": "Poor",
    },
    "feedback": "Your response does not appear to address the target sentence. It may help to write a sentence or two explaining what it means. Can you try again?",
}

PREV_SE_BORDERLINE = {
    "student_response": "Heart attacks happen when something gets blocked.",
    "scores": {
        "Paraphrase presence": "Present",
        "Lexical change": "Present",
        "Syntactic change": "Present",
        "Misconception": "Not present",
        "Monitoring": "Not applicable",
        "Bridge presence": "Not present/Poor",
        "Bridge contribution": "Not applicable",
        "Elaboration presence": "Not present",
        "Life event": "Not present",
        "Overall": "Satisfactory",
    },
    "feedback": "Your response captures the basic idea but stays close to the surface. It may help to connect this to the role of coronary arteries described earlier in the text.",
}

PREV_SE_MULTITEXT_GIBBERISH = {
    "student_response": "asdfasdf zzzz qwerty",
    "scores": {
        "Too short": "Good",
        "Irrelevant": "Irrelevant",
        "Copy-paste presence": "Not present",
        "Evaluative statements / affect": "Not present",
        "Self-monitoring": "Not present",
        "Paraphrase presence": "Not present",
        "Bridge presence": "Not present",
        "Elaboration presence": "Not present",
        "Inter-textual References": "Not present",
        "Source Presence (Sourcing)": "Not present",
        "Source Evaluation": "Not present",
        "Overall Self-Explanation Quality": "Poor",
    },
    "feedback": "Your response does not appear to address the target sentence. It may help to use details from the sources to explain it. Can you try again?",
}

PREV_SE_MULTITEXT_BORDERLINE = {
    "student_response": "Ice melting will let ships pass through.",
    "scores": {
        "Too short": "Good",
        "Irrelevant": "Average",
        "Copy-paste presence": "Not present",
        "Evaluative statements / affect": "Not present",
        "Self-monitoring": "Not present",
        "Paraphrase presence": "Present",
        "Bridge presence": "Not present",
        "Elaboration presence": "Not present",
        "Inter-textual References": "Not present",
        "Source Presence (Sourcing)": "Not present",
        "Source Evaluation": "Not present",
        "Overall Self-Explanation Quality": "Fair quality",
    },
    "feedback": "Your response paraphrases the main idea but stays brief. It may help to link this to information from the other sources about Arctic shipping routes.",
}


SCENARIOS = [
    # ===================== PARAPHRASING =====================
    {
        "name": "paraphrasing_good_syntactic_change",
        "task": "paraphrasing",
        "previous_answer": None,
        "data": {
            "target_sentence": "One of the most harmful air pollutants is acid rain, a mixture of acid and water that falls to earth.",
            "student_response": "A combination of acid and water that falls upon the ground is a harmful pollutant called acid rain.",
        },
    },
    {
        "name": "paraphrasing_decent",
        "task": "paraphrasing",
        "previous_answer": None,
        "data": {
            "target_sentence": "Sometimes blood does not transport enough oxygen, resulting in a condition called anemia.",
            "student_response": "Anemia is caused when the blood doesn't transport enough oxygen.",
        },
    },
    {
        "name": "paraphrasing_garbage",
        "task": "paraphrasing",
        "previous_answer": None,
        "data": {
            "target_sentence": "Red blood cells have the vital role of carrying oxygen to all of the cells in the body.",
            "student_response": ",m.",
        },
    },
    {
        "name": "paraphrasing_garbage_retry",
        "task": "paraphrasing",
        "previous_answer": PREV_PARA_GARBAGE,
        "data": {
            "target_sentence": "Red blood cells have the vital role of carrying oxygen to all of the cells in the body.",
            "student_response": "asdf asdf",
        },
    },
    {
        "name": "paraphrasing_very_poor",
        "task": "paraphrasing",
        "previous_answer": None,
        "data": {
            "target_sentence": "Over two thirds of heat generated by a resting human is created by organs of the thoracic and abdominal cavities and the brain.",
            "student_response": "2/3 OF HUMANS REST.",
        },
    },
    {
        "name": "paraphrasing_very_poor_retry",
        "task": "paraphrasing",
        "previous_answer": PREV_PARA_VERY_POOR,
        "data": {
            "target_sentence": "Over two thirds of heat generated by a resting human is created by organs of the thoracic and abdominal cavities and the brain.",
            "student_response": "Humans rest a lot and produce heat.",
        },
    },
    {
        "name": "paraphrasing_frozen_copypaste",
        "task": "paraphrasing",
        "previous_answer": None,
        "data": {
            "target_sentence": "The disk shape of red blood cells results in a large surface area, which enables them to be efficient at gas diffusion.",
            "student_response": "The disk shape of red blood cells results in a large surface area, which enables them to be efficient at gas diffusion.",
        },
    },
    {
        "name": "paraphrasing_irrelevant",
        "task": "paraphrasing",
        "previous_answer": None,
        "data": {
            "target_sentence": "Hemoglobin binds to the oxygen and carbon dioxide that the red blood cells transport.",
            "student_response": "I had pizza for lunch and it was delicious.",
        },
    },
    {
        "name": "paraphrasing_partial_frozen",
        "task": "paraphrasing",
        "previous_answer": None,
        "data": {
            "target_sentence": "Red blood cells lack a nucleus and the organelles found in other cells.",
            "student_response": "Red blood cells lack a nucleus and the organelles that other cells have.",
        },
    },
    {
        "name": "paraphrasing_good_meaning_preserved",
        "task": "paraphrasing",
        "previous_answer": None,
        "data": {
            "target_sentence": "Each red blood cell contains about 250 million hemoglobin molecules, each carrying four molecules of oxygen.",
            "student_response": "A single red blood cell holds roughly 250 million hemoglobin molecules, and every one of them transports four oxygen molecules.",
        },
    },
    {
        "name": "paraphrasing_random_chars",
        "task": "paraphrasing",
        "previous_answer": None,
        "data": {
            "target_sentence": "Iron from the broken-down cells is returned to the bone marrow to be recycled into new hemoglobin.",
            "student_response": "!!!???@@@###",
        },
    },
    {
        "name": "paraphrasing_random_chars_retry",
        "task": "paraphrasing",
        "previous_answer": PREV_PARA_RANDOM_CHARS,
        "data": {
            "target_sentence": "Iron from the broken-down cells is returned to the bone marrow to be recycled into new hemoglobin.",
            "student_response": "zzzz zzz zz",
        },
    },
    {
        "name": "paraphrasing_single_word",
        "task": "paraphrasing",
        "previous_answer": None,
        "data": {
            "target_sentence": "Anemia can result from too little iron in the diet, loss of blood due to injury or menstruation, or various medical conditions.",
            "student_response": "anemia",
        },
    },
    {
        "name": "paraphrasing_borderline_low",
        "task": "paraphrasing",
        "previous_answer": None,
        "data": {
            "target_sentence": "Hemoglobin also contains iron, which gives blood its red color.",
            "student_response": "Hemoglobin contains iron, which gives blood its red color.",
        },
    },
    {
        "name": "paraphrasing_improved_retry",
        "task": "paraphrasing",
        "previous_answer": PREV_PARA_IMPROVABLE,
        "data": {
            "target_sentence": "Red blood cells lack a nucleus and the organelles found in other cells.",
            "student_response": "Unlike other cells, red blood cells do not have a nucleus or the small structures normally found inside.",
        },
    },

    # ===================== SELF-EXPLANATION =====================
    {
        "name": "selfexplanation_excellent_bridging",
        "task": "selfexplanation",
        "previous_answer": None,
        "data": {
            "target_sentence": "The shape of the cells causes them to clog blood vessels, preventing oxygen from reaching muscles and other tissues.",
            "context": RBC_CONTEXT,
            "student_response": "Blood vessels are naturally shaped to transport the disk shaped red blood cell, if the blood cell changes shape it makes sense to say how it will clog the vessels considering the vessels are already shaped for disk shaped blood cells.",
        },
    },
    {
        "name": "selfexplanation_good_elaboration",
        "task": "selfexplanation",
        "previous_answer": None,
        "data": {
            "target_sentence": "Sometimes blood does not transport enough oxygen, resulting in a condition called anemia.",
            "context": RBC_CONTEXT,
            "student_response": "You develop a condition because you did not have enough oxygen transported, which could mean your red blood cells aren't functioning right.",
        },
    },
    {
        "name": "selfexplanation_basic_paraphrase",
        "task": "selfexplanation",
        "previous_answer": None,
        "data": {
            "target_sentence": "A congenital disease is one with which a person is born.",
            "context": HD_CONTEXT,
            "student_response": "There are certain types of diseases and one of the types of diseases that exists is the type of disease in which someone is given to genetically. They are born with said disease, and this type of disease is classified as a congenital disease.",
        },
    },
    {
        "name": "selfexplanation_detailed_connections",
        "task": "selfexplanation",
        "previous_answer": None,
        "data": {
            "target_sentence": "The blood becomes purplish, and the baby's skin looks blue.",
            "context": HD_CONTEXT,
            "student_response": "When a baby's blood does not recieve the amount of oxygen it needs, both the blood and the skin color of the baby becomes affected. Since there is an insufficient amount of oxygen in the blood, the baby's body fails to eliminate carbon dioxide and as a result, the colors of the blood and skin change to abnormal shade of purple and blue, respectively.",
        },
    },
    {
        "name": "selfexplanation_poor_no_effort",
        "task": "selfexplanation",
        "previous_answer": None,
        "data": {
            "target_sentence": "Hemoglobin also contains iron, which gives blood its red color.",
            "context": RBC_CONTEXT,
            "student_response": "ok",
        },
    },
    {
        "name": "selfexplanation_poor_no_effort_retry",
        "task": "selfexplanation",
        "previous_answer": PREV_SE_OK,
        "data": {
            "target_sentence": "Hemoglobin also contains iron, which gives blood its red color.",
            "context": RBC_CONTEXT,
            "student_response": "I don't know",
        },
    },
    {
        "name": "selfexplanation_offtopic",
        "task": "selfexplanation",
        "previous_answer": None,
        "data": {
            "target_sentence": "Red blood cells live for about three or four months before being broken down in the spleen.",
            "context": RBC_CONTEXT,
            "student_response": "I went to the park yesterday and saw a dog.",
        },
    },
    {
        "name": "selfexplanation_gibberish",
        "task": "selfexplanation",
        "previous_answer": None,
        "data": {
            "target_sentence": "Therefore, most oxygen is carried by hemoglobin.",
            "context": RBC_CONTEXT,
            "student_response": "asjdkasjd ajsdklasd",
        },
    },
    {
        "name": "selfexplanation_gibberish_retry",
        "task": "selfexplanation",
        "previous_answer": PREV_SE_GIBBERISH,
        "data": {
            "target_sentence": "Therefore, most oxygen is carried by hemoglobin.",
            "context": RBC_CONTEXT,
            "student_response": "blah blah blah",
        },
    },
    {
        "name": "selfexplanation_minimal_ontopic",
        "task": "selfexplanation",
        "previous_answer": None,
        "data": {
            "target_sentence": "The most common heart problem is a heart attack, or coronary thrombosis, which is caused when a coronary artery becomes blocked.",
            "context": HD_CONTEXT,
            "student_response": "Heart attacks happen when arteries get blocked.",
        },
    },
    {
        "name": "selfexplanation_borderline_retry",
        "task": "selfexplanation",
        "previous_answer": PREV_SE_BORDERLINE,
        "data": {
            "target_sentence": "The most common heart problem is a heart attack, or coronary thrombosis, which is caused when a coronary artery becomes blocked.",
            "context": HD_CONTEXT,
            "student_response": "A heart attack — coronary thrombosis — happens when a coronary artery is blocked, so the heart muscle stops getting the oxygen it needs.",
        },
    },
    {
        "name": "selfexplanation_good_hd",
        "task": "selfexplanation",
        "previous_answer": None,
        "data": {
            "target_sentence": "The blockage of a coronary artery is usually caused by a thrombus, or blood clot.",
            "context": HD_CONTEXT,
            "student_response": "A blood clot, also called a thrombus, blocks the coronary artery. This connects to the earlier point about coronary thrombosis being the most common heart problem, because if the artery supplying the heart with oxygen gets blocked, the heart muscle cannot work properly.",
        },
    },
    {
        "name": "selfexplanation_copypaste",
        "task": "selfexplanation",
        "previous_answer": None,
        "data": {
            "target_sentence": "Usually the heart recovers, but the heart valves are left with scars.",
            "context": HD_CONTEXT,
            "student_response": "Usually the heart recovers, but the heart valves are left with scars.",
        },
    },
    {
        "name": "selfexplanation_improved_retry",
        "task": "selfexplanation",
        "previous_answer": PREV_SE_OK,
        "data": {
            "target_sentence": "Hemoglobin also contains iron, which gives blood its red color.",
            "context": RBC_CONTEXT,
            "student_response": "The protein hemoglobin has iron in it, and that iron is what makes our blood look red. This connects to the earlier idea that hemoglobin binds oxygen, so the iron must play a role in that binding process too.",
        },
    },

    # ===================== MULTI-TEXT SELF-EXPLANATION =====================
    {
        "name": "selfexplanation_multitext_basic_paraphrase",
        "task": "selfexplanation_multitext",
        "previous_answer": None,
        "data": {
            "target_sentence": "In recent times, climate researchers have found that the earth's average temperature rose by approx. 0.5 °C between 1850 and 2004.",
            "context": CLIMATE_MULTITEXT_CONTEXT,
            "student_response": "Recently, climate researchers have discovered that the earth's average temperature has risen about 0.5 degrees celsius between 1850 and 2004.",
        },
    },
    {
        "name": "selfexplanation_multitext_intertext",
        "task": "selfexplanation_multitext",
        "previous_answer": None,
        "data": {
            "target_sentence": "Since pre-industrial times (around 1750) the concentration of carbon dioxide (CO2) has increased by around 31 per cent, the concentration of methane (CH4) has increased by around 151 per cent and the concentration of nitrogen oxide (N2O) has increased by around 17 per cent.",
            "context": CLIMATE_MULTITEXT_CONTEXT,
            "student_response": "This is interesting to know! I thought the industrial revolution was the first cause of rising trapped gas levels.",
        },
    },
    {
        "name": "selfexplanation_multitext_strong_elaboration",
        "task": "selfexplanation_multitext",
        "previous_answer": None,
        "data": {
            "target_sentence": "Our CO2 discharges from, among other things, the burning of fossil fuels do not form part of this cycle and result in surplus CO2 which remains in the atmosphere for a long time.",
            "context": CLIMATE_MULTITEXT_CONTEXT,
            "student_response": "Now that there's manmade processes and natural processes both contributing separately to climate change, it is now understandable to comply to ecofriendly activities such as recyling, car pooling, and reducing the amount of trash we expell.",
        },
    },
    {
        "name": "selfexplanation_multitext_source_evaluation",
        "task": "selfexplanation_multitext",
        "previous_answer": None,
        "data": {
            "target_sentence": "Since pre-industrial times (around 1750) the concentration of carbon dioxide (CO2) has increased by around 31 per cent, the concentration of methane (CH4) has increased by around 151 per cent and the concentration of nitrogen oxide (N2O) has increased by around 17 per cent.",
            "context": CLIMATE_MULTITEXT_CONTEXT,
            "student_response": "This text is being argued that it is highly probable that manmade climate gases have contributed significantly to claimed changes. Although this is research conducted about 12 years ago that may not be the same today.",
        },
    },
    {
        "name": "selfexplanation_multitext_offtarget",
        "task": "selfexplanation_multitext",
        "previous_answer": None,
        "data": {
            "target_sentence": "These manmade discharges of CO2 are first and foremost due to the consumption of fossil fuels (coal, oil and gas) and the deforestation of tropical regions.",
            "context": CLIMATE_MULTITEXT_CONTEXT,
            "student_response": "It explains how the increase of carbon dioxide in the atmosphere forms the primary constituent.",
        },
    },
    {
        "name": "selfexplanation_multitext_gibberish",
        "task": "selfexplanation_multitext",
        "previous_answer": None,
        "data": {
            "target_sentence": "Stronger storms, more hurricanes and increasingly tumultuous weather are just a few of the negative consequences we can expect in the next few years.",
            "context": CLIMATE_MULTITEXT_CONTEXT,
            "student_response": "asdfasdf zzzz qwerty",
        },
    },
    {
        "name": "selfexplanation_multitext_gibberish_retry",
        "task": "selfexplanation_multitext",
        "previous_answer": PREV_SE_MULTITEXT_GIBBERISH,
        "data": {
            "target_sentence": "Stronger storms, more hurricanes and increasingly tumultuous weather are just a few of the negative consequences we can expect in the next few years.",
            "context": CLIMATE_MULTITEXT_CONTEXT,
            "student_response": "blah blah",
        },
    },
    {
        "name": "selfexplanation_multitext_minimal",
        "task": "selfexplanation_multitext",
        "previous_answer": None,
        "data": {
            "target_sentence": "The Arctic ice is melting so quickly that a sea passage between the Atlantic Ocean and the Pacific Ocean may be accessible to ordinary ships during the summer by 2050.",
            "context": CLIMATE_MULTITEXT_CONTEXT,
            "student_response": "Ice melting will let ships pass through the Arctic.",
        },
    },
    {
        "name": "selfexplanation_multitext_improved_retry",
        "task": "selfexplanation_multitext",
        "previous_answer": PREV_SE_MULTITEXT_BORDERLINE,
        "data": {
            "target_sentence": "The Arctic ice is melting so quickly that a sea passage between the Atlantic Ocean and the Pacific Ocean may be accessible to ordinary ships during the summer by 2050.",
            "context": CLIMATE_MULTITEXT_CONTEXT,
            "student_response": "Source D says rising Arctic temperatures could open a shipping route between the Atlantic and Pacific by 2050. This fits with Source A's point that human-driven warming is now changing the climate, and Source D adds an economic side — the new passage would cut the London–Tokyo route by thousands of kilometres.",
        },
    },
    {
        "name": "selfexplanation_multitext_copypaste",
        "task": "selfexplanation_multitext",
        "previous_answer": None,
        "data": {
            "target_sentence": "If the circulation of the Atlantic is disturbed, we could have a fall in the average temperature of 3-5 °C.",
            "context": CLIMATE_MULTITEXT_CONTEXT,
            "student_response": "If the circulation of the Atlantic is disturbed, we could have a fall in the average temperature of 3-5 °C.",
        },
    },
]


def send_feedback_request(base_url, task, data, previous_answer, timeout):
    url = f"{base_url.rstrip('/')}/feedback/{task}"
    payload = dict(data)
    if previous_answer is not None:
        payload["previous_answer"] = previous_answer
    response = requests.post(
        url,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def print_scenario(scenario, result, elapsed):
    name = scenario["name"]
    task = scenario["task"]
    is_retry = scenario["previous_answer"] is not None
    student = scenario["data"]["student_response"]
    tag = " [retry]" if is_retry else ""
    print(f"\n=== {name} ({task}){tag} — {elapsed:.2f}s ===")
    if is_retry:
        prev = scenario["previous_answer"].get("student_response", "")
        print(f"  Previous response: {prev!r}")
    print(f"  Student response: {student!r}")
    print(f"  try_again: {result.get('try_again')}")
    print("  Scores:")
    for key, value in result.get("scores", {}).items():
        print(f"    - {key}: {value}")
    print(f"  Feedback: {result.get('feedback', '')}")


def main():
    parser = argparse.ArgumentParser(description="POST feedback scenarios to the LLM-Scoring-Endpoint server.")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Base URL of the server (default: {DEFAULT_URL})")
    parser.add_argument("--only", help="Only run scenarios for this task (e.g. paraphrasing, selfexplanation, selfexplanation_multitext)")
    parser.add_argument("--name", help="Only run the scenario with this exact name")
    parser.add_argument("--retries-only", action="store_true", help="Only run scenarios that include a previous_answer")
    parser.add_argument("--output", help="Optional path to write a JSON file with all results")
    parser.add_argument("--timeout", type=float, default=180.0, help="Request timeout in seconds (default: 180)")
    args = parser.parse_args()

    scenarios = SCENARIOS
    if args.only:
        scenarios = [s for s in scenarios if s["task"] == args.only]
    if args.name:
        scenarios = [s for s in scenarios if s["name"] == args.name]
    if args.retries_only:
        scenarios = [s for s in scenarios if s["previous_answer"] is not None]

    if not scenarios:
        print("No scenarios match the given filters.", file=sys.stderr)
        sys.exit(1)

    print(f"Sending {len(scenarios)} scenario(s) to {args.url}")

    results = []
    n_ok = 0
    n_fail = 0
    for scenario in scenarios:
        t0 = time.time()
        try:
            result = send_feedback_request(
                args.url,
                scenario["task"],
                scenario["data"],
                scenario["previous_answer"],
                args.timeout,
            )
        except requests.RequestException as exc:
            elapsed = time.time() - t0
            print(f"\n=== {scenario['name']} ({scenario['task']}) — {elapsed:.2f}s ===", file=sys.stderr)
            print(f"  ERROR: {exc}", file=sys.stderr)
            if exc.response is not None:
                print(f"  Server response: {exc.response.text}", file=sys.stderr)
            n_fail += 1
            continue
        elapsed = time.time() - t0
        print_scenario(scenario, result, elapsed)
        results.append({
            "name": scenario["name"],
            "task": scenario["task"],
            "is_retry": scenario["previous_answer"] is not None,
            "previous_answer": scenario["previous_answer"],
            "student_response": scenario["data"]["student_response"],
            "target_sentence": scenario["data"].get("target_sentence", ""),
            "scores": result.get("scores", {}),
            "feedback": result.get("feedback", ""),
            "try_again": result.get("try_again"),
        })
        n_ok += 1

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\nWrote {len(results)} result(s) to {args.output}")

    print(f"\nDone. Successful: {n_ok}, Failed: {n_fail}")


if __name__ == "__main__":
    main()
