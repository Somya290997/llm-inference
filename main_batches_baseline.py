from runtime_engine.runtime_baseline_2_GPU_inference import Runtime
import time

runtime = Runtime()

prompt1 = '''Tell me something facts about India?'''
prompt2 = '''Tell me something facts about Antratica do people live there?'''
prompt3 = '''Tell me something facts about America and is guns common there?'''
prompt4 = '''Tell me something facts about Asia and do people love living there?'''
prompt5 = '''Tell me something facts about Africa?'''

prompt6 = '''Why moon?'''
prompt7 = '''Do animals have feelings like humans do?'''
prompt8 = '''Explain why some people enjoy spicy food while others cannot tolerate it at all.'''

prompt9 = '''How do airplanes stay in the air using the concepts of lift, thrust, drag, and gravity?
Explain these four forces in simple terms that a school student can understand, and give an example of each.'''

prompt10 = '''What are some of the biggest challenges that countries face when trying to switch 
completely to renewable energy sources like solar, wind, hydro, and nuclear power? 
Explain economic problems, political resistance, and technological limitations, but also 
mention the advantages.'''

prompt11 = '''In the future, will artificial intelligence become more creative than humans?
Discuss this by comparing AI-generated art, music, and writing with human creativity.
Mention current limitations, possible breakthroughs, ethical concerns, and whether AI 
should be considered an artist or just a tool.'''

prompt12 = '''What would happen if every country in the world suddenly banned physical cash 
and moved entirely to digital currency? Describe the potential benefits such as faster 
transactions and reduced corruption, but also the major risks like cyber-attacks, 
loss of privacy, exclusion of older people, and problems during power/internet failures. 
Explain how governments, banks, and technology companies might react to such a drastic change.'''

prompt13 = '''Imagine a fictional world where humans have colonized Mars successfully. 
Describe how cities would be built, how food would be grown, how water would be collected, 
how power would be generated, and how people would travel between Mars and Earth.
Also describe how life would change psychologically: would people feel isolated?
Would they develop a new Martian culture? Would there be Martian-born citizens?
Give at least three different life scenarios of people living on Mars.'''

prompt14 = '''Write a detailed comparison between how humans learn languages and how 
large language models like GPT learn languages. Discuss brain plasticity, childhood learning, 
context understanding, grammar rules, neural embeddings, massive datasets, tokenization, 
self-attention, and probability-based word prediction. Explore the philosophical question: 
if machines can generate perfect language, do they truly "understand" it?'''

prompt15 = '''Assume you are designing a completely new education system for the entire world.
You have full power to redesign exams, assignments, grading system, and curriculum.
The goal is to reduce stress on students but also make learning deeper and more skill-based.
How would you redesign schools? Would there be AI tutors? Would classes be personalized?
Would there still be teachers? Explain your full model including technology, psychology, 
economics, teacher training, student motivation, examination styles, internships, 
practical learning, and how creativity would be encouraged.'''

prompt16 = '''Explain in detail how the human immune system works from birth to adulthood.
Describe innate immunity, adaptive immunity, T cells, B cells, antibodies, memory cells,
vaccination, herd immunity, inflammation, and autoimmune diseases. Use a story-based explanation:
follow a single human body as it faces different challenges at different ages, such as infection
from bacteria, viruses, allergies, vaccine response, and healing from wounds. Compare this system 
to cybersecurity: firewalls, detection systems, databases, recovery mechanisms, and patch updates.'''

prompt17 = '''Write a documentary-style explanation about the history of mathematics from ancient 
civilizations to modern AI. Cover ancient Egyptians, Babylonians, Greeks, Indian mathematicians like 
Aryabhata and Bhaskara, Arabic mathematicians who preserved Greek texts, the European Renaissance, 
Newton and Leibniz inventing calculus, Einstein using tensors, Alan Turing's early work on computing,
DeepMind solving protein folding, and modern AI researchers using mathematics to build LLMs. 
Make it engaging like a Netflix documentary.'''

prompt18 = '''Design a futuristic smart city powered by renewable energy, AI-driven traffic 
management, autonomous public transport, underground waste-processing, rooftop farming, 
and drone-based delivery systems. Describe the architecture, economy, job roles, healthcare, 
education, mental health systems, policing, and ethics for citizens. Explain how data would be 
collected responsibly without violating privacy. Mention how the city handles emergencies, 
pandemics, natural disasters, and cyber threats.'''

prompt19 = '''Create a deep philosophical debate between two fictional characters: one believes 
that humans should merge with AI (cyborg enhancement, brain chips, memory upgrades, robot bodies, 
and AI consciousness), while the other believes that humans should avoid technological dependence 
and live naturally. Let them argue with scientific, ethical, emotional, and spiritual points. 
End with an open question for the reader: What truly defines humanity?'''

prompt20 = '''Write a full research-style essay titled “The Future of Humanity in the Age of Artificial Intelligence”.
Structure it with headings:
1. Introduction
2. Rapid Growth of AI: From Tools to Partners
3. Economic Shifts – Job Creation, Job Loss, New Industries
4. Education Revolution – Fully Personalized AI Tutors
5. Medicine and Longevity – AI Doctors and Gene Editing
6. Ethical Dilemmas – Surveillance, Bias, Weaponization, Privacy
7. AI in Space Exploration – Colonizing Mars & Beyond
8. The Philosophical Question – What is Consciousness?
9. AI Rights – Should Advanced AI Have Legal Protection?
10. A Possible Future Timeline (2025–2100)
11. Human–AI Collaboration Models
12. What It Means to Be Human
13. Conclusion: Hope or Fear?

Make it rich and detailed like a proper research paper, around 2000+ tokens.'''



runtime.submit_request(prompt1)
runtime.submit_request(prompt2)
runtime.submit_request(prompt3)
# runtime.submit_request(prompt4)
# runtime.submit_request(prompt5)
# runtime.submit_request(prompt6)
# runtime.submit_request(prompt7)
# runtime.submit_request(prompt8)
# runtime.submit_request(prompt9)
# runtime.submit_request(prompt10)
# runtime.submit_request(prompt11)
# runtime.submit_request(prompt12)
# runtime.submit_request(prompt13)
# runtime.submit_request(prompt14)
# runtime.submit_request(prompt15)
# runtime.submit_request(prompt16)
# runtime.submit_request(prompt17)
# runtime.submit_request(prompt18)
# runtime.submit_request(prompt19)
# runtime.submit_request(prompt20)




while True:
    time.sleep(15)
    runtime.compute_global_metrics(3)