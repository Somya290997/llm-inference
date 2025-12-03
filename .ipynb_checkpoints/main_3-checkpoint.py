from runtime_engine.runtime import Runtime
import time
# from cpu_kv_manager.cpu_kv_blockmanager import CPUKVBlockManager
runtime = Runtime()

req_id1 = 1
prompt1 = '''Tell me something facts about India?'''

req_id2 = 2
prompt2 = ''' Can you Summarise the give paragraphs in a in few sentences ? Mahendra Singh Dhoni (born 7 July 1981) is an Indian professional cricketer who plays as a right-handed batter and a wicket-keeper. Widely regarded as one of the most prolific wicket-keeper batsmen and captains, he represented the Indian cricket team and was the captain of the side in limited overs formats from 2007 to 2017 and in test cricket from 2008 to 2014. Dhoni has captained the most international matches and is the most successful Indian captain. He has led India to victory in the 2007 ICC World Twenty20, the 2011 Cricket World Cup, and the 2013 ICC Champions Trophy, being the only captain to win three different limited overs ICC tournaments. He also led the teams that won the Asia Cup in 2010 and 2016, and he was a member of the title winning squad in 2018.

Born in Ranchi, Dhoni made his first class debut for Bihar in 1999. He made his debut for the Indian cricket team on 23 December 2004 in an ODI against Bangladesh and played his first test a year later against Sri Lanka. In 2007, he became the captain of the ODI side before taking over in all formats by 2008. Dhoni retired from test cricket in 2014 but continued playing in limited overs cricket till 2019. He has scored 17,266 runs in international cricket including 10,000 plus runs at an average of more than 50 in ODIs.

In the Indian Premier League (IPL), Dhoni plays for Chennai Super Kings (CSK), leading them to the final on ten occasions and winning it five times (2010, 2011, 2018, 2021 and 2023) jointly sharing this title with Rohit Sharma . He has also led CSK to two Champions League T20 titles in 2010 and 2014. Dhoni is among the few batsmen to have scored more than five thousand runs in the IPL, as well as being the first wicket-keeper to do so.'''

req_id3 = 3
prompt3 = ''' Can you summarise this paragrapgh, Modern large-scale AI systems have rapidly evolved into complex, distributed computational pipelines that require careful engineering to achieve both speed and accuracy. Over the past decade, the field has shifted from single-machine training to massively parallelized training strategies that involve sophisticated orchestration across GPUs, TPUs, and heterogeneous accelerators. These systems not only need to support multi-billion-parameter models, but must also optimize communication overhead between devices, manage sharded tensors efficiently, and continuously track memory fragmentation, IO bottlenecks, and latency spikes. As models have grown larger, inference itself has become a significant engineering challenge.'''


# runtime.submit_request(req_id=req_id1,prompt=prompt1)
runtime.submit_request(req_id=req_id2,prompt=prompt1)

time.sleep(30)

runtime.submit_request(req_id=210,prompt=prompt2)
runtime.submit_request(req_id=211,prompt="Can you tell something about Norway ?")

runtime.submit_request(req_id=212,prompt=prompt3)
runtime.submit_request(req_id=213,prompt="Can you tell something about Sweden ?")

# time.sleep(10)

# runtime.submit_request(req_id=212,prompt=prompt2)

# time.sleep(10)

# runtime.submit_request(req_id=213,prompt=prompt2)

# time.sleep(10)

# runtime.submit_request(req_id=214,prompt=prompt2)

# runtime.submit_request(req_id=req_id3,prompt=prompt3)

while True:
    time.sleep(1)